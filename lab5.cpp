#include <mpi.h>
#include <iostream>

using namespace std;

const int MATRIX_SIZE = 3;
const int LENGTH = MATRIX_SIZE * (MATRIX_SIZE + 1);

void copyArray(float *destination) {
    float predefinedMatrix[LENGTH] = {1, 2, 3, 3, 3, 5, 7, 0, 1, 3, 4, 1};
    for (int i = 0; i < LENGTH; ++i) {
        destination[i] = predefinedMatrix[i];
    }
}

void computeResults(float *matrix, float *computedResults, int processRank, int rowsPerProcess) {
    for (int i = MATRIX_SIZE - 1; i >= 0; i--) {
        computedResults[i] = matrix[i * (MATRIX_SIZE + 1) + MATRIX_SIZE];

        for (int j = i + 1; j < MATRIX_SIZE; j++) {
            computedResults[i] -= matrix[i * (MATRIX_SIZE + 1) + j] * computedResults[j];
        }

        computedResults[i] /= matrix[i * (MATRIX_SIZE + 1) + i];
    }
}

int main(int argc, char *argv[]) {
    int processRank, numProcesses;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    int rowsPerProcess = MATRIX_SIZE / numProcesses;
    float *matrix, *computedResults;
    matrix = new float[LENGTH];

    if (processRank == 0) {
        copyArray(matrix);
        computedResults = new float[MATRIX_SIZE];
    }

    float *partialMatrix = new float[(MATRIX_SIZE + 1) * rowsPerProcess];
    MPI_Scatter(matrix, (MATRIX_SIZE + 1) * rowsPerProcess, MPI_FLOAT, partialMatrix, (MATRIX_SIZE + 1) * rowsPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float *row = new float[MATRIX_SIZE + 1];

    int element, scaleElement, targetColumn, localStartRow;
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        int owner = i / rowsPerProcess;
        if (processRank == owner) {
            int localRow = i % rowsPerProcess;
            element = partialMatrix[localRow * (MATRIX_SIZE + 1) + i];

            for (int j = i; j < MATRIX_SIZE + 1; ++j) {
                partialMatrix[localRow * (MATRIX_SIZE + 1) + j] /= element;
            }

            memcpy(row, &partialMatrix[localRow * (MATRIX_SIZE + 1)], (MATRIX_SIZE + 1) * sizeof(float));
        }

        MPI_Bcast(row, MATRIX_SIZE + 1, MPI_FLOAT, owner, MPI_COMM_WORLD);

        if (processRank != owner) {
            for (int j = 0; j < rowsPerProcess; ++j) {
                int globalRowIdx = processRank * rowsPerProcess + j;
                if (globalRowIdx > i) {
                    scaleElement = partialMatrix[j * (MATRIX_SIZE + 1) + i];
                    for (int k = i; k < MATRIX_SIZE + 1; ++k) {
                        partialMatrix[j * (MATRIX_SIZE + 1) + k] -= scaleElement * row[k];
                    }
                }
            }
        }
    }

    MPI_Gather(partialMatrix, (MATRIX_SIZE + 1) * rowsPerProcess, MPI_FLOAT, matrix, (MATRIX_SIZE + 1) * rowsPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (processRank == 0) {
        computeResults(matrix, computedResults, processRank, rowsPerProcess);
        cout << "Answer:" << endl;
        for (int i = 0; i < 3; i++) {
            cout << computedResults[i] << endl;
        }
    }

    MPI_Finalize();

    return 0;
}

