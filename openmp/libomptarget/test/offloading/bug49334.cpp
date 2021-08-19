// RUN: %libomptarget-compilexx-run-and-check-generic

// Currently hangs on amdgpu
// UNSUPPORTED: amdgcn-amd-amdhsa

// UNSUPPORTED: x86_64-pc-linux-gnu

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

class BlockMatrix {
private:
  const int rowsPerBlock;
  const int colsPerBlock;
  const long nRows;
  const long nCols;
  const int nBlocksPerRow;
  const int nBlocksPerCol;
  std::vector<std::vector<std::unique_ptr<float[]>>> Blocks;

public:
  BlockMatrix(const int _rowsPerBlock, const int _colsPerBlock,
              const long _nRows, const long _nCols)
      : rowsPerBlock(_rowsPerBlock), colsPerBlock(_colsPerBlock), nRows(_nRows),
        nCols(_nCols), nBlocksPerRow(_nRows / _rowsPerBlock),
        nBlocksPerCol(_nCols / _colsPerBlock), Blocks(nBlocksPerCol) {
    for (int i = 0; i < nBlocksPerCol; i++) {
      for (int j = 0; j < nBlocksPerRow; j++) {
        Blocks[i].emplace_back(new float[_rowsPerBlock * _colsPerBlock]);
      }
    }
  };

  // Initialize the BlockMatrix from 2D arrays
  void Initialize(const std::vector<float> &matrix) {
    for (int i = 0; i < nBlocksPerCol; i++)
      for (int j = 0; j < nBlocksPerRow; j++) {
        float *CurrBlock = GetBlock(i, j);
        for (int ii = 0; ii < colsPerBlock; ++ii)
          for (int jj = 0; jj < rowsPerBlock; ++jj) {
            int curri = i * colsPerBlock + ii;
            int currj = j * rowsPerBlock + jj;
            CurrBlock[ii + jj * colsPerBlock] = matrix[curri + currj * nCols];
          }
      }
  }

  long Compare(const std::vector<float> &matrix) const {
    long fail = 0;
    for (int i = 0; i < nBlocksPerCol; i++)
      for (int j = 0; j < nBlocksPerRow; j++) {
        float *CurrBlock = GetBlock(i, j);
        for (int ii = 0; ii < colsPerBlock; ++ii)
          for (int jj = 0; jj < rowsPerBlock; ++jj) {
            int curri = i * colsPerBlock + ii;
            int currj = j * rowsPerBlock + jj;
            float m_value = matrix[curri + currj * nCols];
            float bm_value = CurrBlock[ii + jj * colsPerBlock];
            if (bm_value != m_value) {
              fail++;
            }
          }
      }
    return fail;
  }

  float *GetBlock(int i, int j) const {
    assert(i < nBlocksPerCol && j < nBlocksPerRow && "Accessing outside block");
    return Blocks[i][j].get();
  }
};

constexpr const int BS = 16;
constexpr const int N = 256;

int BlockMatMul_TargetNowait(BlockMatrix &A, BlockMatrix &B, BlockMatrix &C) {
#pragma omp parallel
#pragma omp master
  for (int i = 0; i < N / BS; ++i)
    for (int j = 0; j < N / BS; ++j) {
      float *BlockC = C.GetBlock(i, j);
      for (int k = 0; k < N / BS; ++k) {
        float *BlockA = A.GetBlock(i, k);
        float *BlockB = B.GetBlock(k, j);
// clang-format off
#pragma omp target depend(in: BlockA[0], BlockB[0]) depend(inout: BlockC[0])   \
            map(to: BlockA[:BS * BS], BlockB[:BS * BS])                        \
            map(tofrom: BlockC[:BS * BS]) nowait
// clang-format on
#pragma omp parallel for
        for (int ii = 0; ii < BS; ii++)
          for (int jj = 0; jj < BS; jj++) {
            for (int kk = 0; kk < BS; ++kk)
              BlockC[ii + jj * BS] +=
                  BlockA[ii + kk * BS] * BlockB[kk + jj * BS];
          }
      }
    }
  return 0;
}

void Matmul(const std::vector<float> &a, const std::vector<float> &b,
            std::vector<float> &c) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0;
      for (int k = 0; k < N; ++k) {
        sum = sum + a[i * N + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
}

int main(int argc, char *argv[]) {
  std::vector<float> a(N * N);
  std::vector<float> b(N * N);
  std::vector<float> c(N * N, 0.0);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      a[i * N + j] = b[i * N + j] = i + j % 100;
    }
  }

  auto BlockedA = BlockMatrix(BS, BS, N, N);
  BlockedA.Initialize(a);
  BlockedA.Compare(a);
  auto BlockedB = BlockMatrix(BS, BS, N, N);
  BlockedB.Initialize(b);
  BlockedB.Compare(b);

  Matmul(a, b, c);

  auto BlockedC = BlockMatrix(BS, BS, N, N);
  BlockMatMul_TargetNowait(BlockedA, BlockedB, BlockedC);

  if (BlockedC.Compare(c) > 0) {
    return 1;
  }

  std::cout << "PASS\n";

  return 0;
}

// CHECK: PASS
