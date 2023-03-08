//
// Created by tanmay on 3/7/23.
//

#ifndef LLVM_RUNTIMEUTILITIES_H
#define LLVM_RUNTIMEUTILITIES_H

// Find the max value from a pointer array of specified length and return the
// index.
//
// @param[in]  pArray  Pointer to the array
// @param[in]  length  Length of the array
//
// @return     Index of the max value in the array
//
int findMaxIndex(double *PArray, int Length) {
  assert(PArray != NULL && Length > 0);

  int MaxIndex = 0;
  float MaxValue = PArray[0];
  for (int I = 1; I < Length; I++) {
    if (PArray[I] > MaxValue) {
      MaxValue = PArray[I];
      MaxIndex = I;
    }
  }
  return MaxIndex;
}

// Find the min value from a pointer array of specified length and return the
// index.
//
// @param[in]  pArray  Pointer to the array
// @param[in]  length  Length of the array
//
// @return     Index of the min value in the array
//
int findMinIndex(double *PArray, int Length) {
  int MinIndex = 0;
  float MinValue = PArray[0];
  for (int i = 1; i < Length; i++) {
    if (PArray[i] < MinValue) {
      MinValue = PArray[i];
      MinIndex = i;
    }
  }
  return MinIndex;
}

#endif // LLVM_RUNTIMEUTILITIES_H
