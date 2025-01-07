// Split the MLIR string: this will produce %t/input.mlir
// RUN: split-file %s %t

// Compile the MLIR file to LLVM:
// RUN: mlir-opt %t/input.mlir \
// RUN:  -lower-affine  -convert-scf-to-cf  -finalize-memref-to-llvm \
// RUN:  -convert-func-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:  -reconcile-unrealized-casts \
// RUN: | mlir-translate --mlir-to-llvmir -o %t.ll

// Generate an object file for the MLIR code
// RUN: llc %t.ll -o %t.o -filetype=obj

// Compile the current C file and link it to the MLIR code:
// RUN: %host_cc %s %t.o -o %t.exe

// Exec
// RUN: %t.exe | FileCheck %s

/* MLIR_BEGIN
//--- input.mlir
// Performs: arg0[i, j] = arg0[i, j] + arg1[i, j]
func.func private @add_memref(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>) -> i64
   attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dimI = memref.dim %arg0, %c0 : memref<?x?xf64>
  %dimJ = memref.dim %arg0, %c1 : memref<?x?xf64>
  affine.for %i = 0 to %dimI {
    affine.for %j = 0 to %dimJ {
      %load0 = memref.load %arg0[%i, %j] : memref<?x?xf64>
      %load1 = memref.load %arg1[%i, %j] : memref<?x?xf64>
      %add = arith.addf %load0, %load1 : f64
      affine.store %add, %arg0[%i, %j] : memref<?x?xf64>
    }
  }
  %c42 = arith.constant 42 : i64
  return %c42 : i64
}

//--- end_input.mlir

MLIR_END */

#include <stdint.h>
#include <stdio.h>

// Define the API for the MLIR function, see
// https://mlir.llvm.org/docs/TargetLLVMIR/#calling-conventions for details.
//
// The function takes two 2D memref, the signature in MLIR LLVM dialect will be:
// llvm.func @add_memref(
//   // First Memref (%arg0)
//      %allocated_ptr0: !llvm.ptr<f64>, %aligned_ptr0: !llvm.ptr<f64>,
//      %offset0: i64, %size0_d0: i64, %size0_d1: i64, %stride0_d0: i64,
//      %stride0_d1: i64,
//   // Second Memref (%arg1)
//      %allocated_ptr1: !llvm.ptr<f64>, %aligned_ptr1: !llvm.ptr<f64>,
//      %offset1: i64, %size1_d0: i64, %size1_d1: i64, %stride1_d0: i64,
//      %stride1_d1: i64,
//
long long add_memref(double *allocated_ptr0, double *aligned_ptr0,
                     intptr_t offset0, intptr_t size0_d0, intptr_t size0_d1,
                     intptr_t stride0_d0, intptr_t stride0_d1,
                     // Second Memref (%arg1)
                     double *allocated_ptr1, double *aligned_ptr1,
                     intptr_t offset1, intptr_t size1_d0, intptr_t size1_d1,
                     intptr_t stride1_d0, intptr_t stride1_d1);

// The llvm.emit_c_interface will also trigger emission of another wrapper:
// llvm.func @_mlir_ciface_add_memref(
//   %arg0: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64,
//                            array<2 x i64>, array<2 x i64>)>>,
//   %arg1: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64,
//                            array<2 x i64>, array<2 x i64>)>>)
// -> i64
typedef struct {
  double *allocated;
  double *aligned;
  intptr_t offset;
  intptr_t size[2];
  intptr_t stride[2];
} memref_2d_descriptor;
long long _mlir_ciface_add_memref(memref_2d_descriptor *arg0,
                                  memref_2d_descriptor *arg1);

#define N 4
#define M 8
double arg0[N][M];
double arg1[N][M];

void dump() {
  for (int i = 0; i < N; i++) {
    printf("[");
    for (int j = 0; j < M; j++)
      printf("%d,\t", (int)arg0[i][j]);
    printf("] [");
    for (int j = 0; j < M; j++)
      printf("%d,\t", (int)arg1[i][j]);
    printf("]\n");
  }
}

int main() {
  int count = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      arg0[i][j] = count++;
      arg1[i][j] = count++;
    }
  }
  printf("Before:\n");
  dump();
  // clang-format off
  // CHECK-LABEL: Before:
  // CHECK: [0,	  2,	4,	6,	8,	10,	12,	14,	] [1,	  3,	5, 7, 9,	11,	13,	15,	]
  // CHECK: [16,	18,	20,	22, 24, 26,	28,	30,	] [17,	19,	21,	23,	25,	27,	29, 31, ]
  // CHECK: [32,	34,	36,	38,	40,	42,	44,	46,	] [33,	35, 37, 39,	41,	43,	45,	47,	]
  // CHECK: [48,	50,	52, 54, 56,	58,	60,	62,	] [49,	51,	53,	55,	57,	59, 61, 63,	]
  // clang-format on

  // Call into MLIR.
  long long result = add_memref((double *)arg0, (double *)arg0, 0, N, M, M, 0,
                                //
                                (double *)arg1, (double *)arg1, 0, N, M, M, 0);

  // CHECK-LABEL: Result:
  // CHECK: 42
  printf("Result: %d\n", (int)result);

  printf("After:\n");
  dump();

  // clang-format off
  // CHECK-LABEL: After:
  // CHECK: [1,	  5,	  9,	  13,	 17,	21,	  25,	  29,	  ] [1, 3,	5,	7,	9,	11,	13,	15,	] 
  // CHECK: [33,	37,  41,	  45,	 49,	53,	  57,	  61,	  ] [17,	19,	21, 23, 25,	27,	29,	31,	]
  // CHECK: [65,	69,	  73,   77,	 81,	85,	  89,	  93,	  ] [33,	35,	37,	39, 41, 43,	45,	47,	]
  // CHECK: [97,	101,	105,	109, 113,	117,	121,	125,	] [49,	51,	53,	55,	57,	59, 61, 63,	]
  // clang-format on

  // Reset the input and re-apply the same function use the C API wrapper.
  count = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      arg0[i][j] = count++;
      arg1[i][j] = count++;
    }
  }

  // Call into MLIR.
  memref_2d_descriptor arg0_descriptor = {
      (double *)arg0, (double *)arg0, 0, N, M, M, 0};
  memref_2d_descriptor arg1_descriptor = {
      (double *)arg1, (double *)arg1, 0, N, M, M, 0};
  result = _mlir_ciface_add_memref(&arg0_descriptor, &arg1_descriptor);

  // CHECK-LABEL: Result2:
  // CHECK: 42
  printf("Result2: %d\n", (int)result);

  printf("After2:\n");
  dump();

  // clang-format off
  // CHECK-LABEL: After2:
  // CHECK: [1,	  5,	  9,	  13,	 17,	21,	  25,	  29,	  ] [1, 3,	5,	7,	9,	11,	13,	15,	] 
  // CHECK: [33,	37,  41,	  45,	 49,	53,	  57,	  61,	  ] [17,	19,	21, 23, 25,	27,	29,	31,	]
  // CHECK: [65,	69,	  73,   77,	 81,	85,	  89,	  93,	  ] [33,	35,	37,	39, 41, 43,	45,	47,	]
  // CHECK: [97,	101,	105,	109, 113,	117,	121,	125,	] [49,	51,	53,	55,	57,	59, 61, 63,	]
  // clang-format on

  return 0;
}
