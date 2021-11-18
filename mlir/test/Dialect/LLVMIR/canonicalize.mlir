// RUN: mlir-opt -canonicalize %s -split-input-file | FileCheck %s

// CHECK-LABEL: fold_extractvalue
llvm.func @fold_extractvalue() -> i32 {
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  %c0 = arith.constant 0 : i32
  //  CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  %c1 = arith.constant 1 : i32

  %0 = llvm.mlir.undef : !llvm.struct<(i32, i32)>

  // CHECK-NOT: insertvalue
  %1 = llvm.insertvalue %c0, %0[0] : !llvm.struct<(i32, i32)>
  %2 = llvm.insertvalue %c1, %1[1] : !llvm.struct<(i32, i32)>

  // CHECK-NOT: extractvalue
  %3 = llvm.extractvalue %2[0] : !llvm.struct<(i32, i32)>
  %4 = llvm.extractvalue %2[1] : !llvm.struct<(i32, i32)>

  //     CHECK: llvm.add %[[C0]], %[[C1]]
  %5 = llvm.add %3, %4 : i32
  llvm.return %5 : i32
}

// -----

// CHECK-LABEL: no_fold_extractvalue
llvm.func @no_fold_extractvalue(%arr: !llvm.array<4xf32>) -> f32 {
  %f0 = arith.constant 0.0 : f32
  %0 = llvm.mlir.undef : !llvm.array<4 x !llvm.array<4xf32>>

  // CHECK: insertvalue
  // CHECK: insertvalue
  // CHECK: extractvalue
  %1 = llvm.insertvalue %f0, %0[0, 0] : !llvm.array<4 x !llvm.array<4xf32>>
  %2 = llvm.insertvalue %arr, %1[0] : !llvm.array<4 x !llvm.array<4xf32>>
  %3 = llvm.extractvalue %2[0, 0] : !llvm.array<4 x !llvm.array<4xf32>>

  llvm.return %3 : f32
}
