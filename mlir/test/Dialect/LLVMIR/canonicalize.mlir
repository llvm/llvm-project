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
// -----

// CHECK-LABEL: fold_unrelated_extractvalue
llvm.func @fold_unrelated_extractvalue(%arr: !llvm.array<4xf32>) -> f32 {
  %f0 = arith.constant 0.0 : f32
  // CHECK-NOT: insertvalue
  // CHECK: extractvalue
  %2 = llvm.insertvalue %f0, %arr[0] : !llvm.array<4xf32>
  %3 = llvm.extractvalue %2[1] : !llvm.array<4xf32>
  llvm.return %3 : f32
}

// -----

// CHECK-LABEL: fold_bitcast
// CHECK-SAME: %[[a0:arg[0-9]+]]
// CHECK-NEXT: llvm.return %[[a0]]
llvm.func @fold_bitcast(%x : !llvm.ptr<i8>) -> !llvm.ptr<i8> {
  %c = llvm.bitcast %x : !llvm.ptr<i8> to !llvm.ptr<i8>
  llvm.return %c : !llvm.ptr<i8>
}

// CHECK-LABEL: fold_bitcast2
// CHECK-SAME: %[[a0:arg[0-9]+]]
// CHECK-NEXT: llvm.return %[[a0]]
llvm.func @fold_bitcast2(%x : !llvm.ptr<i8>) -> !llvm.ptr<i8> {
  %c = llvm.bitcast %x : !llvm.ptr<i8> to !llvm.ptr<i32>
  %d = llvm.bitcast %c : !llvm.ptr<i32> to !llvm.ptr<i8>
  llvm.return %d : !llvm.ptr<i8>
}

// -----

// CHECK-LABEL: fold_addrcast
// CHECK-SAME: %[[a0:arg[0-9]+]]
// CHECK-NEXT: llvm.return %[[a0]]
llvm.func @fold_addrcast(%x : !llvm.ptr<i8>) -> !llvm.ptr<i8> {
  %c = llvm.addrspacecast %x : !llvm.ptr<i8> to !llvm.ptr<i8>
  llvm.return %c : !llvm.ptr<i8>
}

// CHECK-LABEL: fold_addrcast2
// CHECK-SAME: %[[a0:arg[0-9]+]]
// CHECK-NEXT: llvm.return %[[a0]]
llvm.func @fold_addrcast2(%x : !llvm.ptr<i8>) -> !llvm.ptr<i8> {
  %c = llvm.addrspacecast %x : !llvm.ptr<i8> to !llvm.ptr<i32, 5>
  %d = llvm.addrspacecast %c : !llvm.ptr<i32, 5> to !llvm.ptr<i8>
  llvm.return %d : !llvm.ptr<i8>
}

// -----

// CHECK-LABEL: fold_gep
// CHECK-SAME: %[[a0:arg[0-9]+]]
// CHECK-NEXT: llvm.return %[[a0]]
llvm.func @fold_gep(%x : !llvm.ptr<i8>) -> !llvm.ptr<i8> {
  %c0 = arith.constant 0 : i32
  %c = llvm.getelementptr %x[%c0] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
  llvm.return %c : !llvm.ptr<i8>
}

// -----

// Check that LLVM constants participate in cross-dialect constant folding. The
// resulting constant is created in the arith dialect because the last folded
// operation belongs to it.
// CHECK-LABEL: llvm_constant
func.func @llvm_constant() -> i32 {
  // CHECK-NOT: llvm.mlir.constant
  %0 = llvm.mlir.constant(40 : i32) : i32
  %1 = llvm.mlir.constant(42 : i32) : i32
  // CHECK: %[[RES:.*]] = arith.constant 82 : i32
  // CHECK-NOT: arith.addi
  %2 = arith.addi %0, %1 : i32
  // CHECK: return %[[RES]]
  return %2 : i32
}

// -----

// CHECK-LABEL: load_dce
// CHECK-NEXT: llvm.return
llvm.func @load_dce(%x : !llvm.ptr<i8>) {
  %0 = llvm.load %x : !llvm.ptr<i8>
  llvm.return 
}

llvm.mlir.global external @fp() : !llvm.ptr<i8>

// CHECK-LABEL: addr_dce
// CHECK-NEXT: llvm.return
llvm.func @addr_dce(%x : !llvm.ptr<i8>) {
  %0 = llvm.mlir.addressof @fp : !llvm.ptr<ptr<i8>>
  llvm.return 
}

// CHECK-LABEL: alloca_dce
// CHECK-NEXT: llvm.return
llvm.func @alloca_dce() {
  %c1_i64 = arith.constant 1 : i64
  %0 = llvm.alloca %c1_i64 x i32 : (i64) -> !llvm.ptr<i32>
  llvm.return 
}
