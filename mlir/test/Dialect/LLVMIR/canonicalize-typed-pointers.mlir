// RUN: mlir-opt --pass-pipeline='builtin.module(llvm.func(canonicalize{test-convergence}))' %s -split-input-file | FileCheck %s

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

// CHECK-LABEL: fold_gep_canon
// CHECK-SAME: %[[a0:arg[0-9]+]]
// CHECK-NEXT: %[[RES:.*]] = llvm.getelementptr %[[a0]][2]
// CHECK-NEXT: llvm.return %[[RES]]
llvm.func @fold_gep_canon(%x : !llvm.ptr<i8>) -> !llvm.ptr<i8> {
  %c2 = arith.constant 2 : i32
  %c = llvm.getelementptr %x[%c2] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
  llvm.return %c : !llvm.ptr<i8>
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
