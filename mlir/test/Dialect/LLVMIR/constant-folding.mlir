// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm.func(canonicalize))" --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @zext_basic
llvm.func @zext_basic() -> i64 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.zext %0 : i32 to i64
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.return %[[RES]] : i64
  llvm.return %1 : i64
}

// CHECK-LABEL: llvm.func @zext_neg
llvm.func @zext_neg() -> i64 {
  %0 = llvm.mlir.constant(-1 : i32) : i32
  %1 = llvm.zext %0 : i32 to i64
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(4294967295 : i64) : i64
  // CHECK: llvm.return %[[RES]] : i64
  llvm.return %1 : i64
}

// -----

// CHECK-LABEL: llvm.func @shl_basic
llvm.func @shl_basic() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.shl %0, %1 : i32
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %2 : i32
}

// CHECK-LABEL: llvm.func @shl_multiple
llvm.func @shl_multiple() -> i32 {
  %0 = llvm.mlir.constant(45 : i32) : i32
  %1 = llvm.mlir.constant(7 : i32) : i32
  %2 = llvm.shl %0, %1 : i32
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(5760 : i32) : i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @or_basic
llvm.func @or_basic() -> i32 {
  %0 = llvm.mlir.constant(5 : i32) : i32
  %1 = llvm.mlir.constant(9 : i32) : i32
  %2 = llvm.or %0, %1 : i32
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(13 : i32) : i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @addressof
llvm.func @addressof() {
  // CHECK-NEXT: %[[ADDRESSOF:.+]] = llvm.mlir.addressof @foo
  %0 = llvm.mlir.addressof @foo : !llvm.ptr
  %1 = llvm.mlir.addressof @foo : !llvm.ptr
  // CHECK-NEXT: llvm.call @bar(%[[ADDRESSOF]], %[[ADDRESSOF]])
  llvm.call @bar(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.return
  llvm.return
}

llvm.mlir.global constant @foo() : i32

llvm.func @bar(!llvm.ptr, !llvm.ptr)

// -----

// CHECK-LABEL: llvm.func @addressof_select
llvm.func @addressof_select(%arg: i1) -> !llvm.ptr {
  // CHECK-NEXT: %[[ADDRESSOF:.+]] = llvm.mlir.addressof @foo
  %0 = llvm.mlir.addressof @foo : !llvm.ptr
  %1 = llvm.mlir.addressof @foo : !llvm.ptr
  %2 = arith.select %arg, %0, %1 : !llvm.ptr
  // CHECK-NEXT: llvm.return %[[ADDRESSOF]]
  llvm.return %2 : !llvm.ptr
}

llvm.mlir.global constant @foo() : i32

llvm.func @bar(!llvm.ptr, !llvm.ptr)

// -----

// CHECK-LABEL: llvm.func @addressof_blocks
llvm.func @addressof_blocks(%arg: i1) -> !llvm.ptr {
  // CHECK-NEXT: %[[ADDRESSOF:.+]] = llvm.mlir.addressof @foo
  llvm.cond_br %arg, ^bb1, ^bb2
^bb1:
  %0 = llvm.mlir.addressof @foo : !llvm.ptr
  llvm.return %0 : !llvm.ptr
^bb2:
  %1 = llvm.mlir.addressof @foo : !llvm.ptr
  // CHECK: return %[[ADDRESSOF]]
  llvm.return %1 : !llvm.ptr
}

llvm.mlir.global constant @foo() : i32

// -----

// CHECK-LABEL: llvm.func @undef
llvm.func @undef() {
  // CHECK-NEXT: %[[UNDEF:.+]] = llvm.mlir.undef : i32
  %undef1 = llvm.mlir.undef : i32
  %undef2 = llvm.mlir.undef : i32
  // CHECK-NEXT: llvm.call @foo(%[[UNDEF]], %[[UNDEF]])
  llvm.call @foo(%undef1, %undef2) : (i32, i32) -> ()
  // CHECK-NEXT: llvm.return
  llvm.return
}

llvm.func @foo(i32, i32)

// -----

// CHECK-LABEL: llvm.func @poison
llvm.func @poison() {
  // CHECK-NEXT: %[[POISON:.+]] = llvm.mlir.poison : i32
  %poison1 = llvm.mlir.poison : i32
  %poison2 = llvm.mlir.poison : i32
  // CHECK-NEXT: llvm.call @foo(%[[POISON]], %[[POISON]])
  llvm.call @foo(%poison1, %poison2) : (i32, i32) -> ()
  // CHECK-NEXT: llvm.return
  llvm.return
}

llvm.func @foo(i32, i32)

// -----

llvm.func @foo(!llvm.ptr, !llvm.ptr)

// CHECK-LABEL: llvm.func @null_pointer
llvm.func @null_pointer() {
  // CHECK-NEXT: %[[NULLPTR:.+]] = llvm.mlir.zero : !llvm.ptr
  %nullptr1 = llvm.mlir.zero : !llvm.ptr
  %nullptr2 = llvm.mlir.zero : !llvm.ptr
  // CHECK-NEXT: llvm.call @foo(%[[NULLPTR]], %[[NULLPTR]])
  llvm.call @foo(%nullptr1, %nullptr2) : (!llvm.ptr, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.return
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @zero_integer
llvm.func @zero_integer() -> i64 {
  // CHECK-NEXT: %[[ZERO:.+]] = llvm.mlir.constant(0 : i64) : i64
  %zero = llvm.mlir.zero : i32
  %zero_extended = llvm.zext %zero : i32 to i64
  // CHECK-NEXT: llvm.return %[[ZERO]]
  llvm.return %zero_extended : i64
}

// -----

// CHECK-LABEL: llvm.func @null_pointer_select
llvm.func @null_pointer_select(%cond: i1) -> !llvm.ptr {
  // CHECK-NEXT: %[[NULLPTR:.+]] = llvm.mlir.zero : !llvm.ptr
  %nullptr1 = llvm.mlir.zero : !llvm.ptr
  %nullptr2 = llvm.mlir.zero : !llvm.ptr
  %result = arith.select %cond, %nullptr1, %nullptr2 : !llvm.ptr
  // CHECK-NEXT: llvm.return %[[NULLPTR]]
  llvm.return %result : !llvm.ptr
}

// -----

llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL: func.func @insert_op
func.func @insert_op(%arg0: index, %arg1: memref<13x13xi64>, %arg2: index) {
  %cst_7 = arith.constant dense<1526248407> : vector<1xi64>
  %1 = llvm.mlir.constant(1 : index) : i64
  %101 = vector.insert %1, %cst_7 [0] : i64 into vector<1xi64>
  vector.print %101 : vector<1xi64>
  return
}
