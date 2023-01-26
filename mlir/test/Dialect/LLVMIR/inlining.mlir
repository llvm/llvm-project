// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

func.func @inner_func_inlinable(%ptr : !llvm.ptr) -> i32 {
  %0 = llvm.mlir.constant(42 : i32) : i32
  llvm.store %0, %ptr { alignment = 8 } : i32, !llvm.ptr
  %1 = llvm.load %ptr { alignment = 8 } : !llvm.ptr -> i32
  return %1 : i32
}

// CHECK-LABEL: func.func @test_inline(
// CHECK-SAME: %[[PTR:[a-zA-Z0-9_]+]]
// CHECK-NEXT: %[[CST:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[CST]], %[[PTR]]
// CHECK-NEXT: %[[RES:.+]] = llvm.load %[[PTR]]
// CHECK-NEXT: return %[[RES]] : i32
func.func @test_inline(%ptr : !llvm.ptr) -> i32 {
  %0 = call @inner_func_inlinable(%ptr) : (!llvm.ptr) -> i32
  return %0 : i32
}

// -----

func.func @inner_func_not_inlinable() -> !llvm.ptr<f64> {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.alloca %0 x f64 : (i32) -> !llvm.ptr<f64>
  return %1 : !llvm.ptr<f64>
}

// CHECK-LABEL: func.func @test_not_inline() -> !llvm.ptr<f64> {
// CHECK-NEXT: %[[RES:.*]] = call @inner_func_not_inlinable() : () -> !llvm.ptr<f64>
// CHECK-NEXT: return %[[RES]] : !llvm.ptr<f64>
func.func @test_not_inline() -> !llvm.ptr<f64> {
  %0 = call @inner_func_not_inlinable() : () -> !llvm.ptr<f64>
  return %0 : !llvm.ptr<f64>
}

// -----

llvm.metadata @metadata {
  llvm.access_group @group
  llvm.return
}

func.func private @with_mem_attr(%ptr : !llvm.ptr) {
  %0 = llvm.mlir.constant(42 : i32) : i32
  // Do not inline load/store operations that carry attributes requiring
  // handling while inlining, until this is supported by the inliner.
  llvm.store %0, %ptr { access_groups = [@metadata::@group] }: i32, !llvm.ptr
  return
}

// CHECK-LABEL: func.func @test_not_inline
// CHECK-NEXT: call @with_mem_attr
// CHECK-NEXT: return
func.func @test_not_inline(%ptr : !llvm.ptr) {
  call @with_mem_attr(%ptr) : (!llvm.ptr) -> ()
  return
}

// -----
// Check that llvm.return is correctly handled

func.func @func(%arg0 : i32) -> i32  {
  llvm.return %arg0 : i32
}
// CHECK-LABEL: @llvm_ret
// CHECK-NOT: call
// CHECK:  return %arg0
func.func @llvm_ret(%arg0 : i32) -> i32 {
  %res = call @func(%arg0) : (i32) -> (i32)
  return %res : i32
}

// -----

// Include all function attributes that don't prevent inlining
llvm.func internal fastcc @callee() -> (i32) attributes { function_entry_count = 42 : i64, dso_local } {
  %0 = llvm.mlir.constant(42 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: %[[CST:.+]] = llvm.mlir.constant
// CHECK-NEXT: llvm.return %[[CST]]
llvm.func @caller() -> (i32) {
  // Include all call attributes that don't prevent inlining.
  %0 = llvm.call @callee() { fastmathFlags = #llvm.fastmath<nnan, ninf> } : () -> (i32)
  llvm.return %0 : i32
}

// -----

llvm.func @foo() -> (i32) attributes { passthrough = ["noinline"] } {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

llvm.func @bar() -> (i32) attributes { passthrough = ["noinline"] } {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

llvm.func @callee_with_multiple_blocks(%cond: i1) -> (i32) {
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = llvm.call @foo() : () -> (i32)
  llvm.br ^bb3(%0: i32)
^bb2:
  %1 = llvm.call @bar() : () -> (i32)
  llvm.br ^bb3(%1: i32)
^bb3(%arg: i32):
  llvm.return %arg : i32
}

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: llvm.cond_br {{.+}}, ^[[BB1:.+]], ^[[BB2:.+]]
// CHECK-NEXT: ^[[BB1]]:
// CHECK-NEXT: llvm.call @foo
// CHECK-NEXT: llvm.br ^[[BB3:[a-zA-Z0-9_]+]]
// CHECK-NEXT: ^[[BB2]]:
// CHECK-NEXT: llvm.call @bar
// CHECK-NEXT: llvm.br ^[[BB3]]
// CHECK-NEXT: ^[[BB3]]
// CHECK-NEXT: llvm.br ^[[BB4:[a-zA-Z0-9_]+]]
// CHECK-NEXT: ^[[BB4]]
// CHECK-NEXT: llvm.return
llvm.func @caller(%cond: i1) -> (i32) {
  %0 = llvm.call @callee_with_multiple_blocks(%cond) : (i1) -> (i32)
  llvm.return %0 : i32
}

// -----

llvm.func @personality() -> i32

llvm.func @callee() -> (i32) attributes { personality = @personality } {
  %0 = llvm.mlir.constant(42 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: llvm.call @callee
// CHECK-NEXT: return
llvm.func @caller() -> (i32) {
  %0 = llvm.call @callee() : () -> (i32)
  llvm.return %0 : i32
}

// -----

llvm.func @callee() -> (i32) attributes { passthrough = ["foo"] } {
  %0 = llvm.mlir.constant(42 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: llvm.call @callee
// CHECK-NEXT: return
llvm.func @caller() -> (i32) {
  %0 = llvm.call @callee() : () -> (i32)
  llvm.return %0 : i32
}

// -----

llvm.func @callee() -> (i32) attributes { garbageCollector = "foo" } {
  %0 = llvm.mlir.constant(42 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: llvm.call @callee
// CHECK-NEXT: return
llvm.func @caller() -> (i32) {
  %0 = llvm.call @callee() : () -> (i32)
  llvm.return %0 : i32
}

// -----

llvm.func @callee(%ptr : !llvm.ptr {llvm.byval = !llvm.ptr}) -> (!llvm.ptr) {
  llvm.return %ptr : !llvm.ptr
}

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: llvm.call @callee
// CHECK-NEXT: return
llvm.func @caller(%ptr : !llvm.ptr) -> (!llvm.ptr) {
  %0 = llvm.call @callee(%ptr) : (!llvm.ptr) -> (!llvm.ptr)
  llvm.return %0 : !llvm.ptr
}

// -----

llvm.func @callee() {
  llvm.return
}

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: llvm.call @callee
// CHECK-NEXT: llvm.return
llvm.func @caller() {
  llvm.call @callee() { branch_weights = dense<42> : vector<1xi32> } : () -> ()
  llvm.return
}
