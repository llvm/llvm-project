// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

#file = #llvm.di_file<"foo.mlir" in "/foo/">
#variable = #llvm.di_local_variable<scope = #file>
#variableAddr = #llvm.di_local_variable<scope = #file>
#label = #llvm.di_label<scope = #file>

func.func @inner_func_inlinable(%ptr : !llvm.ptr) -> i32 {
  %0 = llvm.mlir.constant(42 : i32) : i32
  %stack = llvm.intr.stacksave : !llvm.ptr
  llvm.store %0, %ptr { alignment = 8 } : i32, !llvm.ptr
  %1 = llvm.load %ptr { alignment = 8 } : !llvm.ptr -> i32
  llvm.intr.dbg.value #variable = %0 : i32
  llvm.intr.dbg.declare #variableAddr = %ptr : !llvm.ptr
  llvm.intr.dbg.label #label
  %byte = llvm.mlir.constant(43 : i8) : i8
  %true = llvm.mlir.constant(1 : i1) : i1
  "llvm.intr.memset"(%ptr, %byte, %0) <{isVolatile = true}> : (!llvm.ptr, i8, i32) -> ()
  "llvm.intr.memmove"(%ptr, %ptr, %0) <{isVolatile = true}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  "llvm.intr.memcpy"(%ptr, %ptr, %0) <{isVolatile = true}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  "llvm.intr.assume"(%true) : (i1) -> ()
  llvm.fence release
  %2 = llvm.atomicrmw add %ptr, %0 monotonic : !llvm.ptr, i32
  %3 = llvm.cmpxchg %ptr, %0, %1 acq_rel monotonic : !llvm.ptr, i32
  llvm.inline_asm has_side_effects "foo", "bar" : () -> ()
  llvm.cond_br %true, ^bb1, ^bb2
^bb1:
  llvm.unreachable
^bb2:
  llvm.intr.stackrestore %stack : !llvm.ptr
  llvm.call_intrinsic "llvm.x86.sse41.round.ss"() : () -> (vector<8xf32>)
  return %1 : i32
}

// CHECK-LABEL: func.func @test_inline(
// CHECK-SAME: %[[PTR:[a-zA-Z0-9_]+]]
// CHECK: %[[CST:.*]] = llvm.mlir.constant(42
// CHECK: %[[STACK:.+]] = llvm.intr.stacksave
// CHECK: llvm.store %[[CST]], %[[PTR]]
// CHECK: %[[RES:.+]] = llvm.load %[[PTR]]
// CHECK: llvm.intr.dbg.value #{{.+}} = %[[CST]]
// CHECK: llvm.intr.dbg.declare #{{.+}} = %[[PTR]]
// CHECK: llvm.intr.dbg.label #{{.+}}
// CHECK: "llvm.intr.memset"(%[[PTR]]
// CHECK: "llvm.intr.memmove"(%[[PTR]], %[[PTR]]
// CHECK: "llvm.intr.memcpy"(%[[PTR]], %[[PTR]]
// CHECK: "llvm.intr.assume"
// CHECK: llvm.fence release
// CHECK: llvm.atomicrmw add %[[PTR]], %[[CST]] monotonic
// CHECK: llvm.cmpxchg %[[PTR]], %[[CST]], %[[RES]] acq_rel monotonic
// CHECK: llvm.inline_asm has_side_effects "foo", "bar"
// CHECK: llvm.unreachable
// CHECK: llvm.intr.stackrestore %[[STACK]]
// CHECK: llvm.call_intrinsic "llvm.x86.sse41.round.ss"(
func.func @test_inline(%ptr : !llvm.ptr) -> i32 {
  %0 = call @inner_func_inlinable(%ptr) : (!llvm.ptr) -> i32
  return %0 : i32
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
  %0 = llvm.call fastcc @callee() { fastmathFlags = #llvm.fastmath<nnan, ninf>, branch_weights = dense<42> : vector<1xi32> } : () -> (i32)
  llvm.return %0 : i32
}

// -----

llvm.func @foo() -> (i32) attributes { no_inline } {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

llvm.func @bar() -> (i32) attributes { no_inline } {
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

llvm.func @callee() attributes { passthrough = ["foo", "bar"] } {
  llvm.return
}

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: llvm.return
llvm.func @caller() {
  llvm.call @callee() : () -> ()
  llvm.return
}

// -----

llvm.func @callee_noinline() attributes { no_inline } {
  llvm.return
}

llvm.func @callee_noduplicate() attributes { passthrough = ["noduplicate"] } {
  llvm.return
}

llvm.func @callee_presplitcoroutine() attributes { passthrough = ["presplitcoroutine"] } {
  llvm.return
}

llvm.func @callee_returns_twice() attributes { passthrough = ["returns_twice"] } {
  llvm.return
}

llvm.func @callee_strictfp() attributes { passthrough = ["strictfp"] } {
  llvm.return
}

// CHECK-LABEL: llvm.func @caller
// CHECK-NEXT: llvm.call @callee_noinline
// CHECK-NEXT: llvm.call @callee_noduplicate
// CHECK-NEXT: llvm.call @callee_presplitcoroutine
// CHECK-NEXT: llvm.call @callee_returns_twice
// CHECK-NEXT: llvm.call @callee_strictfp
// CHECK-NEXT: llvm.return
llvm.func @caller() {
  llvm.call @callee_noinline() : () -> ()
  llvm.call @callee_noduplicate() : () -> ()
  llvm.call @callee_presplitcoroutine() : () -> ()
  llvm.call @callee_returns_twice() : () -> ()
  llvm.call @callee_strictfp() : () -> ()
  llvm.return
}

// -----

llvm.func @static_alloca() -> f32 {
  %0 = llvm.mlir.constant(4 : i32) : i32
  %1 = llvm.alloca %0 x f32 : (i32) -> !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> f32
  llvm.return %2 : f32
}

llvm.func @dynamic_alloca(%size : i32) -> f32 {
  %0 = llvm.add %size, %size : i32
  %1 = llvm.alloca %0 x f32 : (i32) -> !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> f32
  llvm.return %2 : f32
}

// CHECK-LABEL: llvm.func @test_inline
llvm.func @test_inline(%cond : i1, %size : i32) -> f32 {
  // Check that the static alloca was moved to the entry block after inlining
  // with its size defined by a constant.
  // CHECK-NOT: ^{{.+}}:
  // CHECK-NEXT: llvm.mlir.constant
  // CHECK-NEXT: llvm.alloca
  // CHECK: llvm.cond_br
  llvm.cond_br %cond, ^bb1, ^bb2
  // CHECK: ^{{.+}}:
^bb1:
  // CHECK-NOT: llvm.call @static_alloca
  // CHECK: llvm.intr.lifetime.start
  %0 = llvm.call @static_alloca() : () -> f32
  // CHECK: llvm.intr.lifetime.end
  // CHECK: llvm.br ^[[BB3:[a-zA-Z0-9_]+]]
  llvm.br ^bb3(%0: f32)
  // CHECK: ^{{.+}}:
^bb2:
  // Check that the dynamic alloca was inlined, but that it was not moved to the
  // entry block.
  // CHECK: %[[STACK:[a-zA-Z0-9_]+]] = llvm.intr.stacksave
  // CHECK: llvm.add
  // CHECK: llvm.alloca
  // CHECK: llvm.intr.stackrestore %[[STACK]]
  // CHECK-NOT: llvm.call @dynamic_alloca
  %1 = llvm.call @dynamic_alloca(%size) : (i32) -> f32
  // CHECK: llvm.br ^[[BB3]]
  llvm.br ^bb3(%1: f32)
  // CHECK: ^[[BB3]]
^bb3(%arg : f32):
  // CHECK-NEXT: return
  llvm.return %arg : f32
}

// -----

llvm.func @static_alloca_not_in_entry(%cond : i1) -> f32 {
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = llvm.mlir.constant(4 : i32) : i32
  %1 = llvm.alloca %0 x f32 : (i32) -> !llvm.ptr
  llvm.br ^bb3(%1: !llvm.ptr)
^bb2:
  %2 = llvm.mlir.constant(8 : i32) : i32
  %3 = llvm.alloca %2 x f32 : (i32) -> !llvm.ptr
  llvm.br ^bb3(%3: !llvm.ptr)
^bb3(%ptr : !llvm.ptr):
  %4 = llvm.load %ptr : !llvm.ptr -> f32
  llvm.return %4 : f32
}

// CHECK-LABEL: llvm.func @test_inline
llvm.func @test_inline(%cond : i1) -> f32 {
  // Make sure the alloca was not moved to the entry block.
  // CHECK-NOT: llvm.alloca
  // CHECK: llvm.cond_br
  // CHECK: llvm.alloca
  %0 = llvm.call @static_alloca_not_in_entry(%cond) : (i1) -> f32
  llvm.return %0 : f32
}

// -----

llvm.func @static_alloca(%cond: i1) -> f32 {
  %0 = llvm.mlir.constant(4 : i32) : i32
  %1 = llvm.alloca %0 x f32 : (i32) -> !llvm.ptr
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  %2 = llvm.load %1 : !llvm.ptr -> f32
  llvm.return %2 : f32
^bb2:
  %3 = llvm.mlir.constant(3.14192 : f32) : f32
  llvm.return %3 : f32
}

// CHECK-LABEL: llvm.func @test_inline
llvm.func @test_inline(%cond0 : i1, %cond1 : i1, %funcArg : f32) -> f32 {
  // CHECK-NOT: llvm.cond_br
  // CHECK: %[[PTR:.+]] = llvm.alloca
  // CHECK: llvm.cond_br %{{.+}}, ^[[BB1:.+]], ^{{.+}}
  llvm.cond_br %cond0, ^bb1, ^bb2
  // CHECK: ^[[BB1]]
^bb1:
  // Make sure the lifetime begin intrinsic has been inserted where the call
  // used to be, even though the alloca has been moved to the entry block.
  // CHECK-NEXT: llvm.intr.lifetime.start 4, %[[PTR]]
  %0 = llvm.call @static_alloca(%cond1) : (i1) -> f32
  // CHECK: llvm.cond_br %{{.+}}, ^[[BB2:.+]], ^[[BB3:.+]]
  llvm.br ^bb3(%0: f32)
  // Make sure the lifetime end intrinsic has been inserted at both former
  // return sites of the callee.
  // CHECK: ^[[BB2]]:
  // CHECK-NEXT: llvm.load
  // CHECK-NEXT: llvm.intr.lifetime.end 4, %[[PTR]]
  // CHECK: ^[[BB3]]:
  // CHECK-NEXT: llvm.intr.lifetime.end 4, %[[PTR]]
^bb2:
  llvm.br ^bb3(%funcArg: f32)
^bb3(%blockArg: f32):
  llvm.return %blockArg : f32
}

// -----

llvm.func @static_alloca() -> f32 {
  %0 = llvm.mlir.constant(4 : i32) : i32
  %1 = llvm.alloca %0 x f32 : (i32) -> !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> f32
  llvm.return %2 : f32
}

// CHECK-LABEL: llvm.func @test_inline
llvm.func @test_inline(%cond0 : i1) {
  // Verify the alloca is relocated to the entry block of the parent function
  // if the region operation is neither marked as isolated from above or
  // automatic allocation scope.
  // CHECK: %[[ALLOCA:.+]] = llvm.alloca
  // CHECK: "test.one_region_op"() ({
  "test.one_region_op"() ({
    %0 = llvm.call @static_alloca() : () -> f32
    // CHECK-NEXT: llvm.intr.lifetime.start 4, %[[ALLOCA]]
    // CHECK-NEXT: %[[RES:.+]] = llvm.load %[[ALLOCA]]
    // CHECK-NEXT: llvm.intr.lifetime.end 4, %[[ALLOCA]]
    // CHECK-NEXT: test.region_yield %[[RES]]
    test.region_yield %0 : f32
  }) : () -> ()
  // Verify the alloca is not relocated out of operations that are marked as
  // isolated from above.
  // CHECK-NOT: llvm.alloca
  // CHECK: test.isolated_regions
  test.isolated_regions {
    // CHECK: %[[ALLOCA:.+]] = llvm.alloca
    %0 = llvm.call @static_alloca() : () -> f32
    // CHECK: test.region_yield
    test.region_yield %0 : f32
  }
  // Verify the alloca is not relocated out of operations that are marked as
  // automatic allocation scope.
  // CHECK-NOT: llvm.alloca
  // CHECK: test.alloca_scope_region
  test.alloca_scope_region {
    // CHECK: %[[ALLOCA:.+]] = llvm.alloca
    %0 = llvm.call @static_alloca() : () -> f32
    // CHECK: test.region_yield
    test.region_yield %0 : f32
  }
  llvm.return
}

// -----

llvm.func @alloca_with_lifetime(%cond: i1) -> f32 {
  %0 = llvm.mlir.constant(4 : i32) : i32
  %1 = llvm.alloca %0 x f32 : (i32) -> !llvm.ptr
  llvm.intr.lifetime.start 4, %1 : !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> f32
  llvm.intr.lifetime.end 4, %1 : !llvm.ptr
  %3 = llvm.fadd %2, %2 : f32
  llvm.return %3 : f32
}

// CHECK-LABEL: llvm.func @test_inline
llvm.func @test_inline(%cond0 : i1, %cond1 : i1, %funcArg : f32) -> f32 {
  // CHECK-NOT: llvm.cond_br
  // CHECK: %[[PTR:.+]] = llvm.alloca
  // CHECK: llvm.cond_br %{{.+}}, ^[[BB1:.+]], ^{{.+}}
  llvm.cond_br %cond0, ^bb1, ^bb2
  // CHECK: ^[[BB1]]
^bb1:
  // Make sure the original lifetime intrinsic has been preserved, rather than
  // inserting a new one with a larger scope.
  // CHECK: llvm.intr.lifetime.start 4, %[[PTR]]
  // CHECK-NEXT: llvm.load %[[PTR]]
  // CHECK-NEXT: llvm.intr.lifetime.end 4, %[[PTR]]
  // CHECK: llvm.fadd
  // CHECK-NOT: llvm.intr.lifetime.end
  %0 = llvm.call @alloca_with_lifetime(%cond1) : (i1) -> f32
  llvm.br ^bb3(%0: f32)
^bb2:
  llvm.br ^bb3(%funcArg: f32)
^bb3(%blockArg: f32):
  llvm.return %blockArg : f32
}

// -----

llvm.func @with_byval_arg(%ptr : !llvm.ptr { llvm.byval = f64 }) {
  llvm.return
}

// CHECK-LABEL: llvm.func @test_byval
// CHECK-SAME: %[[PTR:[a-zA-Z0-9_]+]]: !llvm.ptr
llvm.func @test_byval(%ptr : !llvm.ptr) {
  // Make sure the new static alloca goes to the entry block.
  // CHECK: %[[ALLOCA:.+]] = llvm.alloca %{{.+}} x f64
  // CHECK: llvm.br ^[[BB1:[a-zA-Z0-9_]+]]
  llvm.br ^bb1
  // CHECK: ^[[BB1]]
^bb1:
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCA]], %[[PTR]]
  llvm.call @with_byval_arg(%ptr) : (!llvm.ptr) -> ()
  llvm.br ^bb2
^bb2:
  llvm.return
}

// -----

llvm.func @with_byval_arg(%ptr : !llvm.ptr { llvm.byval = f64 }) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = read, inaccessibleMem = readwrite>} {
  llvm.return
}

// CHECK-LABEL: llvm.func @test_byval_read_only
// CHECK-NOT: llvm.call
// CHECK-NEXT: llvm.return
llvm.func @test_byval_read_only(%ptr : !llvm.ptr) {
  llvm.call @with_byval_arg(%ptr) : (!llvm.ptr) -> ()
  llvm.return
}

// -----

llvm.func @with_byval_arg(%ptr : !llvm.ptr { llvm.byval = f64 }) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = write, inaccessibleMem = readwrite>} {
  llvm.return
}

// CHECK-LABEL: llvm.func @test_byval_write_only
// CHECK-SAME: %[[PTR:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: %[[ALLOCA:.+]] = llvm.alloca %{{.+}} x f64
// CHECK: "llvm.intr.memcpy"(%[[ALLOCA]], %[[PTR]]
llvm.func @test_byval_write_only(%ptr : !llvm.ptr) {
  llvm.call @with_byval_arg(%ptr) : (!llvm.ptr) -> ()
  llvm.return
}

// -----

llvm.func @aligned_byval_arg(%ptr : !llvm.ptr { llvm.byval = i16, llvm.align = 16 }) attributes {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>} {
  llvm.return
}

// CHECK-LABEL: llvm.func @test_byval_input_aligned
// CHECK-SAME: %[[UNALIGNED:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK-SAME: %[[ALIGNED:[a-zA-Z0-9_]+]]: !llvm.ptr
llvm.func @test_byval_input_aligned(%unaligned : !llvm.ptr, %aligned : !llvm.ptr { llvm.align = 16 }) {
  // Make sure only the unaligned input triggers a memcpy.
  // CHECK: %[[ALLOCA:.+]] = llvm.alloca %{{.+}} x i16 {alignment = 16
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCA]], %[[UNALIGNED]]
  llvm.call @aligned_byval_arg(%unaligned) : (!llvm.ptr) -> ()
  // CHECK-NOT: memcpy
  llvm.call @aligned_byval_arg(%aligned) : (!llvm.ptr) -> ()
  llvm.return
}

// -----

llvm.func @func_that_uses_ptr(%ptr : !llvm.ptr)

llvm.func @aligned_byval_arg(%ptr : !llvm.ptr { llvm.byval = i16, llvm.align = 16 }) attributes {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>} {
  llvm.call @func_that_uses_ptr(%ptr) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @test_byval_realign_alloca
llvm.func @test_byval_realign_alloca() {
  %size = llvm.mlir.constant(4 : i64) : i64
  // CHECK-NOT: llvm.alloca{{.+}}alignment = 1
  // CHECK: llvm.alloca {{.+}}alignment = 16 : i64
  // CHECK-NOT: llvm.intr.memcpy
  %unaligned = llvm.alloca %size x i16 { alignment = 1 } : (i64) -> !llvm.ptr
  llvm.call @aligned_byval_arg(%unaligned) : (!llvm.ptr) -> ()
  llvm.return
}

// -----

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.stack_alignment", 32 : i32>>
} {

llvm.func @func_that_uses_ptr(%ptr : !llvm.ptr)

llvm.func @aligned_byval_arg(%ptr : !llvm.ptr { llvm.byval = i16, llvm.align = 16 }) attributes {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>} {
  llvm.call @func_that_uses_ptr(%ptr) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @test_exceeds_natural_stack_alignment
llvm.func @test_exceeds_natural_stack_alignment() {
  %size = llvm.mlir.constant(4 : i64) : i64
  // Natural stack alignment is exceeded, so prefer a copy instead of
  // triggering a dynamic stack realignment.
  // CHECK-DAG: %[[SRC:[a-zA-Z0-9_]+]] = llvm.alloca{{.+}}alignment = 2
  // CHECK-DAG: %[[DST:[a-zA-Z0-9_]+]] = llvm.alloca{{.+}}alignment = 16
  // CHECK: "llvm.intr.memcpy"(%[[DST]], %[[SRC]]
  %unaligned = llvm.alloca %size x i16 { alignment = 2 } : (i64) -> !llvm.ptr
  llvm.call @aligned_byval_arg(%unaligned) : (!llvm.ptr) -> ()
  llvm.return
}

}

// -----

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.stack_alignment", 32 : i32>>
} {

llvm.func @func_that_uses_ptr(%ptr : !llvm.ptr)

llvm.func @aligned_byval_arg(%ptr : !llvm.ptr { llvm.byval = i16, llvm.align = 16 }) attributes {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>} {
  llvm.call @func_that_uses_ptr(%ptr) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @test_alignment_exceeded_anyway
llvm.func @test_alignment_exceeded_anyway() {
  %size = llvm.mlir.constant(4 : i64) : i64
  // Natural stack alignment is lower than the target alignment, but the
  // alloca's existing alignment already exceeds it, so we might as well avoid
  // the copy.
  // CHECK-NOT: llvm.alloca{{.+}}alignment = 1
  // CHECK: llvm.alloca {{.+}}alignment = 16 : i64
  // CHECK-NOT: llvm.intr.memcpy
  %unaligned = llvm.alloca %size x i16 { alignment = 8 } : (i64) -> !llvm.ptr
  llvm.call @aligned_byval_arg(%unaligned) : (!llvm.ptr) -> ()
  llvm.return
}

}

// -----

llvm.mlir.global private @unaligned_global(42 : i64) : i64
llvm.mlir.global private @aligned_global(42 : i64) { alignment = 64 } : i64

llvm.func @aligned_byval_arg(%ptr : !llvm.ptr { llvm.byval = i16, llvm.align = 16 }) attributes {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>} {
  llvm.return
}

// CHECK-LABEL: llvm.func @test_byval_global
llvm.func @test_byval_global() {
  // Make sure only the unaligned global triggers a memcpy.
  // CHECK-DAG: %[[UNALIGNED:.+]] = llvm.mlir.addressof @unaligned_global
  // CHECK-DAG: %[[ALLOCA:.+]] = llvm.alloca
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCA]], %[[UNALIGNED]]
  // CHECK-NOT: llvm.alloca
  %unaligned = llvm.mlir.addressof @unaligned_global : !llvm.ptr
  llvm.call @aligned_byval_arg(%unaligned) : (!llvm.ptr) -> ()
  %aligned = llvm.mlir.addressof @aligned_global : !llvm.ptr
  llvm.call @aligned_byval_arg(%aligned) : (!llvm.ptr) -> ()
  llvm.return
}

// -----

llvm.func @ignored_attrs(%ptr : !llvm.ptr { llvm.inreg, llvm.nocapture, llvm.nofree, llvm.preallocated = i32, llvm.returned, llvm.alignstack = 32 : i64, llvm.writeonly, llvm.noundef, llvm.nonnull }, %x : i32 { llvm.zeroext }) -> (!llvm.ptr { llvm.noundef, llvm.inreg, llvm.nonnull }) {
  llvm.return %ptr : !llvm.ptr
}

// CHECK-LABEL: @test_ignored_attrs
// CHECK-NOT: llvm.call
// CHECK-NEXT: llvm.return
llvm.func @test_ignored_attrs(%ptr : !llvm.ptr, %x : i32) {
  llvm.call @ignored_attrs(%ptr, %x) : (!llvm.ptr, i32) -> (!llvm.ptr)
  llvm.return
}

// -----

llvm.func @disallowed_arg_attr(%ptr : !llvm.ptr { llvm.inalloca = i64 }) {
  llvm.return
}

// CHECK-LABEL: @test_disallow_arg_attr
// CHECK-NEXT: llvm.call
llvm.func @test_disallow_arg_attr(%ptr : !llvm.ptr) {
  llvm.call @disallowed_arg_attr(%ptr) : (!llvm.ptr) -> ()
  llvm.return
}

// -----

#callee = #llvm.access_group<id = distinct[0]<>>
#caller = #llvm.access_group<id = distinct[1]<>>

llvm.func @inlinee(%ptr : !llvm.ptr) -> i32 {
  %0 = llvm.load %ptr { access_groups = [#callee] } : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK-DAG: #[[$CALLEE:.*]] = #llvm.access_group<id = {{.*}}>
// CHECK-DAG: #[[$CALLER:.*]] = #llvm.access_group<id = {{.*}}>

// CHECK-LABEL: func @caller
// CHECK: llvm.load
// CHECK-SAME: access_groups = [#[[$CALLEE]], #[[$CALLER]]]
llvm.func @caller(%ptr : !llvm.ptr) -> i32 {
  %0 = llvm.call @inlinee(%ptr) { access_groups = [#caller] } : (!llvm.ptr) -> (i32)
  llvm.return %0 : i32
}

// -----

#caller = #llvm.access_group<id = distinct[1]<>>

llvm.func @inlinee(%ptr : !llvm.ptr) -> i32 {
  %0 = llvm.load %ptr : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK-DAG: #[[$CALLER:.*]] = #llvm.access_group<id = {{.*}}>

// CHECK-LABEL: func @caller
// CHECK: llvm.load
// CHECK-SAME: access_groups = [#[[$CALLER]]]
// CHECK: llvm.store
// CHECK-SAME: access_groups = [#[[$CALLER]]]
llvm.func @caller(%ptr : !llvm.ptr) -> i32 {
  %c5 = llvm.mlir.constant(5 : i32) : i32
  %0 = llvm.call @inlinee(%ptr) { access_groups = [#caller] } : (!llvm.ptr) -> (i32)
  llvm.store %c5, %ptr { access_groups = [#caller] } : i32, !llvm.ptr
  llvm.return %0 : i32
}

// -----

llvm.func @vararg_func(...) {
  llvm.return
}

llvm.func @vararg_intrinrics() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %list = llvm.alloca %0 x !llvm.struct<"struct.va_list_opaque", (ptr)> : (i32) -> !llvm.ptr
  // The vararg intinriscs should normally be part of a variadic function.
  // However, this test uses a non-variadic function to ensure the presence of
  // the intrinsic alone suffices to prevent inlining.
  llvm.intr.vastart %list : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: func @caller
llvm.func @caller() {
  // CHECK-NEXT: llvm.call @vararg_func()
  llvm.call @vararg_func() vararg(!llvm.func<void (...)>) : () -> ()
  // CHECK-NEXT: llvm.call @vararg_intrinrics()
  llvm.call @vararg_intrinrics() : () -> ()
  llvm.return
}
