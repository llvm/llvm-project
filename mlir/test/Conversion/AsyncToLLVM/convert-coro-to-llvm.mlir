// RUN: mlir-opt %s -convert-async-to-llvm | FileCheck %s

// CHECK-LABEL: @coro_id
func.func @coro_id() {
  // CHECK: %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = llvm.mlir.null : !llvm.ptr<i8>
  // CHECK: %2 = llvm.intr.coro.id %0, %1, %1, %1 : !llvm.token
  %0 = async.coro.id
  return
}

// CHECK-LABEL: @coro_begin
func.func @coro_begin() {
  // CHECK: %[[ID:.*]] = llvm.intr.coro.id
  %0 = async.coro.id
  // CHECK: %[[SIZE:.*]] = llvm.intr.coro.size : i64
  // CHECK: %[[ALIGN:.*]] = llvm.intr.coro.align : i64
  // CHECK: %[[SIZE_PLUS_ALIGN:.*]] = llvm.add %[[SIZE]], %[[ALIGN]] : i64
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[SIZE_PLUS_ALIGN_MINUS_ONE:.*]] = llvm.sub %[[SIZE_PLUS_ALIGN]], %[[C1]] : i64
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[NEGATED_ALIGN:.*]] = llvm.sub %[[C0]], %[[ALIGN]]  : i64
  // CHECK: %[[ROUNDED_SIZE:.*]] = llvm.and %[[SIZE_PLUS_ALIGN_MINUS_ONE]], %[[NEGATED_ALIGN]] : i64
  // CHECK: %[[ALLOC:.*]] = llvm.call @_mlir_aligned_alloc(%[[ALIGN]], %[[ROUNDED_SIZE]])
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin %[[ID]], %[[ALLOC]]
  %1 = async.coro.begin %0
  return
}

// CHECK-LABEL: @coro_free
func.func @coro_free() {
  // CHECK: %[[ID:.*]] = llvm.intr.coro.id
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: %[[MEM:.*]] = llvm.intr.coro.free %[[ID]], %[[HDL]]
  // CHECK: llvm.call @_mlir_aligned_free(%[[MEM]])
  async.coro.free %0, %1
  return
}

// CHECK-LABEL: @coro_end
func.func @coro_end() {
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false) : i1
  // CHECK: llvm.intr.coro.end %[[HDL]], %[[FALSE]]
  async.coro.end %1
  return
}

// CHECK-LABEL: @coro_save
func.func @coro_save() {
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: llvm.intr.coro.save %[[HDL]]
  %2 = async.coro.save %1
  return
}

// CHECK-LABEL: @coro_suspend
func.func @coro_suspend() {
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: %[[STATE:.*]] = llvm.intr.coro.save %[[HDL]]
  %2 = async.coro.save %1

  // CHECK: %[[FINAL:.*]] = llvm.mlir.constant(false) : i1
  // CHECK: %[[RET:.*]] = llvm.intr.coro.suspend %[[STATE]], %[[FINAL]]
  // CHECK: %[[SEXT:.*]] = llvm.sext %[[RET]] : i8 to i32
  // CHECK: llvm.switch %[[SEXT]] : i32, ^[[SUSPEND:[b0-9]+]]
  // CHECK-NEXT: 0: ^[[RESUME:[b0-9]+]]
  // CHECK-NEXT: 1: ^[[CLEANUP:[b0-9]+]]
  async.coro.suspend %2, ^suspend, ^resume, ^cleanup
^resume:
  // CHECK: ^[[RESUME]]
  // CHECK:   return {coro.resume}
  return { coro.resume }
^cleanup:
  // CHECK: ^[[CLEANUP]]
  // CHECK:   return {coro.cleanup}
  return { coro.cleanup }
^suspend:
  // CHECK: ^[[SUSPEND]]
  // CHECK:   return {coro.suspend}
  return { coro.suspend }
}
