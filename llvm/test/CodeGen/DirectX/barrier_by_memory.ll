; RUN: opt -S -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.8-library < %s | FileCheck %s

define void @barrier_by_memory_type() {
  ; CHECK-LABEL: define void @barrier_by_memory_type()
  ; CHECK: call void @dx.op.barrierByMemoryType(i32 244, i32 3, i32 5)
  call void @llvm.dx.barrier.by.memory.type(i32 3, i32 5)
  ret void
}

define void @barrier_by_memory_handle() {
  ; CHECK-LABEL: define void @barrier_by_memory_handle()
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle
  ; CHECK-SAME: @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[ANNOTATED:%.*]] = call %dx.types.Handle
  ; CHECK-SAME: @dx.op.annotateHandle(i32 216, %dx.types.Handle [[HANDLE]],
  ; CHECK: call void @dx.op.barrierByMemoryHandle(
  ; CHECK-SAME: i32 245, %dx.types.Handle [[ANNOTATED]], i32 4)
  %buffer = call target("dx.RawBuffer", i8, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  call void @llvm.dx.barrier.by.memory.handle.tdx.RawBuffer_i8_1_0t(
      target("dx.RawBuffer", i8, 1, 0) %buffer, i32 4)
  ret void
}

declare void @llvm.dx.barrier.by.memory.type(i32, i32)
declare target("dx.RawBuffer", i8, 1, 0)
    @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(
        i32, i32, i32, i32, ptr)
declare void @llvm.dx.barrier.by.memory.handle.tdx.RawBuffer_i8_1_0t(
    target("dx.RawBuffer", i8, 1, 0), i32)
