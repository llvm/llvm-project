; RUN: opt -S -dxil-resource-access -dxil-op-lower -mtriple=dxil-pc-shadermodel6.6-compute %s | FileCheck %s

; InterlockedCompareStoreFloatBitwise compares the bit pattern of a float, so
; clang emits an i32 `cmpxchg` on a float resource. Lowering must key the DXIL
; op off the operand type rather than the resource element type.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @typed_buffer_float_compare_store
define void @typed_buffer_float_compare_store(i32 %index, i32 %cmp, i32 %val) {
  %buffer = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.TypedBuffer", float, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %index, i32 poison, i32 0, i32 %cmp, i32 %val)
  %old = cmpxchg ptr %ptr, i32 %cmp, i32 %val monotonic monotonic
  ret void
}

; The raw buffer path carries no element type, so the same i32 op applies.
; CHECK-LABEL: define void @raw_buffer_float_compare_store
define void @raw_buffer_float_compare_store(i32 %offset, i32 %cmp, i32 %val) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)
  ; CHECK: call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %offset, i32 poison, i32 0, i32 %cmp, i32 %val)
  %old = cmpxchg ptr %ptr, i32 %cmp, i32 %val monotonic monotonic
  ret void
}
