; RUN: opt -S -dxil-resource-access -dxil-op-lower -mtriple=dxil-pc-shadermodel6.6-compute %s | FileCheck %s

; InterlockedCompareStore reports nothing, so it emits a `cmpxchg` whose result
; is unused. Lowering must still produce the DXIL AtomicCompareExchange op.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @bab_compare_store
define void @bab_compare_store(i32 %offset, i32 %cmp, i32 %val) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)
  ; CHECK: call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %offset, i32 poison, i32 poison, i32 %cmp, i32 %val)
  %old = cmpxchg ptr %ptr, i32 %cmp, i32 %val monotonic monotonic
  ret void
}

; The same call keeping the result proves the unused case above is not the only
; shape that lowers. The DXIL op returns only the original value, so the user
; of the `cmpxchg` result reads that value directly.
; CHECK-LABEL: define i32 @bab_compare_exchange
define i32 @bab_compare_exchange(i32 %offset, i32 %cmp, i32 %val) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)
  ; CHECK: [[OLD:%.*]] = call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %offset, i32 poison, i32 poison, i32 %cmp, i32 %val)
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %val monotonic monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ; CHECK: ret i32 [[OLD]]
  ret i32 %old
}
