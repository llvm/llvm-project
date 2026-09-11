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
  ; CHECK: call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %offset, i32 poison, i32 0, i32 %cmp, i32 %val)
  %old = cmpxchg ptr %ptr, i32 %cmp, i32 %val monotonic monotonic
  ret void
}

; The same call keeping the result proves the unused case above is not the only
; shape that lowers. `cmpxchg` yields a { value, success } pair, so the pass
; rebuilds that pair from the single value the DXIL op returns.
; CHECK-LABEL: define i32 @bab_compare_exchange
define i32 @bab_compare_exchange(i32 %offset, i32 %cmp, i32 %val) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)
  ; CHECK: [[OLD:%.*]] = call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %offset, i32 poison, i32 0, i32 %cmp, i32 %val)
  ; CHECK: [[OK:%.*]] = icmp eq i32 [[OLD]], %cmp
  ; CHECK: [[P0:%.*]] = insertvalue { i32, i1 } poison, i32 [[OLD]], 0
  ; CHECK: [[P1:%.*]] = insertvalue { i32, i1 } [[P0]], i1 [[OK]], 1
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %val monotonic monotonic
  ; CHECK: [[RES:%.*]] = extractvalue { i32, i1 } [[P1]], 0
  %old = extractvalue { i32, i1 } %pair, 0
  ; CHECK: ret i32 [[RES]]
  ret i32 %old
}
