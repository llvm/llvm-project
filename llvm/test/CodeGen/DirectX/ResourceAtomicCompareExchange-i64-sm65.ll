; RUN: not opt -S -dxil-resource-access -dxil-op-lower -mtriple=dxil-pc-shadermodel6.5-compute %s 2>&1 | FileCheck %s

; Verify resource i64 cmpxchg rejects shader models before SM 6.6, where
; dx.op.atomicCompareExchange gained i64 overload support.

target triple = "dxil-pc-shadermodel6.5-compute"

define i64 @cmpxchg_i64(i32 %index, i64 %cmp, i64 %value) {
  %buffer = call target("dx.RawBuffer", i64, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i64, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: Cannot create AtomicCompareExchange operation: Invalid overload type
  %pair = cmpxchg ptr %ptr, i64 %cmp, i64 %value monotonic monotonic
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}
