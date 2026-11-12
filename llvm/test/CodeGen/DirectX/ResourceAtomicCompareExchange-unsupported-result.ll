; RUN: split-file %s %t
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/success.ll 2>&1 | FileCheck %t/success.ll
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/pair.ll 2>&1 | FileCheck %t/pair.ll

; The DXIL AtomicCompareExchange op returns only the original value, so a
; shader can neither read the success flag of a cmpxchg nor keep the pair.

;--- success.ll

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL cmpxchg provides only the original value
define i1 @cmpxchg_success(i32 %index, i32 %cmp, i32 %value) {
  %buffer = call target("dx.RawBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i32, 1, 0, 0) %buffer, i32 %index)
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %value monotonic monotonic
  %ok = extractvalue { i32, i1 } %pair, 1
  ret i1 %ok
}

;--- pair.ll

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL cmpxchg provides only the original value
define { i32, i1 } @cmpxchg_pair(i32 %index, i32 %cmp, i32 %value) {
  %buffer = call target("dx.RawBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i32, 1, 0, 0) %buffer, i32 %index)
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %value monotonic monotonic
  ret { i32, i1 } %pair
}