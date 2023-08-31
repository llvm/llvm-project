; RUN: opt -S -dxil-prepare < %s | FileCheck %s
target triple = "dxil-unknown-unknown"

; Make sure not crash when has typed ptr.
; CHECK:define i64 @test(ptr %p)
; Make sure no bitcast generated.
; CHECK-NOT:bitcast

@gs = external addrspace(3) global [20 x [6 x float]], align 4

define i64 @test(i64* %p) {
  %base = getelementptr inbounds [20 x [6 x float]], ptr addrspace(3) @gs, i64 0, i64 3
  %addr = getelementptr inbounds [6 x float], ptr addrspace(3) %base, i64 0, i64 2
  store float 1.000000e+00, ptr addrspace(3) %addr, align 4
  %v = load i64, i64* %p
  ret i64 %v
}
