; RUN: opt -S -dxil-prepare < %s | FileCheck %s
target triple = "dxil-unknown-unknown"

@gs = external addrspace(3) global [20 x [6 x float]], align 4

; Make sure not crash when has typed ptr.
define i64 @test(i64* %p) {
; CHECK-LABEL: define i64 @test(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[P]], align 4
; CHECK-NEXT:    ret i64 [[V]]
;
  %v = load i64, i64* %p
  ret i64 %v
}

; Make sure no bitcast generated.
define void @test_gep() {
; CHECK-LABEL: define void @test_gep() {
; CHECK-NEXT:    [[BASE:%.*]] = getelementptr inbounds [20 x [6 x float]], ptr addrspace(3) @gs, i64 0, i64 3
; CHECK-NEXT:    [[ADDR:%.*]] = getelementptr inbounds [6 x float], ptr addrspace(3) [[BASE]], i64 0, i64 2
; CHECK-NEXT:    store float 1.000000e+00, ptr addrspace(3) [[ADDR]], align 4
; CHECK-NEXT:    ret void
;
  %base = getelementptr inbounds [20 x [6 x float]], ptr addrspace(3) @gs, i64 0, i64 3
  %addr = getelementptr inbounds [6 x float], ptr addrspace(3) %base, i64 0, i64 2
  store float 1.000000e+00, ptr addrspace(3) %addr, align 4
  ret void
}
