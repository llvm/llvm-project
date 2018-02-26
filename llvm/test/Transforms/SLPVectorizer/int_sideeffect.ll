; RUN: opt -S < %s -slp-vectorizer -slp-max-reg-size=128 -slp-min-reg-size=128 | FileCheck %s

declare void @llvm.sideeffect()

; SLP vectorization across a @llvm.sideeffect.

; CHECK-LABEL: test
; CHECK: store <4 x float>
define void @test(float* %p) {
    %p0 = getelementptr float, float* %p, i64 0
    %p1 = getelementptr float, float* %p, i64 1
    %p2 = getelementptr float, float* %p, i64 2
    %p3 = getelementptr float, float* %p, i64 3
    %l0 = load float, float* %p0
    %l1 = load float, float* %p1
    %l2 = load float, float* %p2
    call void @llvm.sideeffect()
    %l3 = load float, float* %p3
    store float %l0, float* %p0
    call void @llvm.sideeffect()
    store float %l1, float* %p1
    store float %l2, float* %p2
    store float %l3, float* %p3
    ret void
}
