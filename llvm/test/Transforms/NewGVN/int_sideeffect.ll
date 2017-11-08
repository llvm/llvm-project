; RUN: opt -S < %s -newgvn | FileCheck %s

declare void @llvm.sideeffect()

; Store-to-load forwarding across a @llvm.sideeffect.

; CHECK-LABEL: s2l
; CHECK-NOT: load
define float @s2l(float* %p) {
    store float 0.0, float* %p
    call void @llvm.sideeffect()
    %t = load float, float* %p
    ret float %t
}

; Redundant load elimination across a @llvm.sideeffect.

; CHECK-LABEL: rle
; CHECK: load
; CHECK-NOT: load
define float @rle(float* %p) {
    %r = load float, float* %p
    call void @llvm.sideeffect()
    %s = load float, float* %p
    %t = fadd float %r, %s
    ret float %t
}
