; RUN: opt -S < %s -passes=instcombine | FileCheck %s

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
