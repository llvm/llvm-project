; RUN: opt -S < %s -passes=newgvn | FileCheck %s

declare void @llvm.sideeffect()

; Store-to-load forwarding across a @llvm.sideeffect.

; CHECK-LABEL: s2l
; CHECK-NOT: load
define float @s2l(ptr %p) {
    store float 0.0, ptr %p
    call void @llvm.sideeffect()
    %t = load float, ptr %p
    ret float %t
}

; Redundant load elimination across a @llvm.sideeffect.

; CHECK-LABEL: rle
; CHECK: load
; CHECK-NOT: load
define float @rle(ptr %p) {
    %r = load float, ptr %p
    call void @llvm.sideeffect()
    %s = load float, ptr %p
    %t = fadd float %r, %s
    ret float %t
}
