; RUN: opt -S < %s -passes=instcombine | FileCheck %s

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
