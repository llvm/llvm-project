; RUN: opt -S < %s -passes=instcombine | FileCheck %s

declare void @llvm.sideeffect()

; Don't DCE llvm.sideeffect calls.

; CHECK-LABEL: dce
; CHECK: call void @llvm.sideeffect()
define void @dce() {
    call void @llvm.sideeffect()
    ret void
}
