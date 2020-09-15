; RUN: opt -S < %s -dse -enable-dse-memoryssa | FileCheck %s

declare void @llvm.sideeffect()

; Dead store elimination across a @llvm.sideeffect.

; CHECK-LABEL: dse
; CHECK: store
; CHECK-NOT: store
define void @dse(float* %p) {
    store float 0.0, float* %p
    call void @llvm.sideeffect()
    store float 0.0, float* %p
    ret void
}
