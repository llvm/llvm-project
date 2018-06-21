; RUN: opt -S < %s -functionattrs | FileCheck %s

declare void @llvm.sideeffect()

; Don't add readnone or similar attributes when an @llvm.sideeffect() intrinsic
; is present.

; CHECK: define void @test() {
define void @test() {
    call void @llvm.sideeffect()
    ret void
}

; CHECK: define void @loop() {
define void @loop() {
    br label %loop

loop:
    call void @llvm.sideeffect()
    br label %loop
}
