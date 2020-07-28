; RUN: opt -S < %s -function-attrs | FileCheck %s
; RUN: opt -S < %s -passes=function-attrs | FileCheck %s

; CHECK: Function Attrs
; CHECK-SAME: inaccessiblememonly
; CHECK-NEXT: declare void @llvm.sideeffect()
declare void @llvm.sideeffect()

; Don't add readnone or similar attributes when an @llvm.sideeffect() intrinsic
; is present.

; CHECK: Function Attrs
; CHECK-NOT: readnone
; CHECK: define void @test()
define void @test() {
    call void @llvm.sideeffect()
    ret void
}

; CHECK: Function Attrs
; CHECK-NOT: readnone
; CHECK: define void @loop()
define void @loop() {
    br label %loop

loop:
    call void @llvm.sideeffect()
    br label %loop
}
