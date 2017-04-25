; RUN: opt -basicaa -objc-arc -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare i8* @objc_retain(i8*)
declare i8* @objc_retainAutorelease(i8*)
declare void @objc_release(i8*)
declare i8* @objc_autorelease(i8*)

declare void @clang.arc.use(...)

declare void @test0_helper(i8*, i8**)

; Ensure that we honor clang.arc.use as a use and don't miscompile
; the reduced test case from <rdar://13195034>.
;
; CHECK-LABEL:      define void @test0(
; CHECK:        @objc_retain(i8* %x)
; CHECK-NEXT:   store i8* %y, i8** %temp0
; CHECK-NEXT:   @objc_retain(i8* %y)
; CHECK-NEXT:   call void @test0_helper
; CHECK-NEXT:   [[VAL1:%.*]] = load i8*, i8** %temp0
; CHECK-NEXT:   @objc_retain(i8* [[VAL1]])
; CHECK-NEXT:   call void (...) @clang.arc.use(i8* %y)
; CHECK-NEXT:   @objc_release(i8* %y)
; CHECK-NEXT:   store i8* [[VAL1]], i8** %temp1
; CHECK-NEXT:   call void @test0_helper
; CHECK-NEXT:   [[VAL2:%.*]] = load i8*, i8** %temp1
; CHECK-NEXT:   @objc_retain(i8* [[VAL2]])
; CHECK-NEXT:   call void (...) @clang.arc.use(i8* [[VAL1]])
; CHECK-NEXT:   @objc_release(i8* [[VAL1]])
; CHECK-NEXT:   @objc_autorelease(i8* %x)
; CHECK-NEXT:   store i8* %x, i8** %out
; CHECK-NEXT:   @objc_retain(i8* %x)
; CHECK-NEXT:   @objc_release(i8* [[VAL2]])
; CHECK-NEXT:   @objc_release(i8* %x)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test0(i8** %out, i8* %x, i8* %y) {
entry:
  %temp0 = alloca i8*, align 8
  %temp1 = alloca i8*, align 8
  %0 = call i8* @objc_retain(i8* %x) nounwind
  %1 = call i8* @objc_retain(i8* %y) nounwind
  store i8* %y, i8** %temp0
  call void @test0_helper(i8* %x, i8** %temp0)
  %val1 = load i8*, i8** %temp0
  %2 = call i8* @objc_retain(i8* %val1) nounwind
  call void (...) @clang.arc.use(i8* %y) nounwind
  call void @objc_release(i8* %y) nounwind
  store i8* %val1, i8** %temp1
  call void @test0_helper(i8* %x, i8** %temp1)
  %val2 = load i8*, i8** %temp1
  %3 = call i8* @objc_retain(i8* %val2) nounwind
  call void (...) @clang.arc.use(i8* %val1) nounwind
  call void @objc_release(i8* %val1) nounwind
  %4 = call i8* @objc_retain(i8* %x) nounwind
  %5 = call i8* @objc_autorelease(i8* %x) nounwind
  store i8* %x, i8** %out
  call void @objc_release(i8* %val2) nounwind
  call void @objc_release(i8* %x) nounwind
  ret void
}

; CHECK-LABEL:      define void @test0a(
; CHECK:        @objc_retain(i8* %x)
; CHECK-NEXT:   store i8* %y, i8** %temp0
; CHECK-NEXT:   @objc_retain(i8* %y)
; CHECK-NEXT:   call void @test0_helper
; CHECK-NEXT:   [[VAL1:%.*]] = load i8*, i8** %temp0
; CHECK-NEXT:   @objc_retain(i8* [[VAL1]])
; CHECK-NEXT:   call void (...) @clang.arc.use(i8* %y)
; CHECK-NEXT:   @objc_release(i8* %y)
; CHECK-NEXT:   store i8* [[VAL1]], i8** %temp1
; CHECK-NEXT:   call void @test0_helper
; CHECK-NEXT:   [[VAL2:%.*]] = load i8*, i8** %temp1
; CHECK-NEXT:   @objc_retain(i8* [[VAL2]])
; CHECK-NEXT:   call void (...) @clang.arc.use(i8* [[VAL1]])
; CHECK-NEXT:   @objc_release(i8* [[VAL1]])
; CHECK-NEXT:   @objc_autorelease(i8* %x)
; CHECK-NEXT:   @objc_release(i8* [[VAL2]])
; CHECK-NEXT:   store i8* %x, i8** %out
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test0a(i8** %out, i8* %x, i8* %y) {
entry:
  %temp0 = alloca i8*, align 8
  %temp1 = alloca i8*, align 8
  %0 = call i8* @objc_retain(i8* %x) nounwind
  %1 = call i8* @objc_retain(i8* %y) nounwind
  store i8* %y, i8** %temp0
  call void @test0_helper(i8* %x, i8** %temp0)
  %val1 = load i8*, i8** %temp0
  %2 = call i8* @objc_retain(i8* %val1) nounwind
  call void (...) @clang.arc.use(i8* %y) nounwind
  call void @objc_release(i8* %y) nounwind, !clang.imprecise_release !0
  store i8* %val1, i8** %temp1
  call void @test0_helper(i8* %x, i8** %temp1)
  %val2 = load i8*, i8** %temp1
  %3 = call i8* @objc_retain(i8* %val2) nounwind
  call void (...) @clang.arc.use(i8* %val1) nounwind
  call void @objc_release(i8* %val1) nounwind, !clang.imprecise_release !0
  %4 = call i8* @objc_retain(i8* %x) nounwind
  %5 = call i8* @objc_autorelease(i8* %x) nounwind
  store i8* %x, i8** %out
  call void @objc_release(i8* %val2) nounwind, !clang.imprecise_release !0
  call void @objc_release(i8* %x) nounwind, !clang.imprecise_release !0
  ret void
}


!0 = !{}

