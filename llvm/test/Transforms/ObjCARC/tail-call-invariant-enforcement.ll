; RUN: opt -objc-arc -S < %s | FileCheck %s

declare void @objc_release(i8* %x)
declare i8* @objc_retain(i8* %x)
declare i8* @objc_autorelease(i8* %x)
declare i8* @objc_autoreleaseReturnValue(i8* %x)
declare i8* @objc_retainAutoreleasedReturnValue(i8* %x)
declare i8* @tmp(i8*)

; Never tail call objc_autorelease.

; CHECK: define i8* @test0(i8* %x) [[NUW:#[0-9]+]] {
; CHECK: %tmp0 = call i8* @objc_autorelease(i8* %x) [[NUW]]
; CHECK: %tmp1 = call i8* @objc_autorelease(i8* %x) [[NUW]]
; CHECK: }
define i8* @test0(i8* %x) nounwind {
entry:
  %tmp0 = call i8* @objc_autorelease(i8* %x)
  %tmp1 = tail call i8* @objc_autorelease(i8* %x)

  ret i8* %x
}

; Always tail call autoreleaseReturnValue.

; CHECK: define i8* @test1(i8* %x) [[NUW]] {
; CHECK: %tmp0 = tail call i8* @objc_autoreleaseReturnValue(i8* %x) [[NUW]]
; CHECK: %tmp1 = tail call i8* @objc_autoreleaseReturnValue(i8* %x) [[NUW]]
; CHECK: }
define i8* @test1(i8* %x) nounwind {
entry:
  %tmp0 = call i8* @objc_autoreleaseReturnValue(i8* %x)
  %tmp1 = tail call i8* @objc_autoreleaseReturnValue(i8* %x)
  ret i8* %x
}

; Always tail call objc_retain.

; CHECK: define i8* @test2(i8* %x) [[NUW]] {
; CHECK: %tmp0 = tail call i8* @objc_retain(i8* %x) [[NUW]]
; CHECK: %tmp1 = tail call i8* @objc_retain(i8* %x) [[NUW]]
; CHECK: }
define i8* @test2(i8* %x) nounwind {
entry:
  %tmp0 = call i8* @objc_retain(i8* %x)
  %tmp1 = tail call i8* @objc_retain(i8* %x)
  ret i8* %x
}

; Always tail call objc_retainAutoreleasedReturnValue.
; CHECK: define i8* @test3(i8* %x) [[NUW]] {
; CHECK: %tmp0 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %y) [[NUW]]
; CHECK: %tmp1 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %z) [[NUW]]
; CHECK: }
define i8* @test3(i8* %x) nounwind {
entry:
  %y = call i8* @tmp(i8* %x)
  %tmp0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %y)
  %z = call i8* @tmp(i8* %x)
  %tmp1 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %z)
  ret i8* %x
}

; By itself, we should never change whether or not objc_release is tail called.

; CHECK: define void @test4(i8* %x) [[NUW]] {
; CHECK: call void @objc_release(i8* %x) [[NUW]]
; CHECK: tail call void @objc_release(i8* %x) [[NUW]]
; CHECK: }
define void @test4(i8* %x) nounwind {
entry:
  call void @objc_release(i8* %x)
  tail call void @objc_release(i8* %x)
  ret void
}

; If we convert a tail called @objc_autoreleaseReturnValue to an
; @objc_autorelease, ensure that the tail call is removed.
; CHECK: define i8* @test5(i8* %x) [[NUW]] {
; CHECK: %tmp0 = call i8* @objc_autorelease(i8* %x) [[NUW]]
; CHECK: }
define i8* @test5(i8* %x) nounwind {
entry:
  %tmp0 = tail call i8* @objc_autoreleaseReturnValue(i8* %x)
  ret i8* %tmp0
}

; CHECK: attributes [[NUW]] = { nounwind }

