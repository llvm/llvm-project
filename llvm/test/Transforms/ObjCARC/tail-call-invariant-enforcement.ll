; RUN: opt -objc-arc -S < %s | FileCheck %s

declare void @llvm.objc.release(i8* %x)
declare i8* @llvm.objc.retain(i8* %x)
declare i8* @llvm.objc.autorelease(i8* %x)
declare i8* @llvm.objc.autoreleaseReturnValue(i8* %x)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %x)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %x)
declare i8* @tmp(i8*)

; Never tail call objc_autorelease.

; CHECK: define i8* @test0(i8* %x) [[NUW:#[0-9]+]] {
; CHECK: %tmp0 = call i8* @llvm.objc.autorelease(i8* %x) [[NUW]]
; CHECK: %tmp1 = call i8* @llvm.objc.autorelease(i8* %x) [[NUW]]
; CHECK: }
define i8* @test0(i8* %x) nounwind {
entry:
  %tmp0 = call i8* @llvm.objc.autorelease(i8* %x)
  %tmp1 = tail call i8* @llvm.objc.autorelease(i8* %x)

  ret i8* %x
}

; Always tail call autoreleaseReturnValue.

; CHECK: define i8* @test1(i8* %x) [[NUW]] {
; CHECK: %tmp0 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) [[NUW]]
; CHECK: %tmp1 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) [[NUW]]
; CHECK: }
define i8* @test1(i8* %x) nounwind {
entry:
  %tmp0 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x)
  %tmp1 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %x)
  ret i8* %x
}

; Always tail call objc_retain.

; CHECK: define i8* @test2(i8* %x) [[NUW]] {
; CHECK: %tmp0 = tail call i8* @llvm.objc.retain(i8* %x) [[NUW]]
; CHECK: %tmp1 = tail call i8* @llvm.objc.retain(i8* %x) [[NUW]]
; CHECK: }
define i8* @test2(i8* %x) nounwind {
entry:
  %tmp0 = call i8* @llvm.objc.retain(i8* %x)
  %tmp1 = tail call i8* @llvm.objc.retain(i8* %x)
  ret i8* %x
}

; Always tail call objc_retainAutoreleasedReturnValue unless it's annotated with
; notail.
; CHECK: define i8* @test3(i8* %x) [[NUW]] {
; CHECK: %tmp0 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %y) [[NUW]]
; CHECK: %tmp1 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %z) [[NUW]]
; CHECK: %tmp2 = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %z2) [[NUW]]
; CHECK: }
define i8* @test3(i8* %x) nounwind {
entry:
  %y = call i8* @tmp(i8* %x)
  %tmp0 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %y)
  %z = call i8* @tmp(i8* %x)
  %tmp1 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %z)
  %z2 = call i8* @tmp(i8* %x)
  %tmp2 = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %z2)
  ret i8* %x
}

; By itself, we should never change whether or not objc_release is tail called.

; CHECK: define void @test4(i8* %x) [[NUW]] {
; CHECK: call void @llvm.objc.release(i8* %x) [[NUW]]
; CHECK: tail call void @llvm.objc.release(i8* %x) [[NUW]]
; CHECK: }
define void @test4(i8* %x) nounwind {
entry:
  call void @llvm.objc.release(i8* %x)
  tail call void @llvm.objc.release(i8* %x)
  ret void
}

; If we convert a tail called @llvm.objc.autoreleaseReturnValue to an
; @llvm.objc.autorelease, ensure that the tail call is removed.
; CHECK: define i8* @test5(i8* %x) [[NUW]] {
; CHECK: %tmp0 = call i8* @llvm.objc.autorelease(i8* %x) [[NUW]]
; CHECK: }
define i8* @test5(i8* %x) nounwind {
entry:
  %tmp0 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %x)
  ret i8* %tmp0
}

; Always tail call llvm.objc.unsafeClaimAutoreleasedReturnValue.
; CHECK: define i8* @test6(i8* %x) [[NUW]] {
; CHECK: %tmp0 = tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %y) [[NUW]]
; CHECK: %tmp1 = tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %z) [[NUW]]
; CHECK: }
define i8* @test6(i8* %x) nounwind {
entry:
  %y = call i8* @tmp(i8* %x)
  %tmp0 = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %y)
  %z = call i8* @tmp(i8* %x)
  %tmp1 = tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %z)
  ret i8* %x
}

; CHECK: attributes [[NUW]] = { nounwind }

