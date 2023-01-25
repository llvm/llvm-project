; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

declare void @llvm.objc.release(ptr %x)
declare ptr @llvm.objc.retain(ptr %x)
declare ptr @llvm.objc.autorelease(ptr %x)
declare ptr @llvm.objc.autoreleaseReturnValue(ptr %x)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %x)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %x)
declare ptr @tmp(ptr)

; Never tail call objc_autorelease.

; CHECK: define ptr @test0(ptr %x) [[NUW:#[0-9]+]] {
; CHECK: %tmp0 = call ptr @llvm.objc.autorelease(ptr %x) [[NUW]]
; CHECK: %tmp1 = call ptr @llvm.objc.autorelease(ptr %x) [[NUW]]
; CHECK: }
define ptr @test0(ptr %x) nounwind {
entry:
  %tmp0 = call ptr @llvm.objc.autorelease(ptr %x)
  %tmp1 = tail call ptr @llvm.objc.autorelease(ptr %x)

  ret ptr %x
}

; Always tail call autoreleaseReturnValue.

; CHECK: define ptr @test1(ptr %x) [[NUW]] {
; CHECK: %tmp0 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) [[NUW]]
; CHECK: %tmp1 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) [[NUW]]
; CHECK: }
define ptr @test1(ptr %x) nounwind {
entry:
  %tmp0 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x)
  %tmp1 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %x)
  ret ptr %x
}

; Always tail call objc_retain.

; CHECK: define ptr @test2(ptr %x) [[NUW]] {
; CHECK: %tmp0 = tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK: %tmp1 = tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK: }
define ptr @test2(ptr %x) nounwind {
entry:
  %tmp0 = call ptr @llvm.objc.retain(ptr %x)
  %tmp1 = tail call ptr @llvm.objc.retain(ptr %x)
  ret ptr %x
}

; Always tail call objc_retainAutoreleasedReturnValue unless it's annotated with
; notail.
; CHECK: define ptr @test3(ptr %x) [[NUW]] {
; CHECK: %tmp0 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %y) [[NUW]]
; CHECK: %tmp1 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %z) [[NUW]]
; CHECK: %tmp2 = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %z2) [[NUW]]
; CHECK: }
define ptr @test3(ptr %x) nounwind {
entry:
  %y = call ptr @tmp(ptr %x)
  %tmp0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %y)
  %z = call ptr @tmp(ptr %x)
  %tmp1 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %z)
  %z2 = call ptr @tmp(ptr %x)
  %tmp2 = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %z2)
  ret ptr %x
}

; By itself, we should never change whether or not objc_release is tail called.

; CHECK: define void @test4(ptr %x) [[NUW]] {
; CHECK: call void @llvm.objc.release(ptr %x) [[NUW]]
; CHECK: tail call void @llvm.objc.release(ptr %x) [[NUW]]
; CHECK: }
define void @test4(ptr %x) nounwind {
entry:
  call void @llvm.objc.release(ptr %x)
  tail call void @llvm.objc.release(ptr %x)
  ret void
}

; If we convert a tail called @llvm.objc.autoreleaseReturnValue to an
; @llvm.objc.autorelease, ensure that the tail call is removed.
; CHECK: define ptr @test5(ptr %x) [[NUW]] {
; CHECK: %tmp0 = call ptr @llvm.objc.autorelease(ptr %x) [[NUW]]
; CHECK: }
define ptr @test5(ptr %x) nounwind {
entry:
  %tmp0 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %x)
  ret ptr %tmp0
}

; Always tail call llvm.objc.unsafeClaimAutoreleasedReturnValue.
; CHECK: define ptr @test6(ptr %x) [[NUW]] {
; CHECK: %tmp0 = tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %y) [[NUW]]
; CHECK: %tmp1 = tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %z) [[NUW]]
; CHECK: }
define ptr @test6(ptr %x) nounwind {
entry:
  %y = call ptr @tmp(ptr %x)
  %tmp0 = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %y)
  %z = call ptr @tmp(ptr %x)
  %tmp1 = tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %z)
  ret ptr %x
}

; CHECK: attributes [[NUW]] = { nounwind }

