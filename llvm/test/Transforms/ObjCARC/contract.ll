; RUN: opt -objc-arc-contract -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare i8* @objc_retain(i8*)
declare void @objc_release(i8*)
declare i8* @objc_autorelease(i8*)
declare i8* @objc_autoreleaseReturnValue(i8*)
declare i8* @objc_retainAutoreleasedReturnValue(i8*)

declare void @use_pointer(i8*)
declare i8* @returner()

; CHECK: define void @test0
; CHECK: call void @use_pointer(i8* %0)
; CHECK: }
define void @test0(i8* %x) nounwind {
entry:
  %0 = call i8* @objc_retain(i8* %x) nounwind
  call void @use_pointer(i8* %x)
  ret void
}

; CHECK: define void @test1
; CHECK: call void @use_pointer(i8* %0)
; CHECK: }
define void @test1(i8* %x) nounwind {
entry:
  %0 = call i8* @objc_autorelease(i8* %x) nounwind
  call void @use_pointer(i8* %x)
  ret void
}

; Merge objc_retain and objc_autorelease into objc_retainAutorelease.

; CHECK: define void @test2(
; CHECK: tail call i8* @objc_retainAutorelease(i8* %x) nounwind
; CHECK: }
define void @test2(i8* %x) nounwind {
entry:
  %0 = tail call i8* @objc_retain(i8* %x) nounwind
  tail call i8* @objc_autorelease(i8* %0) nounwind
  call void @use_pointer(i8* %x)
  ret void
}

; Same as test2 but the value is returned. Do an RV optimization.

; CHECK: define i8* @test2b(
; CHECK: tail call i8* @objc_retainAutoreleaseReturnValue(i8* %x) nounwind
; CHECK: }
define i8* @test2b(i8* %x) nounwind {
entry:
  %0 = tail call i8* @objc_retain(i8* %x) nounwind
  tail call i8* @objc_autoreleaseReturnValue(i8* %0) nounwind
  ret i8* %x
}

; Merge a retain,autorelease pair around a call.

; CHECK: define void @test3(
; CHECK: tail call i8* @objc_retainAutorelease(i8* %x) nounwind
; CHECK: @use_pointer(i8* %0)
; CHECK: }
define void @test3(i8* %x, i64 %n) {
entry:
  tail call i8* @objc_retain(i8* %x) nounwind
  call void @use_pointer(i8* %x)
  tail call i8* @objc_autorelease(i8* %x) nounwind
  ret void
}

; Trivial retain,autorelease pair with intervening call, but it's post-dominated
; by another release. The retain and autorelease can be merged.

; CHECK: define void @test4(
; CHECK-NEXT: entry:
; CHECK-NEXT: @objc_retainAutorelease(i8* %x) nounwind
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @objc_release
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test4(i8* %x, i64 %n) {
entry:
  tail call i8* @objc_retain(i8* %x) nounwind
  call void @use_pointer(i8* %x)
  tail call i8* @objc_autorelease(i8* %x) nounwind
  tail call void @objc_release(i8* %x) nounwind
  ret void
}

; Don't merge retain and autorelease if they're not control-equivalent.

; CHECK: define void @test5(
; CHECK: tail call i8* @objc_retain(i8* %p) nounwind
; CHECK: true:
; CHECK: tail call i8* @objc_autorelease(i8* %0) nounwind
; CHECK: }
define void @test5(i8* %p, i1 %a) {
entry:
  tail call i8* @objc_retain(i8* %p) nounwind
  br i1 %a, label %true, label %false

true:
  tail call i8* @objc_autorelease(i8* %p) nounwind
  call void @use_pointer(i8* %p)
  ret void

false:
  ret void
}

; Don't eliminate objc_retainAutoreleasedReturnValue by merging it into
; an objc_autorelease.
; TODO? Merge objc_retainAutoreleasedReturnValue and objc_autorelease into
; objc_retainAutoreleasedReturnValueAutorelease and merge
; objc_retainAutoreleasedReturnValue and objc_autoreleaseReturnValue
; into objc_retainAutoreleasedReturnValueAutoreleaseReturnValue?
; Those entrypoints don't exist yet though.

; CHECK: define i8* @test6(
; CHECK: call i8* @objc_retainAutoreleasedReturnValue(i8* %p) nounwind
; CHECK: %t = tail call i8* @objc_autoreleaseReturnValue(i8* %1) nounwind
; CHECK: }
define i8* @test6() {
  %p = call i8* @returner()
  tail call i8* @objc_retainAutoreleasedReturnValue(i8* %p) nounwind
  %t = tail call i8* @objc_autoreleaseReturnValue(i8* %p) nounwind
  call void @use_pointer(i8* %t)
  ret i8* %t
}

; Don't spoil the RV optimization.

; CHECK: define i8* @test7(i8* %p)
; CHECK: tail call i8* @objc_retain(i8* %p)
; CHECK: call void @use_pointer(i8* %1)
; CHECK: tail call i8* @objc_autoreleaseReturnValue(i8* %1)
; CHECK: ret i8* %2
define i8* @test7(i8* %p) {
  %1 = tail call i8* @objc_retain(i8* %p)
  call void @use_pointer(i8* %p)
  %2 = tail call i8* @objc_autoreleaseReturnValue(i8* %p)
  ret i8* %p
}

; Do the return value substitution for PHI nodes too.

; CHECK: define i8* @test8(
; CHECK: %retval = phi i8* [ %p, %if.then ], [ null, %entry ]
; CHECK: }
define i8* @test8(i1 %x, i8* %c) {
entry:
  br i1 %x, label %return, label %if.then

if.then:                                          ; preds = %entry
  %p = call i8* @objc_retain(i8* %c) nounwind
  br label %return

return:                                           ; preds = %if.then, %entry
  %retval = phi i8* [ %c, %if.then ], [ null, %entry ]
  ret i8* %retval
}
