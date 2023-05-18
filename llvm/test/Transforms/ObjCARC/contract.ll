; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare ptr @llvm.objc.retain(ptr)
declare void @llvm.objc.release(ptr)
declare ptr @llvm.objc.autorelease(ptr)
declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)

declare void @use_pointer(ptr)
declare ptr @returner()
declare void @callee()

; CHECK-LABEL: define void @test0(
; CHECK: call void @use_pointer(ptr %0)
; CHECK: }
define void @test0(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  ret void
}

; CHECK-LABEL: define void @test1(
; CHECK: call void @use_pointer(ptr %0)
; CHECK: }
define void @test1(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.autorelease(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  ret void
}

; Merge objc_retain and objc_autorelease into objc_retainAutorelease.

; CHECK-LABEL: define void @test2(
; CHECK: tail call ptr @llvm.objc.retainAutorelease(ptr %x) [[NUW:#[0-9]+]]
; CHECK: }
define void @test2(ptr %x) nounwind {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.autorelease(ptr %0) nounwind
  call void @use_pointer(ptr %x)
  ret void
}

; Same as test2 but the value is returned. Do an RV optimization.

; CHECK-LABEL: define ptr @test2b(
; CHECK: tail call ptr @llvm.objc.retainAutoreleaseReturnValue(ptr %x) [[NUW]]
; CHECK: }
define ptr @test2b(ptr %x) nounwind {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %x) nounwind
  tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %0) nounwind
  ret ptr %x
}

; Merge a retain,autorelease pair around a call.

; CHECK-LABEL: define void @test3(
; CHECK: tail call ptr @llvm.objc.retainAutorelease(ptr %x) [[NUW]]
; CHECK: @use_pointer(ptr %0)
; CHECK: }
define void @test3(ptr %x, i64 %n) {
entry:
  tail call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call ptr @llvm.objc.autorelease(ptr %x) nounwind
  ret void
}

; Trivial retain,autorelease pair with intervening call, but it's post-dominated
; by another release. The retain and autorelease can be merged.

; CHECK-LABEL: define void @test4(
; CHECK-NEXT: entry:
; CHECK-NEXT: @llvm.objc.retainAutorelease(ptr %x) [[NUW]]
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @llvm.objc.release
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test4(ptr %x, i64 %n) {
entry:
  tail call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call ptr @llvm.objc.autorelease(ptr %x) nounwind
  tail call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Don't merge retain and autorelease if they're not control-equivalent.

; CHECK-LABEL: define void @test5(
; CHECK: tail call ptr @llvm.objc.retain(ptr %p) [[NUW]]
; CHECK: true:
; CHECK: call ptr @llvm.objc.autorelease(ptr %0) [[NUW]]
; CHECK: }
define void @test5(ptr %p, i1 %a) {
entry:
  tail call ptr @llvm.objc.retain(ptr %p) nounwind
  br i1 %a, label %true, label %false

true:
  call ptr @llvm.objc.autorelease(ptr %p) nounwind
  call void @use_pointer(ptr %p)
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

; CHECK-LABEL: define ptr @test6(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p) [[NUW]]
; CHECK: %t = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %1) [[NUW]]
; CHECK: }
define ptr @test6() {
  %p = call ptr @returner()
  tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p) nounwind
  %t = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %p) nounwind
  call void @use_pointer(ptr %t)
  ret ptr %t
}

; Don't spoil the RV optimization.

; CHECK: define ptr @test7(ptr %p)
; CHECK: tail call ptr @llvm.objc.retain(ptr %p)
; CHECK: call void @use_pointer(ptr %1)
; CHECK: tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %1)
; CHECK: ret ptr %2
; CHECK-NEXT: }
define ptr @test7(ptr %p) {
  %1 = tail call ptr @llvm.objc.retain(ptr %p)
  call void @use_pointer(ptr %p)
  %2 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
  ret ptr %p
}

; Do the return value substitution for PHI nodes too.

; CHECK-LABEL: define ptr @test8(
; CHECK: %retval = phi ptr [ %p, %if.then ], [ null, %entry ]
; CHECK: }
define ptr @test8(i1 %x, ptr %c) {
entry:
  br i1 %x, label %return, label %if.then

if.then:                                          ; preds = %entry
  %p = call ptr @llvm.objc.retain(ptr %c) nounwind
  br label %return

return:                                           ; preds = %if.then, %entry
  %retval = phi ptr [ %c, %if.then ], [ null, %entry ]
  ret ptr %retval
}

; Kill calls to @llvm.objc.clang.arc.use(...)
; CHECK-LABEL: define void @test9(
; CHECK-NOT: clang.arc.use
; CHECK: }
define void @test9(ptr %a, ptr %b) {
  call void (...) @llvm.objc.clang.arc.use(ptr %a, ptr %b) nounwind
  ret void
}


; Turn objc_retain into objc_retainAutoreleasedReturnValue if its operand
; is a return value.

; CHECK: define void @test10()
; CHECK: tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p)
define void @test10() {
  %p = call ptr @returner()
  tail call ptr @llvm.objc.retain(ptr %p) nounwind
  ret void
}

; Convert objc_retain to objc_retainAutoreleasedReturnValue if its
; argument is a return value.

; CHECK-LABEL: define void @test11(
; CHECK-NEXT: %y = call ptr @returner()
; CHECK-NEXT: tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %y) [[NUW]]
; CHECK-NEXT: ret void
define void @test11() {
  %y = call ptr @returner()
  tail call ptr @llvm.objc.retain(ptr %y) nounwind
  ret void
}

; Don't convert objc_retain to objc_retainAutoreleasedReturnValue if its
; argument is not a return value.

; CHECK-LABEL: define void @test12(
; CHECK-NEXT: tail call ptr @llvm.objc.retain(ptr %y) [[NUW]]
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test12(ptr %y) {
  tail call ptr @llvm.objc.retain(ptr %y) nounwind
  ret void
}

; Don't Convert objc_retain to objc_retainAutoreleasedReturnValue if it
; isn't next to the call providing its return value.

; CHECK-LABEL: define void @test13(
; CHECK-NEXT: %y = call ptr @returner()
; CHECK-NEXT: call void @callee()
; CHECK-NEXT: tail call ptr @llvm.objc.retain(ptr %y) [[NUW]]
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test13() {
  %y = call ptr @returner()
  call void @callee()
  tail call ptr @llvm.objc.retain(ptr %y) nounwind
  ret void
}

; CHECK-LABEL: define void @test14(
; CHECK-NOT: clang.arc.noop.use
; CHECK: ret void
define void @test14(ptr %a, ptr %b) {
  call void (...) @llvm.objc.clang.arc.noop.use(ptr %a, ptr %b) nounwind
  ret void
}

declare void @llvm.objc.clang.arc.use(...) nounwind
declare void @llvm.objc.clang.arc.noop.use(...) nounwind

; CHECK: attributes [[NUW]] = { nounwind }
