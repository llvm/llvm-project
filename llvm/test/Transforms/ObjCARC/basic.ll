; RUN: opt -aa-pipeline=basic-aa -passes=objc-arc -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare ptr @llvm.objc.retain(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr)
declare void @llvm.objc.release(ptr)
declare ptr @llvm.objc.autorelease(ptr)
declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare void @llvm.objc.autoreleasePoolPop(ptr)
declare ptr @llvm.objc.autoreleasePoolPush()
declare ptr @llvm.objc.retainBlock(ptr)

declare ptr @llvm.objc.retainedObject(ptr)
declare ptr @llvm.objc.unretainedObject(ptr)
declare ptr @llvm.objc.unretainedPointer(ptr)

declare void @use_pointer(ptr)
declare void @callee()
declare void @callee2(ptr, ptr)
declare void @callee_fnptr(ptr)
declare void @invokee()
declare ptr @returner()
declare void @bar(ptr)

declare void @llvm.dbg.value(metadata, metadata, metadata)

declare ptr @objc_msgSend(ptr, ptr, ...)

; Simple retain+release pair deletion, with some intervening control
; flow and harmless instructions.

; CHECK: define void @test0_precise(ptr %x, i1 %p) [[NUW:#[0-9]+]] {
; CHECK: @llvm.objc.retain
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test0_precise(ptr %x, i1 %p) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %return

f:
  store i32 7, ptr %x
  br label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK: define void @test0_imprecise(ptr %x, i1 %p) [[NUW]] {
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test0_imprecise(ptr %x, i1 %p) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %return

f:
  store i32 7, ptr %x
  br label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}

; Like test0 but the release isn't always executed when the retain is,
; so the optimization is not safe.

; TODO: Make the llvm.objc.release's argument be %0.

; CHECK: define void @test1_precise(ptr %x, i1 %p, i1 %q) [[NUW]] {
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: {{^}}}
define void @test1_precise(ptr %x, i1 %p, i1 %q) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %return

f:
  store i32 7, ptr %x
  call void @callee()
  br i1 %q, label %return, label %alt_return

return:
  call void @llvm.objc.release(ptr %x) nounwind
  ret void

alt_return:
  ret void
}

; CHECK: define void @test1_imprecise(ptr %x, i1 %p, i1 %q) [[NUW]] {
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test1_imprecise(ptr %x, i1 %p, i1 %q) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %return

f:
  store i32 7, ptr %x
  call void @callee()
  br i1 %q, label %return, label %alt_return

return:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void

alt_return:
  ret void
}


; Don't do partial elimination into two different CFG diamonds.

; CHECK: define void @test1b_precise(ptr %x, i1 %p, i1 %q) {
; CHECK: entry:
; CHECK:   tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK-NOT: @llvm.objc.
; CHECK: if.end5:
; CHECK:   tail call void @llvm.objc.release(ptr %x) [[NUW]]
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test1b_precise(ptr %x, i1 %p, i1 %q) {
entry:
  tail call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @callee()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  br i1 %q, label %if.then3, label %if.end5

if.then3:                                         ; preds = %if.end
  tail call void @use_pointer(ptr %x)
  br label %if.end5

if.end5:                                          ; preds = %if.then3, %if.end
  tail call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK-LABEL: define void @test1b_imprecise(
; CHECK: entry:
; CHECK:   tail call ptr @llvm.objc.retain(ptr %x) [[NUW:#[0-9]+]]
; CHECK-NOT: @llvm.objc.
; CHECK: if.end5:
; CHECK:   tail call void @llvm.objc.release(ptr %x) [[NUW]], !clang.imprecise_release ![[RELEASE:[0-9]+]]
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test1b_imprecise(ptr %x, i1 %p, i1 %q) {
entry:
  tail call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @callee()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  br i1 %q, label %if.then3, label %if.end5

if.then3:                                         ; preds = %if.end
  tail call void @use_pointer(ptr %x)
  br label %if.end5

if.end5:                                          ; preds = %if.then3, %if.end
  tail call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}


; Like test0 but the pointer is passed to an intervening call,
; so the optimization is not safe.

; CHECK-LABEL: define void @test2_precise(
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test2_precise(ptr %x, i1 %p) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %return

f:
  store i32 7, ptr %x
  call void @use_pointer(ptr %0)
  store float 3.0, ptr %x
  br label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK-LABEL: define void @test2_imprecise(
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test2_imprecise(ptr %x, i1 %p) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %return

f:
  store i32 7, ptr %x
  call void @use_pointer(ptr %0)
  store float 3.0, ptr %x
  br label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}

; Like test0 but the release is in a loop,
; so the optimization is not safe.

; TODO: For now, assume this can't happen.

; CHECK-LABEL: define void @test3_precise(
; TODO: @llvm.objc.retain(ptr %a)
; TODO: @llvm.objc.release
; CHECK: {{^}}}
define void @test3_precise(ptr %x, ptr %q) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %loop

loop:
  call void @llvm.objc.release(ptr %x) nounwind
  %j = load volatile i1, ptr %q
  br i1 %j, label %loop, label %return

return:
  ret void
}

; CHECK-LABEL: define void @test3_imprecise(
; TODO: @llvm.objc.retain(ptr %a)
; TODO: @llvm.objc.release
; CHECK: {{^}}}
define void @test3_imprecise(ptr %x, ptr %q) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %loop

loop:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  %j = load volatile i1, ptr %q
  br i1 %j, label %loop, label %return

return:
  ret void
}


; TODO: For now, assume this can't happen.

; Like test0 but the retain is in a loop,
; so the optimization is not safe.

; CHECK-LABEL: define void @test4_precise(
; TODO: @llvm.objc.retain(ptr %a)
; TODO: @llvm.objc.release
; CHECK: {{^}}}
define void @test4_precise(ptr %x, ptr %q) nounwind {
entry:
  br label %loop

loop:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %j = load volatile i1, ptr %q
  br i1 %j, label %loop, label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK-LABEL: define void @test4_imprecise(
; TODO: @llvm.objc.retain(ptr %a)
; TODO: @llvm.objc.release
; CHECK: {{^}}}
define void @test4_imprecise(ptr %x, ptr %q) nounwind {
entry:
  br label %loop

loop:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %j = load volatile i1, ptr %q
  br i1 %j, label %loop, label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}


; Like test0 but the pointer is conditionally passed to an intervening call,
; so the optimization is not safe.

; CHECK-LABEL: define void @test5a(
; CHECK: @llvm.objc.retain(ptr
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test5a(ptr %x, i1 %q, ptr %y) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %s = select i1 %q, ptr %y, ptr %0
  call void @use_pointer(ptr %s)
  store i32 7, ptr %x
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK-LABEL: define void @test5b(
; CHECK: @llvm.objc.retain(ptr
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test5b(ptr %x, i1 %q, ptr %y) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %s = select i1 %q, ptr %y, ptr %0
  call void @use_pointer(ptr %s)
  store i32 7, ptr %x
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}


; retain+release pair deletion, where the release happens on two different
; flow paths.

; CHECK-LABEL: define void @test6a(
; CHECK: entry:
; CHECK:   tail call ptr @llvm.objc.retain
; CHECK: t:
; CHECK:   call void @llvm.objc.release
; CHECK: f:
; CHECK:   call void @llvm.objc.release
; CHECK: return:
; CHECK: {{^}}}
define void @test6a(ptr %x, i1 %p) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  call void @llvm.objc.release(ptr %x) nounwind
  br label %return

f:
  store i32 7, ptr %x
  call void @callee()
  call void @llvm.objc.release(ptr %x) nounwind
  br label %return

return:
  ret void
}

; CHECK-LABEL: define void @test6b(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test6b(ptr %x, i1 %p) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %return

f:
  store i32 7, ptr %x
  call void @callee()
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %return

return:
  ret void
}

; CHECK-LABEL: define void @test6c(
; CHECK: entry:
; CHECK:   tail call ptr @llvm.objc.retain
; CHECK: t:
; CHECK:   call void @llvm.objc.release
; CHECK: f:
; CHECK:   call void @llvm.objc.release
; CHECK: return:
; CHECK: {{^}}}
define void @test6c(ptr %x, i1 %p) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  call void @llvm.objc.release(ptr %x) nounwind
  br label %return

f:
  store i32 7, ptr %x
  call void @callee()
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %return

return:
  ret void
}

; CHECK-LABEL: define void @test6d(
; CHECK: entry:
; CHECK:   tail call ptr @llvm.objc.retain
; CHECK: t:
; CHECK:   call void @llvm.objc.release
; CHECK: f:
; CHECK:   call void @llvm.objc.release
; CHECK: return:
; CHECK: {{^}}}
define void @test6d(ptr %x, i1 %p) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, ptr %x
  store float 2.0, ptr %x
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %return

f:
  store i32 7, ptr %x
  call void @callee()
  call void @llvm.objc.release(ptr %x) nounwind
  br label %return

return:
  ret void
}


; retain+release pair deletion, where the retain happens on two different
; flow paths.

; CHECK-LABEL:     define void @test7(
; CHECK:     entry:
; CHECK-NOT:   llvm.objc.
; CHECK:     t:
; CHECK:       call ptr @llvm.objc.retain
; CHECK:     f:
; CHECK:       call ptr @llvm.objc.retain
; CHECK:     return:
; CHECK:       call void @llvm.objc.release
; CHECK: {{^}}}
define void @test7(ptr %x, i1 %p) nounwind {
entry:
  br i1 %p, label %t, label %f

t:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %return

f:
  %1 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i32 7, ptr %x
  call void @callee()
  br label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK-LABEL: define void @test7b(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test7b(ptr %x, i1 %p) nounwind {
entry:
  br i1 %p, label %t, label %f

t:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %return

f:
  %1 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i32 7, ptr %x
  call void @callee()
  br label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}

; Like test7, but there's a retain/retainBlock mismatch. Don't delete!

; CHECK-LABEL: define void @test7c(
; CHECK: t:
; CHECK:   call ptr @llvm.objc.retainBlock
; CHECK: f:
; CHECK:   call ptr @llvm.objc.retain
; CHECK: return:
; CHECK:   call void @llvm.objc.release
; CHECK: {{^}}}
define void @test7c(ptr %x, i1 %p) nounwind {
entry:
  br i1 %p, label %t, label %f

t:
  %0 = call ptr @llvm.objc.retainBlock(ptr %x) nounwind
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %return

f:
  %1 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i32 7, ptr %x
  call void @callee()
  br label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; retain+release pair deletion, where the retain and release both happen on
; different flow paths. Wild!

; CHECK-LABEL: define void @test8a(
; CHECK: entry:
; CHECK: t:
; CHECK:   @llvm.objc.retain
; CHECK: f:
; CHECK:   @llvm.objc.retain
; CHECK: mid:
; CHECK: u:
; CHECK:   @llvm.objc.release
; CHECK: g:
; CHECK:   @llvm.objc.release
; CHECK: return:
; CHECK: {{^}}}
define void @test8a(ptr %x, i1 %p, i1 %q) nounwind {
entry:
  br i1 %p, label %t, label %f

t:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %mid

f:
  %1 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i32 7, ptr %x
  br label %mid

mid:
  br i1 %q, label %u, label %g

u:
  call void @callee()
  call void @llvm.objc.release(ptr %x) nounwind
  br label %return

g:
  call void @llvm.objc.release(ptr %x) nounwind
  br label %return

return:
  ret void
}

; CHECK-LABEL: define void @test8b(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test8b(ptr %x, i1 %p, i1 %q) nounwind {
entry:
  br i1 %p, label %t, label %f

t:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %mid

f:
  %1 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i32 7, ptr %x
  br label %mid

mid:
  br i1 %q, label %u, label %g

u:
  call void @callee()
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %return

g:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %return

return:
  ret void
}

; CHECK-LABEL: define void @test8c(
; CHECK: entry:
; CHECK: t:
; CHECK-NOT: @llvm.objc.
; CHECK: f:
; CHECK-NOT: @llvm.objc.
; CHECK: mid:
; CHECK: u:
; CHECK:   @llvm.objc.retain
; CHECK:   @llvm.objc.release
; CHECK: g:
; CHECK-NOT: @llvm.objc.
; CHECK: return:
; CHECK: {{^}}}
define void @test8c(ptr %x, i1 %p, i1 %q) nounwind {
entry:
  br i1 %p, label %t, label %f

t:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %mid

f:
  %1 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i32 7, ptr %x
  br label %mid

mid:
  br i1 %q, label %u, label %g

u:
  call void @callee()
  call void @llvm.objc.release(ptr %x) nounwind
  br label %return

g:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %return

return:
  ret void
}

; CHECK-LABEL: define void @test8d(
; CHECK: entry:
; CHECK: t:
; CHECK:   @llvm.objc.retain
; CHECK: f:
; CHECK:   @llvm.objc.retain
; CHECK: mid:
; CHECK: u:
; CHECK:   @llvm.objc.release
; CHECK: g:
; CHECK:   @llvm.objc.release
; CHECK: return:
; CHECK: {{^}}}
define void @test8d(ptr %x, i1 %p, i1 %q) nounwind {
entry:
  br i1 %p, label %t, label %f

t:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i8 3, ptr %x
  store float 2.0, ptr %x
  br label %mid

f:
  %1 = call ptr @llvm.objc.retain(ptr %x) nounwind
  store i32 7, ptr %x
  br label %mid

mid:
  br i1 %q, label %u, label %g

u:
  call void @callee()
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %return

g:
  call void @llvm.objc.release(ptr %x) nounwind
  br label %return

return:
  ret void
}

; Trivial retain+release pair deletion.

; CHECK-LABEL: define void @test9(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test9(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @llvm.objc.release(ptr %0) nounwind
  ret void
}

; Retain+release pair, but on an unknown pointer relationship. Don't delete!

; CHECK-LABEL: define void @test9b(
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %s)
; CHECK: {{^}}}
define void @test9b(ptr %x, i1 %j, ptr %p) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %s = select i1 %j, ptr %x, ptr %p
  call void @llvm.objc.release(ptr %s) nounwind
  ret void
}

; Trivial retain+release pair with intervening calls - don't delete!

; CHECK-LABEL: define void @test10(
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @callee
; CHECK: @use_pointer
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test10(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @callee()
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %0) nounwind
  ret void
}

; Trivial retain+autoreleaserelease pair. Don't delete!
; Also, add a tail keyword, since llvm.objc.retain can never be passed
; a stack argument.

; CHECK-LABEL: define void @test11(
; CHECK: tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK: call ptr @llvm.objc.autorelease(ptr %0) [[NUW]]
; CHECK: {{^}}}
define void @test11(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.autorelease(ptr %0) nounwind
  call void @use_pointer(ptr %x)
  ret void
}

; Same as test11 but with no use_pointer call. Delete the pair!

; CHECK-LABEL: define void @test11a(
; CHECK: entry:
; CHECK-NEXT: ret void
; CHECK: {{^}}}
define void @test11a(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.autorelease(ptr %0) nounwind
  ret void
}

; Same as test11 but the value is returned. Do not perform an RV optimization
; since if the frontend emitted code for an __autoreleasing variable, we may
; want it to be in the autorelease pool.

; CHECK-LABEL: define ptr @test11b(
; CHECK: tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK: call ptr @llvm.objc.autorelease(ptr %0) [[NUW]]
; CHECK: {{^}}}
define ptr @test11b(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.autorelease(ptr %0) nounwind
  ret ptr %x
}

; We can not delete this retain, release since we do not have a post-dominating
; use of the release.

; CHECK-LABEL: define void @test12(
; CHECK-NEXT: entry:
; CHECK-NEXT: @llvm.objc.retain(ptr %x)
; CHECK-NEXT: @llvm.objc.retain
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test12(ptr %x, i64 %n) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Trivial retain,autorelease pair. Don't delete!

; CHECK-LABEL: define void @test13(
; CHECK: tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK: tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK: @use_pointer(ptr %x)
; CHECK: call ptr @llvm.objc.autorelease(ptr %x) [[NUW]]
; CHECK: {{^}}}
define void @test13(ptr %x, i64 %n) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call ptr @llvm.objc.autorelease(ptr %x) nounwind
  ret void
}

; Delete the retain+release pair.

; CHECK-LABEL: define void @test13b(
; CHECK-NEXT: entry:
; CHECK-NEXT: @llvm.objc.retain(ptr %x)
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @llvm.objc.release
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test13b(ptr %x, i64 %n) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Don't delete the retain+release pair because there's an
; autoreleasePoolPop in the way.

; CHECK-LABEL: define void @test13c(
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.autoreleasePoolPop
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @use_pointer
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test13c(ptr %x, i64 %n) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @llvm.objc.autoreleasePoolPop(ptr undef)
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Like test13c, but there's an autoreleasePoolPush in the way, but that
; doesn't matter.

; CHECK-LABEL: define void @test13d(
; CHECK-NEXT: entry:
; CHECK-NEXT: @llvm.objc.retain(ptr %x)
; CHECK-NEXT: @llvm.objc.autoreleasePoolPush
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @llvm.objc.release
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test13d(ptr %x, i64 %n) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.autoreleasePoolPush()
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Trivial retain,release pair with intervening call, and it's post-dominated by
; another release. But it is not known safe in the top down direction. We can
; not eliminate it.

; CHECK-LABEL: define void @test14(
; CHECK-NEXT: entry:
; CHECK-NEXT: @llvm.objc.retain
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @llvm.objc.release
; CHECK-NEXT: @llvm.objc.release
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test14(ptr %x, i64 %n) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Trivial retain,autorelease pair with intervening call, but it's post-dominated
; by another release. Don't delete anything.

; CHECK-LABEL: define void @test15(
; CHECK-NEXT: entry:
; CHECK-NEXT: @llvm.objc.retain(ptr %x)
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @llvm.objc.autorelease(ptr %x)
; CHECK-NEXT: @llvm.objc.release
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test15(ptr %x, i64 %n) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call ptr @llvm.objc.autorelease(ptr %x) nounwind
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Trivial retain,autorelease pair, post-dominated
; by another release. Delete the retain and release.

; CHECK-LABEL: define void @test15b(
; CHECK-NEXT: entry:
; CHECK-NEXT: @llvm.objc.retain
; CHECK-NEXT: @llvm.objc.autorelease
; CHECK-NEXT: @llvm.objc.release
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test15b(ptr %x, i64 %n) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.autorelease(ptr %x) nounwind
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK-LABEL: define void @test15c(
; CHECK-NEXT: entry:
; CHECK-NEXT: @llvm.objc.autorelease
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test15c(ptr %x, i64 %n) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.autorelease(ptr %x) nounwind
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}

; Retain+release pairs in diamonds, all dominated by a retain.

; CHECK-LABEL: define void @test16a(
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK-NOT: @objc
; CHECK: purple:
; CHECK: @use_pointer
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test16a(i1 %a, i1 %b, ptr %x) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %a, label %red, label %orange

red:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %yellow

orange:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %yellow

yellow:
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  br i1 %b, label %green, label %blue

green:
  call void @llvm.objc.release(ptr %x) nounwind
  br label %purple

blue:
  call void @llvm.objc.release(ptr %x) nounwind
  br label %purple

purple:
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK-LABEL: define void @test16b(
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK-NOT: @objc
; CHECK: purple:
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @use_pointer
; CHECK-NEXT: @llvm.objc.release
; CHECK: {{^}}}
define void @test16b(i1 %a, i1 %b, ptr %x) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %a, label %red, label %orange

red:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %yellow

orange:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %yellow

yellow:
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  br i1 %b, label %green, label %blue

green:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %purple

blue:
  call void @llvm.objc.release(ptr %x) nounwind
  br label %purple

purple:
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK-LABEL: define void @test16c(
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK-NOT: @objc
; CHECK: purple:
; CHECK: @use_pointer
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test16c(i1 %a, i1 %b, ptr %x) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %a, label %red, label %orange

red:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %yellow

orange:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %yellow

yellow:
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  br i1 %b, label %green, label %blue

green:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %purple

blue:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %purple

purple:
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}

; CHECK-LABEL: define void @test16d(
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc
; CHECK: {{^}}}
define void @test16d(i1 %a, i1 %b, ptr %x) {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br i1 %a, label %red, label %orange

red:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %yellow

orange:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  br label %yellow

yellow:
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  br i1 %b, label %green, label %blue

green:
  call void @llvm.objc.release(ptr %x) nounwind
  br label %purple

blue:
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  br label %purple

purple:
  ret void
}

; Delete no-ops.

; CHECK-LABEL: define void @test18(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test18() {
  call ptr @llvm.objc.retain(ptr null)
  call void @llvm.objc.release(ptr null)
  call ptr @llvm.objc.autorelease(ptr null)
  ret void
}

; Delete no-ops where undef can be assumed to be null.

; CHECK-LABEL: define void @test18b(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test18b() {
  call ptr @llvm.objc.retain(ptr undef)
  call void @llvm.objc.release(ptr undef)
  call ptr @llvm.objc.autorelease(ptr undef)
  ret void
}

; Replace uses of arguments with uses of return values, to reduce
; register pressure.

; CHECK: define void @test19(ptr %y) {
; CHECK:   %0 = tail call ptr @llvm.objc.retain(ptr %y)
; CHECK:   call void @use_pointer(ptr %y)
; CHECK:   call void @use_pointer(ptr %y)
; CHECK:   call void @llvm.objc.release(ptr %y)
; CHECK:   ret void
; CHECK: {{^}}}
define void @test19(ptr %y) {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %y) nounwind
  call void @use_pointer(ptr %y)
  call void @use_pointer(ptr %y)
  call void @llvm.objc.release(ptr %y)
  ret void
}

; Bitcast insertion

; CHECK-LABEL: define void @test20(
; CHECK: %tmp1 = tail call ptr @llvm.objc.retain(ptr %self) [[NUW]]
; CHECK-NEXT: invoke
; CHECK: {{^}}}
define void @test20(ptr %self) personality ptr @__gxx_personality_v0 {
if.then12:
  %tmp1 = call ptr @llvm.objc.retain(ptr %self) nounwind
  invoke void @invokee()
          to label %invoke.cont23 unwind label %lpad20

invoke.cont23:                                    ; preds = %if.then12
  invoke void @invokee()
          to label %if.end unwind label %lpad20

lpad20:                                           ; preds = %invoke.cont23, %if.then12
  %tmp502 = phi ptr [ undef, %invoke.cont23 ], [ %self, %if.then12 ]
  %exn = landingpad {ptr, i32}
           cleanup
  unreachable

if.end:                                           ; preds = %invoke.cont23
  ret void
}

; Delete a redundant retain,autorelease when forwaring a call result
; directly to a return value.

; CHECK-LABEL: define ptr @test21(
; CHECK: call ptr @returner()
; CHECK-NEXT: ret ptr %call
; CHECK-NEXT: }
define ptr @test21() {
entry:
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retain(ptr %call) nounwind
  %1 = call ptr @llvm.objc.autorelease(ptr %0) nounwind
  ret ptr %1
}

; Move an objc call up through a phi that has null operands.

; CHECK-LABEL: define void @test22(
; CHECK: B:
; CHECK:   call void @llvm.objc.release(ptr %p)
; CHECK:   br label %C
; CHECK: C:                                                ; preds = %B, %A
; CHECK-NOT: @llvm.objc.release
; CHECK: {{^}}}
define void @test22(ptr %p, i1 %a) {
  br i1 %a, label %A, label %B
A:
  br label %C
B:
  br label %C
C:
  %h = phi ptr [ null, %A ], [ %p, %B ]
  call void @llvm.objc.release(ptr %h), !clang.imprecise_release !0
  ret void
}

; Do not move an llvm.objc.release that doesn't have the clang.imprecise_release tag.

; CHECK-LABEL: define void @test22_precise(
; CHECK: %[[P0:.*]] = phi ptr
; CHECK: call void @llvm.objc.release(ptr %[[P0]])
; CHECK: ret void
define void @test22_precise(ptr %p, i1 %a) {
  br i1 %a, label %A, label %B
A:
  br label %C
B:
  br label %C
C:
  %h = phi ptr [ null, %A ], [ %p, %B ]
  call void @llvm.objc.release(ptr %h)
  ret void
}

; Any call can decrement a retain count.

; CHECK-LABEL: define void @test24(
; CHECK: @llvm.objc.retain(ptr %a)
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test24(ptr %r, ptr %a) {
  call ptr @llvm.objc.retain(ptr %a)
  call void @use_pointer(ptr %r)
  %q = load i8, ptr %a
  call void @llvm.objc.release(ptr %a)
  ret void
}

; Don't move a retain/release pair if the release can be moved
; but the retain can't be moved to balance it.

; CHECK-LABEL: define void @test25(
; CHECK: entry:
; CHECK:   call ptr @llvm.objc.retain(ptr %p)
; CHECK: true:
; CHECK: done:
; CHECK:   call void @llvm.objc.release(ptr %p)
; CHECK: {{^}}}
define void @test25(ptr %p, i1 %x) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  br i1 %x, label %true, label %done

true:
  store i8 0, ptr %p
  br label %done

done:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Don't move a retain/release pair if the retain can be moved
; but the release can't be moved to balance it.

; CHECK-LABEL: define void @test26(
; CHECK: entry:
; CHECK:   call ptr @llvm.objc.retain(ptr %p)
; CHECK: true:
; CHECK: done:
; CHECK:   call void @llvm.objc.release(ptr %p)
; CHECK: {{^}}}
define void @test26(ptr %p, i1 %x) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  call void @callee()
  br label %done

done:
  store i8 0, ptr %p
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Don't sink the retain,release into the loop.

; CHECK-LABEL: define void @test27(
; CHECK: entry:
; CHECK: call ptr @llvm.objc.retain(ptr %p)
; CHECK: loop:
; CHECK-NOT: @llvm.objc.
; CHECK: done:
; CHECK: call void @llvm.objc.release
; CHECK: {{^}}}
define void @test27(ptr %p, i1 %x, i1 %y) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %loop, label %done

loop:
  call void @callee()
  store i8 0, ptr %p
  br i1 %y, label %done, label %loop

done:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Trivial code motion case: Triangle.

; CHECK-LABEL: define void @test28(
; CHECK-NOT: @llvm.objc.
; CHECK: true:
; CHECK: call ptr @llvm.objc.retain
; CHECK: call void @callee()
; CHECK: store
; CHECK: call void @llvm.objc.release
; CHECK: done:
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test28(ptr %p, i1 %x) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  call void @callee()
  store i8 0, ptr %p
  br label %done

done:
  call void @llvm.objc.release(ptr %p), !clang.imprecise_release !0
  ret void
}

; Trivial code motion case: Triangle, but no metadata. Don't move past
; unrelated memory references!

; CHECK-LABEL: define void @test28b(
; CHECK: call ptr @llvm.objc.retain
; CHECK: true:
; CHECK-NOT: @llvm.objc.
; CHECK: call void @callee()
; CHECK-NOT: @llvm.objc.
; CHECK: store
; CHECK-NOT: @llvm.objc.
; CHECK: done:
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test28b(ptr %p, i1 %x, ptr noalias %t) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  call void @callee()
  store i8 0, ptr %p
  br label %done

done:
  store i8 0, ptr %t
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Trivial code motion case: Triangle, with metadata. Do move past
; unrelated memory references! And preserve the metadata.

; CHECK-LABEL: define void @test28c(
; CHECK-NOT: @llvm.objc.
; CHECK: true:
; CHECK: call ptr @llvm.objc.retain
; CHECK: call void @callee()
; CHECK: store
; CHECK: call void @llvm.objc.release(ptr %p) [[NUW]], !clang.imprecise_release
; CHECK: done:
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test28c(ptr %p, i1 %x, ptr noalias %t) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  call void @callee()
  store i8 0, ptr %p
  br label %done

done:
  store i8 0, ptr %t
  call void @llvm.objc.release(ptr %p), !clang.imprecise_release !0
  ret void
}

; Like test28. but with two releases.

; CHECK-LABEL: define void @test29(
; CHECK: call ptr @llvm.objc.retain
; CHECK: true:
; CHECK: call void @callee()
; CHECK: store
; CHECK: done:
; CHECK: call void @llvm.objc.release
; CHECK: ohno:
; CHECK: call void @llvm.objc.release
; CHECK: {{^}}}
define void @test29(ptr %p, i1 %x, i1 %y) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  call void @callee()
  store i8 0, ptr %p
  br i1 %y, label %done, label %ohno

done:
  call void @llvm.objc.release(ptr %p)
  ret void

ohno:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Basic case with the use and call in a diamond
; with an extra release.

; CHECK-LABEL: define void @test30(
; CHECK: call ptr @llvm.objc.retain
; CHECK: true:
; CHECK: call void @callee()
; CHECK: store
; CHECK: false:
; CHECK: done:
; CHECK: call void @llvm.objc.release
; CHECK: ohno:
; CHECK: call void @llvm.objc.release
; CHECK: {{^}}}
define void @test30(ptr %p, i1 %x, i1 %y, i1 %z) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %false

true:
  call void @callee()
  store i8 0, ptr %p
  br i1 %y, label %done, label %ohno

false:
  br i1 %z, label %done, label %ohno

done:
  call void @llvm.objc.release(ptr %p)
  ret void

ohno:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Basic case with a mergeable release.

; CHECK-LABEL: define void @test31(
; CHECK: call ptr @llvm.objc.retain(ptr %p)
; CHECK: call void @callee()
; CHECK: store
; CHECK: true:
; CHECK: call void @llvm.objc.release
; CHECK: false:
; CHECK: call void @llvm.objc.release
; CHECK: ret void
; CHECK: {{^}}}
define void @test31(ptr %p, i1 %x) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  store i8 0, ptr %p
  br i1 %x, label %true, label %false
true:
  call void @llvm.objc.release(ptr %p)
  ret void
false:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Don't consider bitcasts or getelementptrs direct uses.

; CHECK-LABEL: define void @test32(
; CHECK: call ptr @llvm.objc.retain
; CHECK: true:
; CHECK: call void @callee()
; CHECK: store
; CHECK: done:
; CHECK: call void @llvm.objc.release
; CHECK: {{^}}}
define void @test32(ptr %p, i1 %x) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  call void @callee()
  store i8 0, ptr %p
  br label %done

done:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Do consider icmps to be direct uses.

; CHECK-LABEL: define void @test33(
; CHECK: call ptr @llvm.objc.retain
; CHECK: true:
; CHECK: call void @callee()
; CHECK: icmp
; CHECK: done:
; CHECK: call void @llvm.objc.release
; CHECK: {{^}}}
define void @test33(ptr %p, i1 %x, ptr %y) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  call void @callee()
  %v = icmp eq ptr %p, %y
  br label %done

done:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Delete retain,release if there's just a possible dec and we have imprecise
; releases.

; CHECK-LABEL: define void @test34a(
; CHECK:   call ptr @llvm.objc.retain
; CHECK: true:
; CHECK: done:
; CHECK: call void @llvm.objc.release
; CHECK: {{^}}}
define void @test34a(ptr %p, i1 %x, ptr %y) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  call void @callee()
  br label %done

done:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; CHECK-LABEL: define void @test34b(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test34b(ptr %p, i1 %x, ptr %y) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  call void @callee()
  br label %done

done:
  call void @llvm.objc.release(ptr %p), !clang.imprecise_release !0
  ret void
}


; Delete retain,release if there's just a use and we do not have a precise
; release.

; Precise.
; CHECK-LABEL: define void @test35a(
; CHECK: entry:
; CHECK:   call ptr @llvm.objc.retain
; CHECK: true:
; CHECK: done:
; CHECK:   call void @llvm.objc.release
; CHECK: {{^}}}
define void @test35a(ptr %p, i1 %x, ptr %y) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  %v = icmp eq ptr %p, %y
  br label %done

done:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Imprecise.
; CHECK-LABEL: define void @test35b(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test35b(ptr %p, i1 %x, ptr %y) {
entry:
  %f0 = call ptr @llvm.objc.retain(ptr %p)
  br i1 %x, label %true, label %done

true:
  %v = icmp eq ptr %p, %y
  br label %done

done:
  call void @llvm.objc.release(ptr %p), !clang.imprecise_release !0
  ret void
}

; Delete a retain,release if there's no actual use and we have precise release.

; CHECK-LABEL: define void @test36a(
; CHECK: @llvm.objc.retain
; CHECK: call void @callee()
; CHECK-NOT: @llvm.objc.
; CHECK: call void @callee()
; CHECK: @llvm.objc.release
; CHECK: {{^}}}
define void @test36a(ptr %p) {
entry:
  call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  call void @callee()
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Like test36, but with metadata.

; CHECK-LABEL: define void @test36b(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test36b(ptr %p) {
entry:
  call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  call void @callee()
  call void @llvm.objc.release(ptr %p), !clang.imprecise_release !0
  ret void
}

; Be aggressive about analyzing phis to eliminate possible uses.

; CHECK-LABEL: define void @test38(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test38(ptr %p, i1 %u, i1 %m, ptr %z, ptr %y, ptr %x, ptr %w) {
entry:
  call ptr @llvm.objc.retain(ptr %p)
  br i1 %u, label %true, label %false
true:
  br i1 %m, label %a, label %b
false:
  br i1 %m, label %c, label %d
a:
  br label %e
b:
  br label %e
c:
  br label %f
d:
  br label %f
e:
  %j = phi ptr [ %z, %a ], [ %y, %b ]
  br label %g
f:
  %k = phi ptr [ %w, %c ], [ %x, %d ]
  br label %g
g:
  %h = phi ptr [ %j, %e ], [ %k, %f ]
  call void @use_pointer(ptr %h)
  call void @llvm.objc.release(ptr %p), !clang.imprecise_release !0
  ret void
}

; Delete retain,release pairs around loops.

; CHECK-LABEL: define void @test39(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test39(ptr %p) {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %p)
  br label %loop

loop:                                             ; preds = %loop, %entry
  br i1 undef, label %loop, label %exit

exit:                                             ; preds = %loop
  call void @llvm.objc.release(ptr %0), !clang.imprecise_release !0
  ret void
}

; Delete retain,release pairs around loops containing uses.

; CHECK-LABEL: define void @test39b(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test39b(ptr %p) {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %p)
  br label %loop

loop:                                             ; preds = %loop, %entry
  store i8 0, ptr %0
  br i1 undef, label %loop, label %exit

exit:                                             ; preds = %loop
  call void @llvm.objc.release(ptr %0), !clang.imprecise_release !0
  ret void
}

; Delete retain,release pairs around loops containing potential decrements.

; CHECK-LABEL: define void @test39c(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test39c(ptr %p) {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %p)
  br label %loop

loop:                                             ; preds = %loop, %entry
  call void @use_pointer(ptr %0)
  br i1 undef, label %loop, label %exit

exit:                                             ; preds = %loop
  call void @llvm.objc.release(ptr %0), !clang.imprecise_release !0
  ret void
}

; Delete retain,release pairs around loops even if
; the successors are in a different order.

; CHECK-LABEL: define void @test40(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test40(ptr %p) {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %p)
  br label %loop

loop:                                             ; preds = %loop, %entry
  call void @use_pointer(ptr %0)
  br i1 undef, label %exit, label %loop

exit:                                             ; preds = %loop
  call void @llvm.objc.release(ptr %0), !clang.imprecise_release !0
  ret void
}

; Do the known-incremented retain+release elimination even if the pointer
; is also autoreleased.

; CHECK-LABEL: define void @test42(
; CHECK-NEXT: entry:
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %p)
; CHECK-NEXT: call ptr @llvm.objc.autorelease(ptr %p)
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK-NEXT: call void @llvm.objc.release(ptr %p)
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test42(ptr %p) {
entry:
  call ptr @llvm.objc.retain(ptr %p)
  call ptr @llvm.objc.autorelease(ptr %p)
  call ptr @llvm.objc.retain(ptr %p)
  call void @use_pointer(ptr %p)
  call void @use_pointer(ptr %p)
  call void @llvm.objc.release(ptr %p)
  call void @use_pointer(ptr %p)
  call void @use_pointer(ptr %p)
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Don't the known-incremented retain+release elimination if the pointer is
; autoreleased and there's an autoreleasePoolPop.

; CHECK-LABEL: define void @test43(
; CHECK-NEXT: entry:
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %p)
; CHECK-NEXT: call ptr @llvm.objc.autorelease(ptr %p)
; CHECK-NEXT: call ptr @llvm.objc.retain
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK-NEXT: call void @llvm.objc.autoreleasePoolPop(ptr undef)
; CHECK-NEXT: call void @llvm.objc.release
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test43(ptr %p) {
entry:
  call ptr @llvm.objc.retain(ptr %p)
  call ptr @llvm.objc.autorelease(ptr %p)
  call ptr @llvm.objc.retain(ptr %p)
  call void @use_pointer(ptr %p)
  call void @use_pointer(ptr %p)
  call void @llvm.objc.autoreleasePoolPop(ptr undef)
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Do the known-incremented retain+release elimination if the pointer is
; autoreleased and there's an autoreleasePoolPush.

; CHECK-LABEL: define void @test43b(
; CHECK-NEXT: entry:
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %p)
; CHECK-NEXT: call ptr @llvm.objc.autorelease(ptr %p)
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK-NEXT: call ptr @llvm.objc.autoreleasePoolPush()
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK-NEXT: call void @llvm.objc.release
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test43b(ptr %p) {
entry:
  call ptr @llvm.objc.retain(ptr %p)
  call ptr @llvm.objc.autorelease(ptr %p)
  call ptr @llvm.objc.retain(ptr %p)
  call void @use_pointer(ptr %p)
  call void @use_pointer(ptr %p)
  call ptr @llvm.objc.autoreleasePoolPush()
  call void @llvm.objc.release(ptr %p)
  call void @use_pointer(ptr %p)
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Do retain+release elimination for non-provenance pointers.

; CHECK-LABEL: define void @test44(
; CHECK-NOT: llvm.objc.
; CHECK: {{^}}}
define void @test44(ptr %pp) {
  %p = load ptr, ptr %pp
  %q = call ptr @llvm.objc.retain(ptr %p)
  call void @llvm.objc.release(ptr %q)
  ret void
}

; Don't delete retain+release with an unknown-provenance
; may-alias llvm.objc.release between them.

; CHECK-LABEL: define void @test45(
; CHECK: call ptr @llvm.objc.retain(ptr %p)
; CHECK: call void @llvm.objc.release(ptr %q)
; CHECK: call void @use_pointer(ptr %p)
; CHECK: call void @llvm.objc.release(ptr %p)
; CHECK: {{^}}}
define void @test45(ptr %pp, ptr %qq) {
  %p = load ptr, ptr %pp
  %q = load ptr, ptr %qq
  call ptr @llvm.objc.retain(ptr %p)
  call void @llvm.objc.release(ptr %q)
  call void @use_pointer(ptr %p)
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Don't delete retain and autorelease here.

; CHECK-LABEL: define void @test46(
; CHECK: tail call ptr @llvm.objc.retain(ptr %p) [[NUW]]
; CHECK: true:
; CHECK: call ptr @llvm.objc.autorelease(ptr %p) [[NUW]]
; CHECK: {{^}}}
define void @test46(ptr %p, i1 %a) {
entry:
  call ptr @llvm.objc.retain(ptr %p)
  br i1 %a, label %true, label %false

true:
  call ptr @llvm.objc.autorelease(ptr %p)
  call void @use_pointer(ptr %p)
  ret void

false:
  ret void
}

; Delete no-op cast calls.

; CHECK-LABEL: define ptr @test47(
; CHECK-NOT: call
; CHECK: ret ptr %p
; CHECK: {{^}}}
define ptr @test47(ptr %p) nounwind {
  %x = call ptr @llvm.objc.retainedObject(ptr %p)
  ret ptr %x
}

; Delete no-op cast calls.

; CHECK-LABEL: define ptr @test48(
; CHECK-NOT: call
; CHECK: ret ptr %p
; CHECK: {{^}}}
define ptr @test48(ptr %p) nounwind {
  %x = call ptr @llvm.objc.unretainedObject(ptr %p)
  ret ptr %x
}

; Delete no-op cast calls.

; CHECK-LABEL: define ptr @test49(
; CHECK-NOT: call
; CHECK: ret ptr %p
; CHECK: {{^}}}
define ptr @test49(ptr %p) nounwind {
  %x = call ptr @llvm.objc.unretainedPointer(ptr %p)
  ret ptr %x
}

; Do delete retain+release with intervening stores of the address value if we
; have imprecise release attached to llvm.objc.release.

; CHECK-LABEL:      define void @test50a(
; CHECK-NEXT:   call ptr @llvm.objc.retain
; CHECK-NEXT:   call void @callee
; CHECK-NEXT:   store
; CHECK-NEXT:   call void @llvm.objc.release
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test50a(ptr %p, ptr %pp) {
  call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  store ptr %p, ptr %pp
  call void @llvm.objc.release(ptr %p)
  ret void
}

; CHECK-LABEL: define void @test50b(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test50b(ptr %p, ptr %pp) {
  call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  store ptr %p, ptr %pp
  call void @llvm.objc.release(ptr %p), !clang.imprecise_release !0
  ret void
}


; Don't delete retain+release with intervening stores through the
; address value.

; CHECK-LABEL: define void @test51a(
; CHECK: call ptr @llvm.objc.retain(ptr %p)
; CHECK: call void @llvm.objc.release(ptr %p)
; CHECK: ret void
; CHECK: {{^}}}
define void @test51a(ptr %p) {
  call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  store i8 0, ptr %p
  call void @llvm.objc.release(ptr %p)
  ret void
}

; CHECK-LABEL: define void @test51b(
; CHECK: call ptr @llvm.objc.retain(ptr %p)
; CHECK: call void @llvm.objc.release(ptr %p)
; CHECK: ret void
; CHECK: {{^}}}
define void @test51b(ptr %p) {
  call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  store i8 0, ptr %p
  call void @llvm.objc.release(ptr %p), !clang.imprecise_release !0
  ret void
}

; Don't delete retain+release with intervening use of a pointer of
; unknown provenance.

; CHECK-LABEL: define void @test52a(
; CHECK: call ptr @llvm.objc.retain
; CHECK: call void @callee()
; CHECK: call void @use_pointer(ptr %z)
; CHECK: call void @llvm.objc.release
; CHECK: ret void
; CHECK: {{^}}}
define void @test52a(ptr %zz, ptr %pp) {
  %p = load ptr, ptr %pp
  %1 = call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  %z = load ptr, ptr %zz
  call void @use_pointer(ptr %z)
  call void @llvm.objc.release(ptr %p)
  ret void
}

; CHECK-LABEL: define void @test52b(
; CHECK: call ptr @llvm.objc.retain
; CHECK: call void @callee()
; CHECK: call void @use_pointer(ptr %z)
; CHECK: call void @llvm.objc.release
; CHECK: ret void
; CHECK: {{^}}}
define void @test52b(ptr %zz, ptr %pp) {
  %p = load ptr, ptr %pp
  %1 = call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  %z = load ptr, ptr %zz
  call void @use_pointer(ptr %z)
  call void @llvm.objc.release(ptr %p), !clang.imprecise_release !0
  ret void
}

; Like test52, but the pointer has function type, so it's assumed to
; be not reference counted.
; Oops. That's wrong. Clang sometimes uses function types gratuitously.
; See rdar://10551239.

; CHECK-LABEL: define void @test53(
; CHECK: @llvm.objc.
; CHECK: {{^}}}
define void @test53(ptr %zz, ptr %pp) {
  %p = load ptr, ptr %pp
  %1 = call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  %z = load ptr, ptr %zz
  call void @callee_fnptr(ptr %z)
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Convert autorelease to release if the value is unused.

; CHECK-LABEL: define void @test54(
; CHECK: call ptr @returner()
; CHECK-NEXT: call void @llvm.objc.release(ptr %t) [[NUW]], !clang.imprecise_release ![[RELEASE]]
; CHECK-NEXT: ret void
; CHECK: {{^}}}
define void @test54() {
  %t = call ptr @returner()
  call ptr @llvm.objc.autorelease(ptr %t)
  ret void
}

; Nested retain+release pairs. Delete them both.

; CHECK-LABEL: define void @test55(
; CHECK-NOT: @objc
; CHECK: {{^}}}
define void @test55(ptr %x) {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %1 = call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @llvm.objc.release(ptr %x) nounwind
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Nested retain+release pairs where the inner pair depends
; on the outer pair to be removed, and then the outer pair
; can be partially eliminated. Plus an extra outer pair to
; eliminate, for fun.

; CHECK-LABEL: define void @test56(
; CHECK-NOT: @objc
; CHECK: if.then:
; CHECK-NEXT: %0 = tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK-NEXT: tail call void @use_pointer(ptr %x)
; CHECK-NEXT: tail call void @use_pointer(ptr %x)
; CHECK-NEXT: tail call void @llvm.objc.release(ptr %x) [[NUW]], !clang.imprecise_release ![[RELEASE]]
; CHECK-NEXT: br label %if.end
; CHECK-NOT: @objc
; CHECK: {{^}}}
define void @test56(ptr %x, i32 %n) {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %x) nounwind
  %1 = tail call ptr @llvm.objc.retain(ptr %0) nounwind
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %2 = tail call ptr @llvm.objc.retain(ptr %1) nounwind
  tail call void @use_pointer(ptr %2)
  tail call void @use_pointer(ptr %2)
  tail call void @llvm.objc.release(ptr %2) nounwind, !clang.imprecise_release !0
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  tail call void @llvm.objc.release(ptr %1) nounwind, !clang.imprecise_release !0
  tail call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; When there are adjacent retain+release pairs, the first one is known
; unnecessary because the presence of the second one means that the first one
; won't be deleting the object.

; CHECK-LABEL:      define void @test57(
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   call void @llvm.objc.release(ptr %x) [[NUW]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test57(ptr %x) nounwind {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; An adjacent retain+release pair is sufficient even if it will be
; removed itself.

; CHECK-LABEL:      define void @test58(
; CHECK-NEXT: entry:
; CHECK-NEXT:   @llvm.objc.retain
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test58(ptr %x) nounwind {
entry:
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Don't delete the second retain+release pair in an adjacent set.

; CHECK-LABEL:      define void @test59(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call ptr @llvm.objc.retain(ptr %x) [[NUW]]
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   call void @llvm.objc.release(ptr %x) [[NUW]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test59(ptr %x) nounwind {
entry:
  %a = call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @llvm.objc.release(ptr %x) nounwind
  %b = call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %x)
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Constant pointers to objects don't need reference counting.

@constptr = external constant ptr
@something = external global ptr

; We have a precise lifetime retain/release here. We can not remove them since
; @something is not constant.

; CHECK-LABEL: define void @test60a(
; CHECK: call ptr @llvm.objc.retain
; CHECK: call void @llvm.objc.release
; CHECK: {{^}}}
define void @test60a() {
  %t = load ptr, ptr @constptr
  %s = load ptr, ptr @something
  call ptr @llvm.objc.retain(ptr %s)
  call void @callee()
  call void @use_pointer(ptr %t)
  call void @llvm.objc.release(ptr %s)
  ret void
}

; CHECK-LABEL: define void @test60b(
; CHECK: call ptr @llvm.objc.retain
; CHECK-NOT: call ptr @llvm.objc.retain
; CHECK-NOT: call ptr @llvm.objc.release
; CHECK: {{^}}}
define void @test60b() {
  %t = load ptr, ptr @constptr
  %s = load ptr, ptr @something
  call ptr @llvm.objc.retain(ptr %t)
  call ptr @llvm.objc.retain(ptr %t)
  call void @callee()
  call void @use_pointer(ptr %s)
  call void @llvm.objc.release(ptr %t)
  ret void
}

; CHECK-LABEL: define void @test60c(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test60c() {
  %t = load ptr, ptr @constptr
  %s = load ptr, ptr @something
  call ptr @llvm.objc.retain(ptr %t)
  call void @callee()
  call void @use_pointer(ptr %s)
  call void @llvm.objc.release(ptr %t), !clang.imprecise_release !0
  ret void
}

; CHECK-LABEL: define void @test60d(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test60d() {
  %t = load ptr, ptr @constptr
  %s = load ptr, ptr @something
  call ptr @llvm.objc.retain(ptr %t)
  call void @callee()
  call void @use_pointer(ptr %s)
  call void @llvm.objc.release(ptr %t)
  ret void
}

; CHECK-LABEL: define void @test60e(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test60e() {
  %t = load ptr, ptr @constptr
  %s = load ptr, ptr @something
  call ptr @llvm.objc.retain(ptr %t)
  call void @callee()
  call void @use_pointer(ptr %s)
  call void @llvm.objc.release(ptr %t), !clang.imprecise_release !0
  ret void
}

; Constant pointers to objects don't need to be considered related to other
; pointers.

; CHECK-LABEL: define void @test61(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test61() {
  %t = load ptr, ptr @constptr
  call ptr @llvm.objc.retain(ptr %t)
  call void @callee()
  call void @use_pointer(ptr %t)
  call void @llvm.objc.release(ptr %t)
  ret void
}

; Delete a retain matched by releases when one is inside the loop and the
; other is outside the loop.

; CHECK-LABEL: define void @test62(
; CHECK-NOT: @llvm.objc.
; CHECK: {{^}}}
define void @test62(ptr %x, ptr %p) nounwind {
entry:
  br label %loop

loop:
  call ptr @llvm.objc.retain(ptr %x)
  %q = load i1, ptr %p
  br i1 %q, label %loop.more, label %exit

loop.more:
  call void @llvm.objc.release(ptr %x)
  br label %loop

exit:
  call void @llvm.objc.release(ptr %x)
  ret void
}

; Like test62 but with no release in exit.
; Don't delete anything!

; CHECK-LABEL: define void @test63(
; CHECK: loop:
; CHECK:   tail call ptr @llvm.objc.retain(ptr %x)
; CHECK: loop.more:
; CHECK:   call void @llvm.objc.release(ptr %x)
; CHECK: {{^}}}
define void @test63(ptr %x, ptr %p) nounwind {
entry:
  br label %loop

loop:
  call ptr @llvm.objc.retain(ptr %x)
  %q = load i1, ptr %p
  br i1 %q, label %loop.more, label %exit

loop.more:
  call void @llvm.objc.release(ptr %x)
  br label %loop

exit:
  ret void
}

; Like test62 but with no release in loop.more.
; Don't delete anything!

; CHECK-LABEL: define void @test64(
; CHECK: loop:
; CHECK:   tail call ptr @llvm.objc.retain(ptr %x)
; CHECK: exit:
; CHECK:   call void @llvm.objc.release(ptr %x)
; CHECK: {{^}}}
define void @test64(ptr %x, ptr %p) nounwind {
entry:
  br label %loop

loop:
  call ptr @llvm.objc.retain(ptr %x)
  %q = load i1, ptr %p
  br i1 %q, label %loop.more, label %exit

loop.more:
  br label %loop

exit:
  call void @llvm.objc.release(ptr %x)
  ret void
}

; Move an autorelease past a phi with a null.

; CHECK-LABEL: define ptr @test65(
; CHECK: if.then:
; CHECK:   call ptr @llvm.objc.autorelease(
; CHECK: return:
; CHECK-NOT: @llvm.objc.autorelease
; CHECK: {{^}}}
define ptr @test65(i1 %x) {
entry:
  br i1 %x, label %return, label %if.then

if.then:                                          ; preds = %entry
  %c = call ptr @returner()
  %s = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %c) nounwind
  br label %return

return:                                           ; preds = %if.then, %entry
  %retval = phi ptr [ %s, %if.then ], [ null, %entry ]
  %q = call ptr @llvm.objc.autorelease(ptr %retval) nounwind
  ret ptr %retval
}

; Don't move an autorelease past an autorelease pool boundary.

; CHECK-LABEL: define ptr @test65b(
; CHECK: if.then:
; CHECK-NOT: @llvm.objc.autorelease
; CHECK: return:
; CHECK:   call ptr @llvm.objc.autorelease(
; CHECK: {{^}}}
define ptr @test65b(i1 %x) {
entry:
  %t = call ptr @llvm.objc.autoreleasePoolPush()
  br i1 %x, label %return, label %if.then

if.then:                                          ; preds = %entry
  %c = call ptr @returner()
  %s = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %c) nounwind
  br label %return

return:                                           ; preds = %if.then, %entry
  %retval = phi ptr [ %s, %if.then ], [ null, %entry ]
  call void @llvm.objc.autoreleasePoolPop(ptr %t)
  %q = call ptr @llvm.objc.autorelease(ptr %retval) nounwind
  ret ptr %retval
}

; Don't move an autoreleaseReuturnValue, which would break
; the RV optimization.

; CHECK-LABEL: define ptr @test65c(
; CHECK: if.then:
; CHECK-NOT: @llvm.objc.autorelease
; CHECK: return:
; CHECK:   call ptr @llvm.objc.autoreleaseReturnValue(
; CHECK: {{^}}}
define ptr @test65c(i1 %x) {
entry:
  br i1 %x, label %return, label %if.then

if.then:                                          ; preds = %entry
  %c = call ptr @returner()
  %s = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %c) nounwind
  br label %return

return:                                           ; preds = %if.then, %entry
  %retval = phi ptr [ %s, %if.then ], [ null, %entry ]
  %q = call ptr @llvm.objc.autoreleaseReturnValue(ptr %retval) nounwind
  ret ptr %retval
}

; CHECK-LABEL: define ptr @test65d(
; CHECK: if.then:
; CHECK-NOT: @llvm.objc.autorelease
; CHECK: return:
; CHECK:   call ptr @llvm.objc.autoreleaseReturnValue(
; CHECK: {{^}}}
define ptr @test65d(i1 %x) {
entry:
  br i1 %x, label %return, label %if.then

if.then:                                          ; preds = %entry
  %c = call ptr @returner()
  %s = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %c) nounwind
  br label %return

return:                                           ; preds = %if.then, %entry
  %retval = phi ptr [ %s, %if.then ], [ null, %entry ]
  %q = call ptr @llvm.objc.autoreleaseReturnValue(ptr %retval) nounwind
  ret ptr %retval
}

; An llvm.objc.retain can serve as a may-use for a different pointer.
; rdar://11931823

; CHECK-LABEL: define void @test66a(
; CHECK:   tail call ptr @llvm.objc.retain(ptr %cond) [[NUW]]
; CHECK:   tail call void @llvm.objc.release(ptr %call) [[NUW]]
; CHECK:   tail call ptr @llvm.objc.retain(ptr %tmp8) [[NUW]]
; CHECK:   tail call void @llvm.objc.release(ptr %cond) [[NUW]]
; CHECK: {{^}}}
define void @test66a(ptr %tmp5, ptr %bar, i1 %tobool, i1 %tobool1, ptr %call) {
entry:
  br i1 %tobool, label %cond.true, label %cond.end

cond.true:
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %entry
  %cond = phi ptr [ %tmp5, %cond.true ], [ %call, %entry ]
  %tmp7 = tail call ptr @llvm.objc.retain(ptr %cond) nounwind
  tail call void @llvm.objc.release(ptr %call) nounwind
  %tmp8 = select i1 %tobool1, ptr %cond, ptr %bar
  %tmp9 = tail call ptr @llvm.objc.retain(ptr %tmp8) nounwind
  tail call void @llvm.objc.release(ptr %cond) nounwind
  ret void
}

; CHECK-LABEL: define void @test66b(
; CHECK:   tail call ptr @llvm.objc.retain(ptr %cond) [[NUW]]
; CHECK:   tail call void @llvm.objc.release(ptr %call) [[NUW]]
; CHECK:   tail call ptr @llvm.objc.retain(ptr %tmp8) [[NUW]]
; CHECK:   tail call void @llvm.objc.release(ptr %cond) [[NUW]]
; CHECK: {{^}}}
define void @test66b(ptr %tmp5, ptr %bar, i1 %tobool, i1 %tobool1, ptr %call) {
entry:
  br i1 %tobool, label %cond.true, label %cond.end

cond.true:
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %entry
  %cond = phi ptr [ %tmp5, %cond.true ], [ %call, %entry ]
  %tmp7 = tail call ptr @llvm.objc.retain(ptr %cond) nounwind
  tail call void @llvm.objc.release(ptr %call) nounwind, !clang.imprecise_release !0
  %tmp8 = select i1 %tobool1, ptr %cond, ptr %bar
  %tmp9 = tail call ptr @llvm.objc.retain(ptr %tmp8) nounwind
  tail call void @llvm.objc.release(ptr %cond) nounwind
  ret void
}

; CHECK-LABEL: define void @test66c(
; CHECK:   tail call ptr @llvm.objc.retain(ptr %cond) [[NUW]]
; CHECK:   tail call void @llvm.objc.release(ptr %call) [[NUW]]
; CHECK:   tail call ptr @llvm.objc.retain(ptr %tmp8) [[NUW]]
; CHECK:   tail call void @llvm.objc.release(ptr %cond) [[NUW]]
; CHECK: {{^}}}
define void @test66c(ptr %tmp5, ptr %bar, i1 %tobool, i1 %tobool1, ptr %call) {
entry:
  br i1 %tobool, label %cond.true, label %cond.end

cond.true:
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %entry
  %cond = phi ptr [ %tmp5, %cond.true ], [ %call, %entry ]
  %tmp7 = tail call ptr @llvm.objc.retain(ptr %cond) nounwind
  tail call void @llvm.objc.release(ptr %call) nounwind
  %tmp8 = select i1 %tobool1, ptr %cond, ptr %bar
  %tmp9 = tail call ptr @llvm.objc.retain(ptr %tmp8) nounwind, !clang.imprecise_release !0
  tail call void @llvm.objc.release(ptr %cond) nounwind
  ret void
}

; CHECK-LABEL: define void @test66d(
; CHECK:   tail call ptr @llvm.objc.retain(ptr %cond) [[NUW]]
; CHECK:   tail call void @llvm.objc.release(ptr %call) [[NUW]]
; CHECK:   tail call ptr @llvm.objc.retain(ptr %tmp8) [[NUW]]
; CHECK:   tail call void @llvm.objc.release(ptr %cond) [[NUW]]
; CHECK: {{^}}}
define void @test66d(ptr %tmp5, ptr %bar, i1 %tobool, i1 %tobool1, ptr %call) {
entry:
  br i1 %tobool, label %cond.true, label %cond.end

cond.true:
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %entry
  %cond = phi ptr [ %tmp5, %cond.true ], [ %call, %entry ]
  %tmp7 = tail call ptr @llvm.objc.retain(ptr %cond) nounwind
  tail call void @llvm.objc.release(ptr %call) nounwind, !clang.imprecise_release !0
  %tmp8 = select i1 %tobool1, ptr %cond, ptr %bar
  %tmp9 = tail call ptr @llvm.objc.retain(ptr %tmp8) nounwind
  tail call void @llvm.objc.release(ptr %cond) nounwind, !clang.imprecise_release !0
  ret void
}

; A few real-world testcases.

@.str4 = private unnamed_addr constant [33 x i8] c"-[A z] = { %f, %f, { %f, %f } }\0A\00"
@"OBJC_IVAR_$_A.myZ" = global i64 20, section "__DATA, __objc_const", align 8
declare i32 @printf(ptr nocapture, ...) nounwind
declare i32 @puts(ptr nocapture) nounwind
@str = internal constant [16 x i8] c"-[ Top0 _getX ]\00"

; FIXME: Should be able to eliminate the retain and release
; CHECK-LABEL: define { <2 x float>, <2 x float> } @"\01-[A z]"(ptr %self, ptr nocapture %_cmd)
; CHECK: tail call ptr @llvm.objc.retain(ptr %self)
; CHECK-NEXT: %call = tail call i32 (ptr, ...) @printf(
; CHECK: tail call void @llvm.objc.release(ptr %self)
; CHECK: {{^}}}
define { <2 x float>, <2 x float> } @"\01-[A z]"(ptr %self, ptr nocapture %_cmd) nounwind {
invoke.cont:
  %i1 = tail call ptr @llvm.objc.retain(ptr %self) nounwind
  tail call void @llvm.dbg.value(metadata ptr %self, metadata !DILocalVariable(scope: !2), metadata !DIExpression()), !dbg !DILocation(scope: !2)
  tail call void @llvm.dbg.value(metadata ptr %self, metadata !DILocalVariable(scope: !2), metadata !DIExpression()), !dbg !DILocation(scope: !2)
  %ivar = load i64, ptr @"OBJC_IVAR_$_A.myZ", align 8
  %add.ptr = getelementptr i8, ptr %self, i64 %ivar
  %tmp2 = load float, ptr %add.ptr, align 4
  %conv = fpext float %tmp2 to double
  %add.ptr.sum = add i64 %ivar, 4
  %tmp6 = getelementptr inbounds i8, ptr %self, i64 %add.ptr.sum
  %tmp7 = load float, ptr %tmp6, align 4
  %conv8 = fpext float %tmp7 to double
  %add.ptr.sum36 = add i64 %ivar, 8
  %tmp12 = getelementptr inbounds i8, ptr %self, i64 %add.ptr.sum36
  %tmp13 = load float, ptr %tmp12, align 4
  %conv14 = fpext float %tmp13 to double
  %tmp12.sum = add i64 %ivar, 12
  %arrayidx19 = getelementptr inbounds i8, ptr %self, i64 %tmp12.sum
  %tmp20 = load float, ptr %arrayidx19, align 4
  %conv21 = fpext float %tmp20 to double
  %call = tail call i32 (ptr, ...) @printf(ptr @.str4, double %conv, double %conv8, double %conv14, double %conv21)
  %ivar23 = load i64, ptr @"OBJC_IVAR_$_A.myZ", align 8
  %add.ptr24 = getelementptr i8, ptr %self, i64 %ivar23
  %srcval = load i128, ptr %add.ptr24, align 4
  tail call void @llvm.objc.release(ptr %self) nounwind
  %tmp29 = trunc i128 %srcval to i64
  %tmp30 = bitcast i64 %tmp29 to <2 x float>
  %tmp31 = insertvalue { <2 x float>, <2 x float> } undef, <2 x float> %tmp30, 0
  %tmp32 = lshr i128 %srcval, 64
  %tmp33 = trunc i128 %tmp32 to i64
  %tmp34 = bitcast i64 %tmp33 to <2 x float>
  %tmp35 = insertvalue { <2 x float>, <2 x float> } %tmp31, <2 x float> %tmp34, 1
  ret { <2 x float>, <2 x float> } %tmp35
}

; FIXME: Should be able to eliminate the retain and release
; CHECK-LABEL: @"\01-[Top0 _getX]"(ptr %self, ptr nocapture %_cmd)
; CHECK: tail call ptr @llvm.objc.retain(ptr %self)
; CHECK: %puts = tail call i32 @puts
; CHECK: tail call void @llvm.objc.release(ptr %self)
define i32 @"\01-[Top0 _getX]"(ptr %self, ptr nocapture %_cmd) nounwind {
invoke.cont:
  %i1 = tail call ptr @llvm.objc.retain(ptr %self) nounwind
  %puts = tail call i32 @puts(ptr @str)
  tail call void @llvm.objc.release(ptr %self) nounwind
  ret i32 0
}

@"\01L_OBJC_METH_VAR_NAME_" = internal global [5 x i8] c"frob\00", section "__TEXT,__cstring,cstring_literals", align 1@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global ptr @"\01L_OBJC_METH_VAR_NAME_", section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__DATA, __objc_imageinfo, regular, no_dead_strip"
@llvm.used = appending global [3 x ptr] [ptr @"\01L_OBJC_METH_VAR_NAME_", ptr @"\01L_OBJC_SELECTOR_REFERENCES_", ptr @"\01L_OBJC_IMAGE_INFO"], section "llvm.metadata"

; A simple loop. Eliminate the retain and release inside of it!

; CHECK: define void @loop(ptr %x, i64 %n) {
; CHECK: for.body:
; CHECK-NOT: @llvm.objc.
; CHECK: @objc_msgSend
; CHECK-NOT: @llvm.objc.
; CHECK: for.end:
; CHECK: {{^}}}
define void @loop(ptr %x, i64 %n) {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %x) nounwind
  %cmp9 = icmp sgt i64 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %1 = tail call ptr @llvm.objc.retain(ptr %x) nounwind
  %tmp5 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call = tail call ptr (ptr, ptr, ...) @objc_msgSend(ptr %1, ptr %tmp5)
  tail call void @llvm.objc.release(ptr %1) nounwind, !clang.imprecise_release !0
  %inc = add nsw i64 %i.010, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  tail call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}

; ObjCARCOpt can delete the retain,release on self.

; CHECK: define void @TextEditTest(ptr %self, ptr %pboard) {
; CHECK-NOT: call ptr @llvm.objc.retain(ptr %tmp7)
; CHECK: {{^}}}

%0 = type { ptr, ptr }
%1 = type opaque
%2 = type opaque
%3 = type opaque
%4 = type opaque
%5 = type opaque
%struct.NSConstantString = type { ptr, i32, ptr, i64 }
%struct._NSRange = type { i64, i64 }
%struct.__CFString = type opaque
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }
%struct._message_ref_t = type { ptr, ptr }
%struct._objc_cache = type opaque
%struct._objc_method = type { ptr, ptr, ptr }
%struct._objc_protocol_list = type { i64, [0 x ptr] }
%struct._prop_list_t = type { i32, i32, [0 x %struct._message_ref_t] }
%struct._protocol_t = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32 }

@"\01L_OBJC_CLASSLIST_REFERENCES_$_17" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@kUTTypePlainText = external constant ptr
@"\01L_OBJC_SELECTOR_REFERENCES_19" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_21" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_23" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_25" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_26" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_28" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_29" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_31" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_33" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_35" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_37" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_38" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_40" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_42" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@_unnamed_cfstring_44 = external hidden constant %struct.NSConstantString, section "__DATA,__cfstring"
@"\01L_OBJC_SELECTOR_REFERENCES_46" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_48" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01l_objc_msgSend_fixup_isEqual_" = external hidden global %0, section "__DATA, __objc_msgrefs, coalesced", align 16
@"\01L_OBJC_CLASSLIST_REFERENCES_$_50" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@NSCocoaErrorDomain = external constant ptr
@"\01L_OBJC_CLASSLIST_REFERENCES_$_51" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@NSFilePathErrorKey = external constant ptr
@"\01L_OBJC_SELECTOR_REFERENCES_53" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_55" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_56" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_58" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_60" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

declare ptr @truncatedString(ptr, i64)
define void @TextEditTest(ptr %self, ptr %pboard) {
entry:
  %err = alloca ptr, align 8
  %tmp8 = call ptr @llvm.objc.retain(ptr %self) nounwind
  store ptr null, ptr %err, align 8
  %tmp1 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_17", align 8
  %tmp2 = load ptr, ptr @kUTTypePlainText, align 8
  %tmp3 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_19", align 8
  %call5 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %tmp1, ptr %tmp3, ptr %tmp2)
  %tmp5 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_21", align 8
  %call76 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %pboard, ptr %tmp5, ptr %call5)
  %tmp9 = call ptr @llvm.objc.retain(ptr %call76) nounwind
  %tobool = icmp eq ptr %tmp9, null
  br i1 %tobool, label %end, label %land.lhs.true

land.lhs.true:                                    ; preds = %entry
  %tmp11 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_23", align 8
  %call137 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %pboard, ptr %tmp11, ptr %tmp9)
  %tmp10 = call ptr @llvm.objc.retain(ptr %call137) nounwind
  call void @llvm.objc.release(ptr null) nounwind
  %tmp12 = call ptr @llvm.objc.retain(ptr %call137) nounwind
  call void @llvm.objc.release(ptr null) nounwind
  %tobool16 = icmp eq ptr %call137, null
  br i1 %tobool16, label %end, label %if.then

if.then:                                          ; preds = %land.lhs.true
  %tmp19 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_25", align 8
  %call21 = call signext i8 @objc_msgSend(ptr %call137, ptr %tmp19)
  %tobool22 = icmp eq i8 %call21, 0
  br i1 %tobool22, label %if.then44, label %land.lhs.true23

land.lhs.true23:                                  ; preds = %if.then
  %tmp24 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_26", align 8
  %tmp26 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_28", align 8
  %call2822 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %tmp24, ptr %tmp26, ptr %call137)
  %tmp14 = call ptr @llvm.objc.retain(ptr %call2822) nounwind
  call void @llvm.objc.release(ptr null) nounwind
  %tobool30 = icmp eq ptr %call2822, null
  br i1 %tobool30, label %if.then44, label %if.end

if.end:                                           ; preds = %land.lhs.true23
  %tmp32 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_29", align 8
  %tmp33 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_31", align 8
  %call35 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %tmp32, ptr %tmp33)
  %tmp37 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_33", align 8
  %call3923 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %call35, ptr %tmp37, ptr %call2822, i32 signext 1, ptr %err)
  %cmp = icmp eq ptr %call3923, null
  br i1 %cmp, label %if.then44, label %end

if.then44:                                        ; preds = %if.end, %land.lhs.true23, %if.then
  %url.025 = phi ptr [ %call2822, %if.end ], [ %call2822, %land.lhs.true23 ], [ null, %if.then ]
  %tmp49 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_35", align 8
  %call51 = call %struct._NSRange @objc_msgSend(ptr %call137, ptr %tmp49, i64 0, i64 0)
  %call513 = extractvalue %struct._NSRange %call51, 0
  %call514 = extractvalue %struct._NSRange %call51, 1
  %tmp52 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_37", align 8
  %call548 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %call137, ptr %tmp52, i64 %call513, i64 %call514)
  %tmp55 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_38", align 8
  %tmp56 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_40", align 8
  %call58 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %tmp55, ptr %tmp56)
  %tmp59 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_42", align 8
  %call6110 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %call548, ptr %tmp59, ptr %call58)
  %tmp15 = call ptr @llvm.objc.retain(ptr %call6110) nounwind
  call void @llvm.objc.release(ptr %call137) nounwind
  %tmp64 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_46", align 8
  %call66 = call signext i8 @objc_msgSend(ptr %call6110, ptr %tmp64, ptr @_unnamed_cfstring_44)
  %tobool67 = icmp eq i8 %call66, 0
  br i1 %tobool67, label %if.end74, label %if.then68

if.then68:                                        ; preds = %if.then44
  %tmp70 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_48", align 8
  %call7220 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %call6110, ptr %tmp70)
  %tmp16 = call ptr @llvm.objc.retain(ptr %call7220) nounwind
  call void @llvm.objc.release(ptr %call6110) nounwind
  br label %if.end74

if.end74:                                         ; preds = %if.then68, %if.then44
  %filename.0.in = phi ptr [ %call7220, %if.then68 ], [ %call6110, %if.then44 ]
  %tmp17 = load ptr, ptr @"\01l_objc_msgSend_fixup_isEqual_", align 16
  %call78 = call signext i8 (ptr, ptr, ptr, ...) %tmp17(ptr %call137, ptr @"\01l_objc_msgSend_fixup_isEqual_", ptr %filename.0.in)
  %tobool79 = icmp eq i8 %call78, 0
  br i1 %tobool79, label %land.lhs.true80, label %if.then109

land.lhs.true80:                                  ; preds = %if.end74
  %tmp82 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_25", align 8
  %call84 = call signext i8 @objc_msgSend(ptr %filename.0.in, ptr %tmp82)
  %tobool86 = icmp eq i8 %call84, 0
  br i1 %tobool86, label %if.then109, label %if.end106

if.end106:                                        ; preds = %land.lhs.true80
  %tmp88 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_26", align 8
  %tmp90 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_28", align 8
  %call9218 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %tmp88, ptr %tmp90, ptr %filename.0.in)
  %tmp21 = call ptr @llvm.objc.retain(ptr %call9218) nounwind
  call void @llvm.objc.release(ptr %url.025) nounwind
  %tmp94 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_29", align 8
  %tmp95 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_31", align 8
  %call97 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %tmp94, ptr %tmp95)
  %tmp99 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_33", align 8
  %call10119 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %call97, ptr %tmp99, ptr %call9218, i32 signext 1, ptr %err)
  %phitmp = icmp eq ptr %call10119, null
  br i1 %phitmp, label %if.then109, label %end

if.then109:                                       ; preds = %if.end106, %land.lhs.true80, %if.end74
  %url.129 = phi ptr [ %call9218, %if.end106 ], [ %url.025, %if.end74 ], [ %url.025, %land.lhs.true80 ]
  %tmp110 = load ptr, ptr %err, align 8
  %tobool111 = icmp eq ptr %tmp110, null
  br i1 %tobool111, label %if.then112, label %if.end125

if.then112:                                       ; preds = %if.then109
  %tmp113 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_50", align 8
  %tmp114 = load ptr, ptr @NSCocoaErrorDomain, align 8
  %tmp115 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_51", align 8
  %call117 = call ptr @truncatedString(ptr %filename.0.in, i64 1034)
  %tmp118 = load ptr, ptr @NSFilePathErrorKey, align 8
  %tmp119 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_53", align 8
  %call12113 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %tmp115, ptr %tmp119, ptr %call117, ptr %tmp118, ptr null)
  %tmp122 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_55", align 8
  %call12414 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %tmp113, ptr %tmp122, ptr %tmp114, i64 258, ptr %call12113)
  %tmp23 = call ptr @llvm.objc.retain(ptr %call12414) nounwind
  %tmp25 = call ptr @llvm.objc.autorelease(ptr %tmp23) nounwind
  store ptr %tmp25, ptr %err, align 8
  br label %if.end125

if.end125:                                        ; preds = %if.then112, %if.then109
  %tmp127 = phi ptr [ %tmp110, %if.then109 ], [ %tmp25, %if.then112 ]
  %tmp126 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_56", align 8
  %tmp128 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_58", align 8
  %call13015 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %tmp126, ptr %tmp128, ptr %tmp127)
  %tmp131 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_60", align 8
  %call13317 = call ptr (ptr, ptr, ...) @objc_msgSend(ptr %call13015, ptr %tmp131)
  br label %end

end:                                              ; preds = %if.end125, %if.end106, %if.end, %land.lhs.true, %entry
  %filename.2 = phi ptr [ %filename.0.in, %if.end106 ], [ %filename.0.in, %if.end125 ], [ %call137, %land.lhs.true ], [ null, %entry ], [ %call137, %if.end ]
  %origFilename.0 = phi ptr [ %call137, %if.end106 ], [ %call137, %if.end125 ], [ %call137, %land.lhs.true ], [ null, %entry ], [ %call137, %if.end ]
  %url.2 = phi ptr [ %call9218, %if.end106 ], [ %url.129, %if.end125 ], [ null, %land.lhs.true ], [ null, %entry ], [ %call2822, %if.end ]
  call void @llvm.objc.release(ptr %tmp9) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %url.2) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %origFilename.0) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %filename.2) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %self) nounwind, !clang.imprecise_release !0
  ret void
}

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.objc.sync.enter(ptr)
declare i32 @llvm.objc.sync.exit(ptr)

; Make sure that we understand that objc_sync_{enter,exit} are IC_User not
; IC_Call/IC_CallOrUser.

; CHECK-LABEL:      define void @test67(
; CHECK-NEXT:   call i32 @llvm.objc.sync.enter(ptr %x)
; CHECK-NEXT:   call i32 @llvm.objc.sync.exit(ptr %x)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test67(ptr %x) {
  call ptr @llvm.objc.retain(ptr %x)
  call i32 @llvm.objc.sync.enter(ptr %x)
  call i32 @llvm.objc.sync.exit(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}

; CHECK-LABEL: define void @test68(
; CHECK-NOT:     call
; CHECK:         call void @callee2(
; CHECK-NOT:     call
; CHECK:         ret void

define void @test68(ptr %a, ptr %b) {
  call ptr @llvm.objc.retain(ptr %a)
  call ptr @llvm.objc.retain(ptr %b)
  call void @callee2(ptr %a, ptr %b)
  call void @llvm.objc.release(ptr %b), !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %a), !clang.imprecise_release !0
  ret void
}

!llvm.module.flags = !{!1}
!llvm.dbg.cu = !{!3}

!0 = !{}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = distinct !DISubprogram(unit: !3)
!3 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !4,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
!4 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!5 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: attributes [[NUW]] = { nounwind }
; CHECK: ![[RELEASE]] = !{}
