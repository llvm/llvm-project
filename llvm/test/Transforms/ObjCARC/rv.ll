; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare ptr @llvm.objc.retain(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare void @llvm.objc.release(ptr)
declare ptr @llvm.objc.autorelease(ptr)
declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare ptr @llvm.objc.retainAutoreleaseReturnValue(ptr)
declare void @llvm.objc.autoreleasePoolPop(ptr)
declare void @llvm.objc.autoreleasePoolPush()
declare ptr @llvm.objc.retainBlock(ptr)
declare void @llvm.objc.clang.arc.noop.use(...)

declare ptr @objc_retainedObject(ptr)
declare ptr @objc_unretainedObject(ptr)
declare ptr @objc_unretainedPointer(ptr)

declare void @use_pointer(ptr)
declare void @callee()
declare void @callee_fnptr(ptr)
declare void @invokee()
declare ptr @returner()
declare ptr @returner1(ptr)
declare i32 @__gxx_personality_v0(...)

; Test that retain+release elimination is suppressed when the
; retain is an objc_retainAutoreleasedReturnValue, since it's
; better to do the RV optimization.

; CHECK-LABEL:      define void @test0(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x = call ptr @returner
; CHECK-NEXT:   %0 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %x) [[NUW:#[0-9]+]]
; CHECK: t:
; CHECK-NOT: @llvm.objc.
; CHECK: return:
; CHECK-NEXT: call void @llvm.objc.release(ptr %x)
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test0(i1 %p) nounwind {
entry:
  %x = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %x)
  br i1 %p, label %t, label %return

t:
  call void @use_pointer(ptr %x)
  store i8 0, ptr %x
  br label %return

return:
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; Delete no-ops.

; CHECK-LABEL: define void @test2(
; CHECK-NOT: @llvm.objc.
; CHECK: }
define void @test2() {
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr null)
  call ptr @llvm.objc.autoreleaseReturnValue(ptr null)
  ; call ptr @llvm.objc.retainAutoreleaseReturnValue(ptr null) ; TODO
  %rb = call ptr @llvm.objc.retainBlock(ptr null)
  call void @use_pointer(ptr %rb)
  %rb2 = call ptr @llvm.objc.retainBlock(ptr undef)
  call void @use_pointer(ptr %rb2)
  ret void
}

; Delete a redundant retainRV,autoreleaseRV when forwaring a call result
; directly to a return value.

; CHECK-LABEL: define ptr @test3(
; CHECK: call ptr @returner()
; CHECK-NEXT: ret ptr %call
define ptr @test3() {
entry:
  %call = tail call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  %1 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %0) nounwind
  ret ptr %1
}

; Delete a redundant retain,autoreleaseRV when forwaring a call result
; directly to a return value.

; CHECK-LABEL: define ptr @test4(
; CHECK: call ptr @returner()
; CHECK-NEXT: ret ptr %call
define ptr @test4() {
entry:
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retain(ptr %call) nounwind
  %1 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %0) nounwind
  ret ptr %1
}

; Delete a redundant fused retain+autoreleaseRV when forwaring a call result
; directly to a return value.

; TODO
; COM: CHECK: define ptr @test5
; COM: CHECK: call ptr @returner()
; COM: CHECK-NEXT: ret ptr %call
;define ptr @test5() {
;entry:
;  %call = call ptr @returner()
;  %0 = call ptr @llvm.objc.retainAutoreleaseReturnValue(ptr %call) nounwind
;  ret ptr %0
;}

; Don't eliminate objc_retainAutoreleasedReturnValue by merging it into
; an objc_autorelease.
; TODO? Merge objc_retainAutoreleasedReturnValue and objc_autorelease into
; objc_retainAutoreleasedReturnValueAutorelease and merge
; objc_retainAutoreleasedReturnValue and objc_autoreleaseReturnValue
; into objc_retainAutoreleasedReturnValueAutoreleaseReturnValue?
; Those entrypoints don't exist yet though.

; CHECK-LABEL: define ptr @test7(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p)
; CHECK: %t = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
define ptr @test7() {
  %p = call ptr @returner()
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p)
  %t = call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
  call void @use_pointer(ptr %p)
  ret ptr %t
}

; CHECK-LABEL: define ptr @test7b(
; CHECK: call ptr @llvm.objc.retain(ptr %p)
; CHECK: %t = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
define ptr @test7b() {
  %p = call ptr @returner()
  call void @use_pointer(ptr %p)
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p)
  %t = call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
  ret ptr %p
}

; Don't apply the RV optimization to autorelease if there's no retain.

; CHECK: define ptr @test9(ptr %p)
; CHECK: call ptr @llvm.objc.autorelease(ptr %p)
define ptr @test9(ptr %p) {
  call ptr @llvm.objc.autorelease(ptr %p)
  ret ptr %p
}

; Do not apply the RV optimization.

; CHECK: define ptr @test10(ptr %p)
; CHECK: tail call ptr @llvm.objc.retain(ptr %p) [[NUW]]
; CHECK: call ptr @llvm.objc.autorelease(ptr %p) [[NUW]]
; CHECK-NEXT: ret ptr %p
define ptr @test10(ptr %p) {
  %1 = call ptr @llvm.objc.retain(ptr %p)
  %2 = call ptr @llvm.objc.autorelease(ptr %p)
  ret ptr %p
}

; Don't do the autoreleaseRV optimization because @use_pointer
; could undo the retain.

; CHECK: define ptr @test11(ptr %p)
; CHECK: tail call ptr @llvm.objc.retain(ptr %p)
; CHECK-NEXT: call void @use_pointer(ptr %p)
; CHECK: call ptr @llvm.objc.autorelease(ptr %p)
; CHECK-NEXT: ret ptr %p
define ptr @test11(ptr %p) {
  %1 = call ptr @llvm.objc.retain(ptr %p)
  call void @use_pointer(ptr %p)
  %2 = call ptr @llvm.objc.autorelease(ptr %p)
  ret ptr %p
}

; Don't spoil the RV optimization.

; CHECK: define ptr @test12(ptr %p)
; CHECK: tail call ptr @llvm.objc.retain(ptr %p)
; CHECK: call void @use_pointer(ptr %p)
; CHECK: tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
; CHECK: ret ptr %p
define ptr @test12(ptr %p) {
  %1 = call ptr @llvm.objc.retain(ptr %p)
  call void @use_pointer(ptr %p)
  %2 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
  ret ptr %p
}

; Don't zap the objc_retainAutoreleasedReturnValue.

; CHECK-LABEL: define ptr @test13(
; CHECK: tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p)
; CHECK: call ptr @llvm.objc.autorelease(ptr %p)
; CHECK: ret ptr %p
define ptr @test13() {
  %p = call ptr @returner()
  %1 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p)
  call void @callee()
  %2 = call ptr @llvm.objc.autorelease(ptr %p)
  ret ptr %p
}

; Convert objc_retainAutoreleasedReturnValue to objc_retain if its
; argument is not a return value.

; CHECK-LABEL: define void @test14(
; CHECK-NEXT: tail call ptr @llvm.objc.retain(ptr %p) [[NUW]]
; CHECK-NEXT: ret void
define void @test14(ptr %p) {
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p)
  ret void
}

; Don't convert objc_retainAutoreleasedReturnValue to objc_retain if its
; argument is a return value.

; CHECK-LABEL: define void @test15(
; CHECK-NEXT: %y = call ptr @returner()
; CHECK-NEXT: tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %y) [[NUW]]
; CHECK-NEXT: ret void
define void @test15() {
  %y = call ptr @returner()
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %y)
  ret void
}

; Delete autoreleaseRV+retainRV pairs.

; CHECK: define ptr @test19(ptr %p) {
; CHECK-NEXT: ret ptr %p
define ptr @test19(ptr %p) {
  call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p)
  ret ptr %p
}

; Delete autoreleaseRV+retainRV pairs when they have equivalent PHIs as inputs

; CHECK: define ptr @test19phi(ptr %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT: br label %test19bb
; CHECK: test19bb:
; CHECK-NEXT: ret ptr %p
define ptr @test19phi(ptr %p) {
entry:
  br label %test19bb
test19bb:
  %phi1 = phi ptr [ %p, %entry ]
  %phi2 = phi ptr [ %p, %entry ]
  call ptr @llvm.objc.autoreleaseReturnValue(ptr %phi1)
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %phi2)
  ret ptr %p
}

; Like test19 but with plain autorelease.

; CHECK: define ptr @test20(ptr %p) {
; CHECK-NEXT: call ptr @llvm.objc.autorelease(ptr %p)
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %p)
; CHECK-NEXT: ret ptr %p
define ptr @test20(ptr %p) {
  call ptr @llvm.objc.autorelease(ptr %p)
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p)
  ret ptr %p
}

; Like test19 but with plain retain.

; CHECK: define ptr @test21(ptr %p) {
; CHECK-NEXT: call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %p)
; CHECK-NEXT: ret ptr %p
define ptr @test21(ptr %p) {
  call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
  call ptr @llvm.objc.retain(ptr %p)
  ret ptr %p
}

; Like test19 but with plain retain and autorelease.

; CHECK: define ptr @test22(ptr %p) {
; CHECK-NEXT: call ptr @llvm.objc.autorelease(ptr %p)
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %p)
; CHECK-NEXT: ret ptr %p
define ptr @test22(ptr %p) {
  call ptr @llvm.objc.autorelease(ptr %p)
  call ptr @llvm.objc.retain(ptr %p)
  ret ptr %p
}

; Convert autoreleaseRV to autorelease.

; CHECK-LABEL: define void @test23(
; CHECK: call ptr @llvm.objc.autorelease(ptr %p) [[NUW]]
define void @test23(ptr %p) {
  store i8 0, ptr %p
  call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
  ret void
}

; Don't convert autoreleaseRV to autorelease if the result is returned,
; even through a bitcast.

; CHECK-LABEL: define ptr @test24(
; CHECK: tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
define ptr @test24(ptr %p) {
  %t = call ptr @llvm.objc.autoreleaseReturnValue(ptr %p)
  ret ptr %p
}

declare ptr @first_test25();
declare ptr @second_test25(ptr);
declare void @somecall_test25();

; ARC optimizer used to move the last release between the call to second_test25
; and the call to objc_retainAutoreleasedReturnValue, causing %second to be
; released prematurely when %first and %second were pointing to the same object.

; CHECK-LABEL: define void @test25(
; CHECK: %[[CALL1:.*]] = call ptr @second_test25(
; CHECK-NEXT: tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %[[CALL1]])

define void @test25() {
  %first = call ptr @first_test25()
  %v0 = call ptr @llvm.objc.retain(ptr %first)
  call void @somecall_test25()
  %second = call ptr @second_test25(ptr %first)
  %call2 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %second)
  call void @llvm.objc.release(ptr %second), !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %first), !clang.imprecise_release !0
  ret void
}

; Check that ObjCARCOpt::OptimizeReturns removes the redundant calls even when
; they are not in the same basic block. This code used to cause an assertion
; failure.

; CHECK-LABEL: define ptr @test26()
; CHECK: call ptr @returner()
; CHECK-NOT:  call
define ptr @test26() {
bb0:
  %v0 = call ptr @returner()
  %v1 = tail call ptr @llvm.objc.retain(ptr %v0)
  br label %bb1
bb1:
  %v2 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %v1)
  br label %bb2
bb2:
  ret ptr %v2
}

declare ptr @func27(i32);

; Check that ObjCARCOpt::OptimizeAutoreleaseRVCall doesn't turn a call to
; @llvm.objc.autoreleaseReturnValue into a call to @llvm.objc.autorelease when a return
; instruction uses a value equivalent to @llvm.objc.autoreleaseReturnValue's operand.
; In the code below, %phival and %retval are considered equivalent.

; CHECK-LABEL: define ptr @test27(
; CHECK: %[[PHIVAL:.*]] = phi ptr [ %{{.*}}, %bb1 ], [ %{{.*}}, %bb2 ]
; CHECK: %[[RETVAL:.*]] = phi ptr [ %{{.*}}, %bb1 ], [ %{{.*}}, %bb2 ]
; CHECK: tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %[[PHIVAL]])
; CHECK: ret ptr %[[RETVAL]]

define ptr @test27(i1 %cond) {
entry:
  br i1 %cond, label %bb1, label %bb2
bb1:
  %v0 = call ptr @func27(i32 1)
  br label %bb3
bb2:
  %v2 = call ptr @func27(i32 2)
  br label %bb3
bb3:
  %phival = phi ptr [ %v0, %bb1 ], [ %v2, %bb2 ]
  %retval = phi ptr [ %v0, %bb1 ], [ %v2, %bb2 ]
  %v4 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %phival)
  ret ptr %retval
}

; Don't eliminate the retainRV/autoreleaseRV pair if the call isn't a tail call.

; CHECK-LABEL: define ptr @test28(
; CHECK: call ptr @returner()
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue(
; CHECK: call ptr @llvm.objc.autoreleaseReturnValue(
define ptr @test28() {
entry:
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  %1 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %0) nounwind
  ret ptr %1
}

; CHECK-LABEL: define ptr @test29(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue(
; CHECK: call ptr @llvm.objc.autoreleaseReturnValue(

define ptr @test29(ptr %k) local_unnamed_addr personality ptr @__gxx_personality_v0 {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %k)
  %call = invoke ptr @returner1(ptr %k)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  %1 = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call)
  tail call void @llvm.objc.release(ptr %k), !clang.imprecise_release !0
  %2 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call)
  ret ptr %call

lpad:
  %3 = landingpad { ptr, i32 }
          cleanup
  tail call void @llvm.objc.release(ptr %k) #1, !clang.imprecise_release !0
  resume { ptr, i32 } %3
}

; The second retainRV/autoreleaseRV pair can be removed since the call to
; @returner is a tail call.

; CHECK-LABEL: define ptr @test30(
; CHECK: %[[V0:.*]] = call ptr @returner()
; CHECK-NEXT: call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %[[V0]])
; CHECK-NEXT: call ptr @llvm.objc.autoreleaseReturnValue(ptr %[[V0]])
; CHECK-NEXT: ret ptr %[[V0]]
; CHECK: %[[V3:.*]] = tail call ptr @returner()
; CHECK-NEXT: ret ptr %[[V3]]

define ptr @test30(i1 %cond) {
  br i1 %cond, label %bb0, label %bb1
bb0:
  %v0 = call ptr @returner()
  %v1 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %v0)
  %v2 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %v0)
  ret ptr %v0
bb1:
  %v3 = tail call ptr @returner()
  %v4 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %v3)
  %v5 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %v3)
  ret ptr %v3
}

; Remove operand bundle "clang.arc.attachedcall" and the autoreleaseRV call if the call
; is a tail call.

; CHECK-LABEL: define ptr @test31(
; CHECK-NEXT: %[[CALL:.*]] = tail call ptr @returner()
; CHECK-NEXT: ret ptr %[[CALL]]

define ptr @test31() {
  %call = tail call ptr @returner() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  call void (...) @llvm.objc.clang.arc.noop.use(ptr %call)
  %1 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %call)
  ret ptr %1
}

; CHECK-LABEL: define ptr @test32(
; CHECK: %[[CALL:.*]] = call ptr @returner() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK: call void (...) @llvm.objc.clang.arc.noop.use(ptr %[[CALL]])
; CHECK: call ptr @llvm.objc.autoreleaseReturnValue(ptr %[[CALL]])

define ptr @test32() {
  %call = call ptr @returner() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  call void (...) @llvm.objc.clang.arc.noop.use(ptr %call)
  %1 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %call)
  ret ptr %1
}

!0 = !{}

; CHECK: attributes [[NUW]] = { nounwind }
