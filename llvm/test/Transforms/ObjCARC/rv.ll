; RUN: opt -objc-arc -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare i8* @llvm.objc.retain(i8*)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare void @llvm.objc.release(i8*)
declare i8* @llvm.objc.autorelease(i8*)
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare i8* @llvm.objc.retainAutoreleaseReturnValue(i8*)
declare void @llvm.objc.autoreleasePoolPop(i8*)
declare void @llvm.objc.autoreleasePoolPush()
declare i8* @llvm.objc.retainBlock(i8*)
declare void @llvm.objc.clang.arc.noop.use(...)

declare i8* @objc_retainedObject(i8*)
declare i8* @objc_unretainedObject(i8*)
declare i8* @objc_unretainedPointer(i8*)

declare void @use_pointer(i8*)
declare void @callee()
declare void @callee_fnptr(void ()*)
declare void @invokee()
declare i8* @returner()
declare i8* @returner1(i8*)
declare i32 @__gxx_personality_v0(...)

; Test that retain+release elimination is suppressed when the
; retain is an objc_retainAutoreleasedReturnValue, since it's
; better to do the RV optimization.

; CHECK-LABEL:      define void @test0(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x = call i8* @returner
; CHECK-NEXT:   %0 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %x) [[NUW:#[0-9]+]]
; CHECK: t:
; CHECK-NOT: @llvm.objc.
; CHECK: return:
; CHECK-NEXT: call void @llvm.objc.release(i8* %x)
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test0(i1 %p) nounwind {
entry:
  %x = call i8* @returner()
  %0 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %x)
  br i1 %p, label %t, label %return

t:
  call void @use_pointer(i8* %x)
  store i8 0, i8* %x
  br label %return

return:
  call void @llvm.objc.release(i8* %x) nounwind
  ret void
}

; Delete no-ops.

; CHECK-LABEL: define void @test2(
; CHECK-NOT: @llvm.objc.
; CHECK: }
define void @test2() {
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* null)
  call i8* @llvm.objc.autoreleaseReturnValue(i8* null)
  ; call i8* @llvm.objc.retainAutoreleaseReturnValue(i8* null) ; TODO
  %bitcast = bitcast i32* null to i8*
  %rb = call i8* @llvm.objc.retainBlock(i8* %bitcast)
  call void @use_pointer(i8* %rb)
  %rb2 = call i8* @llvm.objc.retainBlock(i8* undef)
  call void @use_pointer(i8* %rb2)
  ret void
}

; Delete a redundant retainRV,autoreleaseRV when forwaring a call result
; directly to a return value.

; CHECK-LABEL: define i8* @test3(
; CHECK: call i8* @returner()
; CHECK-NEXT: ret i8* %call
define i8* @test3() {
entry:
  %call = tail call i8* @returner()
  %0 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call) nounwind
  %1 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %0) nounwind
  ret i8* %1
}

; Delete a redundant retain,autoreleaseRV when forwaring a call result
; directly to a return value.

; CHECK-LABEL: define i8* @test4(
; CHECK: call i8* @returner()
; CHECK-NEXT: ret i8* %call
define i8* @test4() {
entry:
  %call = call i8* @returner()
  %0 = call i8* @llvm.objc.retain(i8* %call) nounwind
  %1 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %0) nounwind
  ret i8* %1
}

; Delete a redundant fused retain+autoreleaseRV when forwaring a call result
; directly to a return value.

; TODO
; HECK: define i8* @test5
; HECK: call i8* @returner()
; HECK-NEXT: ret i8* %call
;define i8* @test5() {
;entry:
;  %call = call i8* @returner()
;  %0 = call i8* @llvm.objc.retainAutoreleaseReturnValue(i8* %call) nounwind
;  ret i8* %0
;}

; Don't eliminate objc_retainAutoreleasedReturnValue by merging it into
; an objc_autorelease.
; TODO? Merge objc_retainAutoreleasedReturnValue and objc_autorelease into
; objc_retainAutoreleasedReturnValueAutorelease and merge
; objc_retainAutoreleasedReturnValue and objc_autoreleaseReturnValue
; into objc_retainAutoreleasedReturnValueAutoreleaseReturnValue?
; Those entrypoints don't exist yet though.

; CHECK-LABEL: define i8* @test7(
; CHECK: call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %p)
; CHECK: %t = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
define i8* @test7() {
  %p = call i8* @returner()
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %p)
  %t = call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
  call void @use_pointer(i8* %p)
  ret i8* %t
}

; CHECK-LABEL: define i8* @test7b(
; CHECK: call i8* @llvm.objc.retain(i8* %p)
; CHECK: %t = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
define i8* @test7b() {
  %p = call i8* @returner()
  call void @use_pointer(i8* %p)
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %p)
  %t = call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
  ret i8* %p
}

; Don't apply the RV optimization to autorelease if there's no retain.

; CHECK: define i8* @test9(i8* %p)
; CHECK: call i8* @llvm.objc.autorelease(i8* %p)
define i8* @test9(i8* %p) {
  call i8* @llvm.objc.autorelease(i8* %p)
  ret i8* %p
}

; Do not apply the RV optimization.

; CHECK: define i8* @test10(i8* %p)
; CHECK: tail call i8* @llvm.objc.retain(i8* %p) [[NUW]]
; CHECK: call i8* @llvm.objc.autorelease(i8* %p) [[NUW]]
; CHECK-NEXT: ret i8* %p
define i8* @test10(i8* %p) {
  %1 = call i8* @llvm.objc.retain(i8* %p)
  %2 = call i8* @llvm.objc.autorelease(i8* %p)
  ret i8* %p
}

; Don't do the autoreleaseRV optimization because @use_pointer
; could undo the retain.

; CHECK: define i8* @test11(i8* %p)
; CHECK: tail call i8* @llvm.objc.retain(i8* %p)
; CHECK-NEXT: call void @use_pointer(i8* %p)
; CHECK: call i8* @llvm.objc.autorelease(i8* %p)
; CHECK-NEXT: ret i8* %p
define i8* @test11(i8* %p) {
  %1 = call i8* @llvm.objc.retain(i8* %p)
  call void @use_pointer(i8* %p)
  %2 = call i8* @llvm.objc.autorelease(i8* %p)
  ret i8* %p
}

; Don't spoil the RV optimization.

; CHECK: define i8* @test12(i8* %p)
; CHECK: tail call i8* @llvm.objc.retain(i8* %p)
; CHECK: call void @use_pointer(i8* %p)
; CHECK: tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
; CHECK: ret i8* %p
define i8* @test12(i8* %p) {
  %1 = call i8* @llvm.objc.retain(i8* %p)
  call void @use_pointer(i8* %p)
  %2 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
  ret i8* %p
}

; Don't zap the objc_retainAutoreleasedReturnValue.

; CHECK-LABEL: define i8* @test13(
; CHECK: tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %p)
; CHECK: call i8* @llvm.objc.autorelease(i8* %p)
; CHECK: ret i8* %p
define i8* @test13() {
  %p = call i8* @returner()
  %1 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %p)
  call void @callee()
  %2 = call i8* @llvm.objc.autorelease(i8* %p)
  ret i8* %p
}

; Convert objc_retainAutoreleasedReturnValue to objc_retain if its
; argument is not a return value.

; CHECK-LABEL: define void @test14(
; CHECK-NEXT: tail call i8* @llvm.objc.retain(i8* %p) [[NUW]]
; CHECK-NEXT: ret void
define void @test14(i8* %p) {
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %p)
  ret void
}

; Don't convert objc_retainAutoreleasedReturnValue to objc_retain if its
; argument is a return value.

; CHECK-LABEL: define void @test15(
; CHECK-NEXT: %y = call i8* @returner()
; CHECK-NEXT: tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %y) [[NUW]]
; CHECK-NEXT: ret void
define void @test15() {
  %y = call i8* @returner()
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %y)
  ret void
}

; Delete autoreleaseRV+retainRV pairs.

; CHECK: define i8* @test19(i8* %p) {
; CHECK-NEXT: ret i8* %p
define i8* @test19(i8* %p) {
  call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %p)
  ret i8* %p
}

; Delete autoreleaseRV+retainRV pairs when they have equivalent PHIs as inputs

; CHECK: define i8* @test19phi(i8* %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT: br label %test19bb
; CHECK: test19bb:
; CHECK-NEXT: ret i8* %p
define i8* @test19phi(i8* %p) {
entry:
  br label %test19bb
test19bb:
  %phi1 = phi i8* [ %p, %entry ]
  %phi2 = phi i8* [ %p, %entry ]
  call i8* @llvm.objc.autoreleaseReturnValue(i8* %phi1)
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %phi2)
  ret i8* %p
}

; Like test19 but with plain autorelease.

; CHECK: define i8* @test20(i8* %p) {
; CHECK-NEXT: call i8* @llvm.objc.autorelease(i8* %p)
; CHECK-NEXT: call i8* @llvm.objc.retain(i8* %p)
; CHECK-NEXT: ret i8* %p
define i8* @test20(i8* %p) {
  call i8* @llvm.objc.autorelease(i8* %p)
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %p)
  ret i8* %p
}

; Like test19 but with plain retain.

; CHECK: define i8* @test21(i8* %p) {
; CHECK-NEXT: call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
; CHECK-NEXT: call i8* @llvm.objc.retain(i8* %p)
; CHECK-NEXT: ret i8* %p
define i8* @test21(i8* %p) {
  call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
  call i8* @llvm.objc.retain(i8* %p)
  ret i8* %p
}

; Like test19 but with plain retain and autorelease.

; CHECK: define i8* @test22(i8* %p) {
; CHECK-NEXT: call i8* @llvm.objc.autorelease(i8* %p)
; CHECK-NEXT: call i8* @llvm.objc.retain(i8* %p)
; CHECK-NEXT: ret i8* %p
define i8* @test22(i8* %p) {
  call i8* @llvm.objc.autorelease(i8* %p)
  call i8* @llvm.objc.retain(i8* %p)
  ret i8* %p
}

; Convert autoreleaseRV to autorelease.

; CHECK-LABEL: define void @test23(
; CHECK: call i8* @llvm.objc.autorelease(i8* %p) [[NUW]]
define void @test23(i8* %p) {
  store i8 0, i8* %p
  call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
  ret void
}

; Don't convert autoreleaseRV to autorelease if the result is returned,
; even through a bitcast.

; CHECK-LABEL: define {}* @test24(
; CHECK: tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
define {}* @test24(i8* %p) {
  %t = call i8* @llvm.objc.autoreleaseReturnValue(i8* %p)
  %s = bitcast i8* %p to {}*
  ret {}* %s
}

declare i8* @first_test25();
declare i8* @second_test25(i8*);
declare void @somecall_test25();

; ARC optimizer used to move the last release between the call to second_test25
; and the call to objc_retainAutoreleasedReturnValue, causing %second to be
; released prematurely when %first and %second were pointing to the same object.

; CHECK-LABEL: define void @test25(
; CHECK: %[[CALL1:.*]] = call i8* @second_test25(
; CHECK-NEXT: tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[CALL1]])

define void @test25() {
  %first = call i8* @first_test25()
  %v0 = call i8* @llvm.objc.retain(i8* %first)
  call void @somecall_test25()
  %second = call i8* @second_test25(i8* %first)
  %call2 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %second)
  call void @llvm.objc.release(i8* %second), !clang.imprecise_release !0
  call void @llvm.objc.release(i8* %first), !clang.imprecise_release !0
  ret void
}

; Check that ObjCARCOpt::OptimizeReturns removes the redundant calls even when
; they are not in the same basic block. This code used to cause an assertion
; failure.

; CHECK-LABEL: define i8* @test26()
; CHECK: call i8* @returner()
; CHECK-NOT:  call
define i8* @test26() {
bb0:
  %v0 = call i8* @returner()
  %v1 = tail call i8* @llvm.objc.retain(i8* %v0)
  br label %bb1
bb1:
  %v2 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %v1)
  br label %bb2
bb2:
  ret i8* %v2
}

declare i32* @func27(i32);

; Check that ObjCARCOpt::OptimizeAutoreleaseRVCall doesn't turn a call to
; @llvm.objc.autoreleaseReturnValue into a call to @llvm.objc.autorelease when a return
; instruction uses a value equivalent to @llvm.objc.autoreleaseReturnValue's operand.
; In the code below, %phival and %retval are considered equivalent.

; CHECK-LABEL: define i32* @test27(
; CHECK: %[[PHIVAL:.*]] = phi i8* [ %{{.*}}, %bb1 ], [ %{{.*}}, %bb2 ]
; CHECK: %[[RETVAL:.*]] = phi i32* [ %{{.*}}, %bb1 ], [ %{{.*}}, %bb2 ]
; CHECK: tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %[[PHIVAL]])
; CHECK: ret i32* %[[RETVAL]]

define i32* @test27(i1 %cond) {
entry:
  br i1 %cond, label %bb1, label %bb2
bb1:
  %v0 = call i32* @func27(i32 1)
  %v1 = bitcast i32* %v0 to i8*
  br label %bb3
bb2:
  %v2 = call i32* @func27(i32 2)
  %v3 = bitcast i32* %v2 to i8*
  br label %bb3
bb3:
  %phival = phi i8* [ %v1, %bb1 ], [ %v3, %bb2 ]
  %retval = phi i32* [ %v0, %bb1 ], [ %v2, %bb2 ]
  %v4 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %phival)
  ret i32* %retval
}

; Don't eliminate the retainRV/autoreleaseRV pair if the call isn't a tail call.

; CHECK-LABEL: define i8* @test28(
; CHECK: call i8* @returner()
; CHECK: call i8* @llvm.objc.retainAutoreleasedReturnValue(
; CHECK: call i8* @llvm.objc.autoreleaseReturnValue(
define i8* @test28() {
entry:
  %call = call i8* @returner()
  %0 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call) nounwind
  %1 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %0) nounwind
  ret i8* %1
}

; CHECK-LABEL: define i8* @test29(
; CHECK: call i8* @llvm.objc.retainAutoreleasedReturnValue(
; CHECK: call i8* @llvm.objc.autoreleaseReturnValue(

define i8* @test29(i8* %k) local_unnamed_addr personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %0 = tail call i8* @llvm.objc.retain(i8* %k)
  %call = invoke i8* @returner1(i8* %k)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  %1 = bitcast i8* %call to i8*
  %2 = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %1)
  tail call void @llvm.objc.release(i8* %k), !clang.imprecise_release !0
  %3 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %1)
  ret i8* %call

lpad:
  %4 = landingpad { i8*, i32 }
          cleanup
  tail call void @llvm.objc.release(i8* %k) #1, !clang.imprecise_release !0
  resume { i8*, i32 } %4
}

; The second retainRV/autoreleaseRV pair can be removed since the call to
; @returner is a tail call.

; CHECK-LABEL: define i8* @test30(
; CHECK: %[[V0:.*]] = call i8* @returner()
; CHECK-NEXT: call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[V0]])
; CHECK-NEXT: call i8* @llvm.objc.autoreleaseReturnValue(i8* %[[V0]])
; CHECK-NEXT: ret i8* %[[V0]]
; CHECK: %[[V3:.*]] = tail call i8* @returner()
; CHECK-NEXT: ret i8* %[[V3]]

define i8* @test30(i1 %cond) {
  br i1 %cond, label %bb0, label %bb1
bb0:
  %v0 = call i8* @returner()
  %v1 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %v0)
  %v2 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %v0)
  ret i8* %v0
bb1:
  %v3 = tail call i8* @returner()
  %v4 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %v3)
  %v5 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %v3)
  ret i8* %v3
}

; Remove operand bundle "clang.arc.attachedcall" and the autoreleaseRV call if the call
; is a tail call.

; CHECK-LABEL: define i8* @test31(
; CHECK-NEXT: %[[CALL:.*]] = tail call i8* @returner()
; CHECK-NEXT: ret i8* %[[CALL]]

define i8* @test31() {
  %call = tail call i8* @returner() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  call void (...) @llvm.objc.clang.arc.noop.use(i8* %call)
  %1 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %call)
  ret i8* %1
}

; CHECK-LABEL: define i8* @test32(
; CHECK: %[[CALL:.*]] = call i8* @returner() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK: call void (...) @llvm.objc.clang.arc.noop.use(i8* %[[CALL]])
; CHECK: call i8* @llvm.objc.autoreleaseReturnValue(i8* %[[CALL]])

define i8* @test32() {
  %call = call i8* @returner() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  call void (...) @llvm.objc.clang.arc.noop.use(i8* %call)
  %1 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %call)
  ret i8* %1
}

!0 = !{}

; CHECK: attributes [[NUW]] = { nounwind }
