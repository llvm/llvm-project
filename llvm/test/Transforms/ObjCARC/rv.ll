; RUN: opt -objc-arc -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare i8* @objc_retain(i8*)
declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare void @objc_release(i8*)
declare i8* @objc_autorelease(i8*)
declare i8* @objc_autoreleaseReturnValue(i8*)
declare i8* @objc_retainAutoreleaseReturnValue(i8*)
declare void @objc_autoreleasePoolPop(i8*)
declare void @objc_autoreleasePoolPush()
declare i8* @objc_retainBlock(i8*)

declare i8* @objc_retainedObject(i8*)
declare i8* @objc_unretainedObject(i8*)
declare i8* @objc_unretainedPointer(i8*)

declare void @use_pointer(i8*)
declare void @callee()
declare void @callee_fnptr(void ()*)
declare void @invokee()
declare i8* @returner()

; Test that retain+release elimination is suppressed when the
; retain is an objc_retainAutoreleasedReturnValue, since it's
; better to do the RV optimization.

; CHECK:      define void @test0(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x = call i8* @returner
; CHECK-NEXT:   %0 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %x) nounwind
; CHECK: t:
; CHECK-NOT: @objc_
; CHECK: return:
; CHECK-NEXT: call void @objc_release(i8* %x)
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test0(i1 %p) nounwind {
entry:
  %x = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %x)
  br i1 %p, label %t, label %return

t:
  call void @use_pointer(i8* %x)
  store i8 0, i8* %x
  br label %return

return:
  call void @objc_release(i8* %x) nounwind
  ret void
}

; Delete no-ops.

; CHECK: define void @test2
; CHECK-NOT: @objc_
; CHECK: }
define void @test2() {
  call i8* @objc_retainAutoreleasedReturnValue(i8* null)
  call i8* @objc_autoreleaseReturnValue(i8* null)
  ; call i8* @objc_retainAutoreleaseReturnValue(i8* null) ; TODO
  ret void
}

; Delete a redundant retainRV,autoreleaseRV when forwaring a call result
; directly to a return value.

; CHECK: define i8* @test3
; CHECK: call i8* @returner()
; CHECK-NEXT: ret i8* %call
define i8* @test3() {
entry:
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %1 = call i8* @objc_autoreleaseReturnValue(i8* %0) nounwind
  ret i8* %1
}

; Delete a redundant retain,autoreleaseRV when forwaring a call result
; directly to a return value.

; CHECK: define i8* @test4
; CHECK: call i8* @returner()
; CHECK-NEXT: ret i8* %call
define i8* @test4() {
entry:
  %call = call i8* @returner()
  %0 = call i8* @objc_retain(i8* %call) nounwind
  %1 = call i8* @objc_autoreleaseReturnValue(i8* %0) nounwind
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
;  %0 = call i8* @objc_retainAutoreleaseReturnValue(i8* %call) nounwind
;  ret i8* %0
;}

; Don't eliminate objc_retainAutoreleasedReturnValue by merging it into
; an objc_autorelease.
; TODO? Merge objc_retainAutoreleasedReturnValue and objc_autorelease into
; objc_retainAutoreleasedReturnValueAutorelease and merge
; objc_retainAutoreleasedReturnValue and objc_autoreleaseReturnValue
; into objc_retainAutoreleasedReturnValueAutoreleaseReturnValue?
; Those entrypoints don't exist yet though.

; CHECK: define i8* @test7(
; CHECK: call i8* @objc_retainAutoreleasedReturnValue(i8* %p)
; CHECK: %t = tail call i8* @objc_autoreleaseReturnValue(i8* %p)
define i8* @test7() {
  %p = call i8* @returner()
  call i8* @objc_retainAutoreleasedReturnValue(i8* %p)
  %t = call i8* @objc_autoreleaseReturnValue(i8* %p)
  call void @use_pointer(i8* %t)
  ret i8* %t
}

; CHECK: define i8* @test7b(
; CHECK: call i8* @objc_retain(i8* %p)
; CHECK: %t = tail call i8* @objc_autoreleaseReturnValue(i8* %p)
define i8* @test7b() {
  %p = call i8* @returner()
  call void @use_pointer(i8* %p)
  call i8* @objc_retainAutoreleasedReturnValue(i8* %p)
  %t = call i8* @objc_autoreleaseReturnValue(i8* %p)
  ret i8* %t
}

; Turn objc_retain into objc_retainAutoreleasedReturnValue if its operand
; is a return value.

; CHECK: define void @test8()
; CHECK: tail call i8* @objc_retainAutoreleasedReturnValue(i8* %p)
define void @test8() {
  %p = call i8* @returner()
  call i8* @objc_retain(i8* %p)
  ret void
}

; Don't apply the RV optimization to autorelease if there's no retain.

; CHECK: define i8* @test9(i8* %p)
; CHECK: tail call i8* @objc_autorelease(i8* %p)
define i8* @test9(i8* %p) {
  call i8* @objc_autorelease(i8* %p)
  ret i8* %p
}

; Apply the RV optimization.

; CHECK: define i8* @test10(i8* %p)
; CHECK: tail call i8* @objc_retain(i8* %p) nounwind
; CHECK: tail call i8* @objc_autoreleaseReturnValue(i8* %p) nounwind
; CHECK-NEXT: ret i8* %p
define i8* @test10(i8* %p) {
  %1 = call i8* @objc_retain(i8* %p)
  %2 = call i8* @objc_autorelease(i8* %p)
  ret i8* %p
}

; Don't do the autoreleaseRV optimization because @use_pointer
; could undo the retain.

; CHECK: define i8* @test11(i8* %p)
; CHECK: tail call i8* @objc_retain(i8* %p)
; CHECK-NEXT: call void @use_pointer(i8* %p)
; CHECK: tail call i8* @objc_autorelease(i8* %p)
; CHECK-NEXT: ret i8* %p
define i8* @test11(i8* %p) {
  %1 = call i8* @objc_retain(i8* %p)
  call void @use_pointer(i8* %p)
  %2 = call i8* @objc_autorelease(i8* %p)
  ret i8* %p
}

; Don't spoil the RV optimization.

; CHECK: define i8* @test12(i8* %p)
; CHECK: tail call i8* @objc_retain(i8* %p)
; CHECK: call void @use_pointer(i8* %p)
; CHECK: tail call i8* @objc_autoreleaseReturnValue(i8* %p)
; CHECK: ret i8* %p
define i8* @test12(i8* %p) {
  %1 = call i8* @objc_retain(i8* %p)
  call void @use_pointer(i8* %p)
  %2 = call i8* @objc_autoreleaseReturnValue(i8* %p)
  ret i8* %p
}

; Don't zap the objc_retainAutoreleasedReturnValue.

; CHECK: define i8* @test13(
; CHECK: tail call i8* @objc_retainAutoreleasedReturnValue(i8* %p)
; CHECK: tail call i8* @objc_autorelease(i8* %p)
; CHECK: ret i8* %p
define i8* @test13() {
  %p = call i8* @returner()
  %1 = call i8* @objc_retainAutoreleasedReturnValue(i8* %p)
  call void @callee()
  %2 = call i8* @objc_autorelease(i8* %p)
  ret i8* %p
}

; Convert objc_retainAutoreleasedReturnValue to objc_retain if its
; argument is not a return value.

; CHECK: define void @test14(
; CHECK-NEXT: tail call i8* @objc_retain(i8* %p) nounwind
; CHECK-NEXT: ret void
define void @test14(i8* %p) {
  call i8* @objc_retainAutoreleasedReturnValue(i8* %p)
  ret void
}

; Don't convert objc_retainAutoreleasedReturnValue to objc_retain if its
; argument is a return value.

; CHECK: define void @test15(
; CHECK-NEXT: %y = call i8* @returner()
; CHECK-NEXT: tail call i8* @objc_retainAutoreleasedReturnValue(i8* %y) nounwind
; CHECK-NEXT: ret void
define void @test15() {
  %y = call i8* @returner()
  call i8* @objc_retainAutoreleasedReturnValue(i8* %y)
  ret void
}

; Convert objc_retain to objc_retainAutoreleasedReturnValue if its
; argument is a return value.

; CHECK: define void @test16(
; CHECK-NEXT: %y = call i8* @returner()
; CHECK-NEXT: tail call i8* @objc_retainAutoreleasedReturnValue(i8* %y) nounwind
; CHECK-NEXT: ret void
define void @test16() {
  %y = call i8* @returner()
  call i8* @objc_retain(i8* %y)
  ret void
}

; Don't convert objc_retain to objc_retainAutoreleasedReturnValue if its
; argument is not a return value.

; CHECK: define void @test17(
; CHECK-NEXT: tail call i8* @objc_retain(i8* %y) nounwind
; CHECK-NEXT: ret void
define void @test17(i8* %y) {
  call i8* @objc_retain(i8* %y)
  ret void
}

; Don't Convert objc_retain to objc_retainAutoreleasedReturnValue if it
; isn't next to the call providing its return value.

; CHECK: define void @test18(
; CHECK-NEXT: %y = call i8* @returner()
; CHECK-NEXT: call void @callee()
; CHECK-NEXT: tail call i8* @objc_retain(i8* %y) nounwind
; CHECK-NEXT: ret void
define void @test18() {
  %y = call i8* @returner()
  call void @callee()
  call i8* @objc_retain(i8* %y)
  ret void
}

; Delete autoreleaseRV+retainRV pairs.

; CHECK: define i8* @test19(i8* %p) {
; CHECK-NEXT: ret i8* %p
define i8* @test19(i8* %p) {
  call i8* @objc_autoreleaseReturnValue(i8* %p)
  call i8* @objc_retainAutoreleasedReturnValue(i8* %p)
  ret i8* %p
}

; Like test19 but with plain autorelease.

; CHECK: define i8* @test20(i8* %p) {
; CHECK-NEXT: call i8* @objc_autorelease(i8* %p)
; CHECK-NEXT: call i8* @objc_retain(i8* %p)
; CHECK-NEXT: ret i8* %p
define i8* @test20(i8* %p) {
  call i8* @objc_autorelease(i8* %p)
  call i8* @objc_retainAutoreleasedReturnValue(i8* %p)
  ret i8* %p
}

; Like test19 but with plain retain.

; CHECK: define i8* @test21(i8* %p) {
; CHECK-NEXT: call i8* @objc_autoreleaseReturnValue(i8* %p)
; CHECK-NEXT: call i8* @objc_retain(i8* %p)
; CHECK-NEXT: ret i8* %p
define i8* @test21(i8* %p) {
  call i8* @objc_autoreleaseReturnValue(i8* %p)
  call i8* @objc_retain(i8* %p)
  ret i8* %p
}

; Like test19 but with plain retain and autorelease.

; CHECK: define i8* @test22(i8* %p) {
; CHECK-NEXT: call i8* @objc_autorelease(i8* %p)
; CHECK-NEXT: call i8* @objc_retain(i8* %p)
; CHECK-NEXT: ret i8* %p
define i8* @test22(i8* %p) {
  call i8* @objc_autorelease(i8* %p)
  call i8* @objc_retain(i8* %p)
  ret i8* %p
}

; Convert autoreleaseRV to autorelease.

; CHECK: define void @test23(
; CHECK: tail call i8* @objc_autorelease(i8* %p) nounwind
define void @test23(i8* %p) {
  store i8 0, i8* %p
  call i8* @objc_autoreleaseReturnValue(i8* %p)
  ret void
}

; Don't convert autoreleaseRV to autorelease if the result is returned,
; even through a bitcast.

; CHECK: define {}* @test24(
; CHECK: tail call i8* @objc_autoreleaseReturnValue(i8* %p)
define {}* @test24(i8* %p) {
  %t = call i8* @objc_autoreleaseReturnValue(i8* %p)
  %s = bitcast i8* %p to {}*
  ret {}* %s
}
