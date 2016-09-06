; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

define i32 @f1(i32 %x) {
       %y = add i32 %z, 1
       %z = add i32 %x, 1
       ret i32 %y
; CHECK: Instruction does not dominate all uses!
; CHECK-NEXT:  %z = add i32 %x, 1
; CHECK-NEXT:  %y = add i32 %z, 1
}

declare i32 @g()
define void @f2(i32 %x) personality i32 ()* @g {
bb0:
  %y1 = invoke i32 @g() to label %bb1 unwind label %bb2
bb1:
  ret void
bb2:
  %y2 = phi i32 [%y1, %bb0]
  %y3 = landingpad i32
          cleanup
  ret void
; CHECK: Instruction does not dominate all uses!
; CHECK-NEXT:  %y1 = invoke i32 @g()
; CHECK-NEXT:        to label %bb1 unwind label %bb2
; CHECK-NEXT:  %y2 = phi i32 [ %y1, %bb0 ]
}

define void @f3(i32 %x) personality i32 ()* @g {
bb0:
  %y1 = invoke i32 @g() to label %bb1 unwind label %bb2
bb1:
  ret void
bb2:
  %y2 = landingpad i32
          cleanup
  br label %bb3
bb3:
  %y3 = phi i32 [%y1, %bb2]
  ret void
; CHECK: Instruction does not dominate all uses!
; CHECK-NEXT:  %y1 = invoke i32 @g()
; CHECK-NEXT:          to label %bb1 unwind label %bb2
; CHECK-NEXT:  %y3 = phi i32 [ %y1, %bb2 ]
}

define void @f4(i32 %x) {
bb0:
  br label %bb1
bb1:
  %y3 = phi i32 [%y1, %bb0]
  %y1 = add i32 %x, 1
  ret void
; CHECK: Instruction does not dominate all uses!
; CHECK-NEXT:  %y1 = add i32 %x, 1
; CHECK-NEXT:  %y3 = phi i32 [ %y1, %bb0 ]
}

define void @f5() {
entry:
  br label %next

next:
  %y = phi i32 [ 0, %entry ]
  %x = phi i32 [ %y, %entry ]
  ret void
; CHECK: Instruction does not dominate all uses!
; CHECK-NEXT:  %y = phi i32 [ 0, %entry ]
; CHECK-NEXT:  %x = phi i32 [ %y, %entry ]
}
