; RUN: llc -march=arc < %s | FileCheck %s

; CHECK-LABEL: brcc1
; CHECK: brne %r0, %r1
define i32 @brcc1(i32 %a, i32 %b) nounwind {
entry:
  %wb = icmp eq i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

; CHECK-LABEL: brcc2
; CHECK: breq %r0, %r1
define i32 @brcc2(i32 %a, i32 %b) nounwind {
entry:
  %wb = icmp ne i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}


