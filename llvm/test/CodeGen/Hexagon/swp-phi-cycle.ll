; RUN: llc -mtriple=hexagon-unknown-linux-gnu -enable-pipeliner -debug-only=pipeliner < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: Cannot pipeline loop due to PHI cycle

define void @phi_cycle_loop(i32 %a, i32 %b) {
entry:
  br label %loop

loop:
  %1 = phi i32 [ %a, %entry ], [ %3, %loop ]
  %2 = phi i32 [ %a, %entry ], [ %1, %loop ]
  %3 = phi i32 [ %b, %entry ], [ %2, %loop ]

  ; Prevent PHI elimination by using all values
  %add1 = add i32 %1, %2
  %add2 = add i32 %add1, %3
  %cmp = icmp slt i32 %add2, 100
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
