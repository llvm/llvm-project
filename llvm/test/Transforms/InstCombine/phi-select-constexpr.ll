; RUN: opt < %s -S -instcombine | FileCheck %s
@A = extern_weak global i32, align 4
@B = extern_weak global i32, align 4

define i32 @foo(i1 %which) {
entry:
  br i1 %which, label %final, label %delay

delay:
  br label %final

; CHECK-LABEL: final:
; CHECK: phi i32 [ 1, %entry ], [ select (i1 icmp eq (i32* @A, i32* @B), i32 2, i32 1), %delay ]
final:
  %use2 = phi i1 [ false, %entry ], [ icmp eq (i32* @A, i32* @B), %delay ]
  %value = select i1 %use2, i32 2, i32 1
  ret i32 %value
}

