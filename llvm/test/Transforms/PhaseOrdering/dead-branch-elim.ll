; RUN: opt -O2 -S %s | FileCheck %s
; RUN: opt -O2 -enable-dead-branch-elim=false -S %s | FileCheck %s --check-prefix=DISABLED

; With dead-branch-elim in the pipeline the circular-dependency branch is
; removed before SimplifyCFG can speculate it into a select, and -O2 then
; collapses the whole loop:
;   int a = 0, b = 0, limit = 100;
;   while (a < limit) {
;     if (b == limit) limit += 1;
;     a++; b++;
;   }
;   return b;   // == 100

; CHECK-LABEL: define {{.*}}i32 @run()
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret i32 100

; DISABLED-LABEL: define {{.*}}i32 @run()
; DISABLED: icmp eq
define i32 @run() {
entry:
  br label %header

header:
  %limit = phi i32 [ 100, %entry ], [ %limit.next, %latch ]
  %b = phi i32 [ 0, %entry ], [ %b.next, %latch ]
  %a = phi i32 [ 0, %entry ], [ %a.next, %latch ]
  %guard = icmp slt i32 %a, %limit
  br i1 %guard, label %body, label %exit

body:
  %cmp.inner = icmp eq i32 %b, %limit
  br i1 %cmp.inner, label %if.then, label %latch

if.then:
  %limit.inc = add nsw i32 %limit, 1
  br label %latch

latch:
  %limit.next = phi i32 [ %limit.inc, %if.then ], [ %limit, %body ]
  %a.next = add nsw i32 %a, 1
  %b.next = add nsw i32 %b, 1
  br label %header

exit:
  ret i32 %b
}
