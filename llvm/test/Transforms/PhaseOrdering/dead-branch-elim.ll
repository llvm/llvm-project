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
;
; The input is the frontend's alloca form (clang -O0 without optnone): the
; early SimplifyCFG cannot speculate stores, SROA then promotes the allocas,
; and dead-branch-elim must catch the branch before the GlobalCleanup
; SimplifyCFG turns it into a data dependency.

; CHECK-LABEL: define {{.*}}i32 @run()
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret i32 100

; DISABLED-LABEL: define {{.*}}i32 @run()
; DISABLED: icmp eq
define i32 @run() {
entry:
  %a = alloca i32
  %b = alloca i32
  %limit = alloca i32
  store i32 0, ptr %a
  store i32 0, ptr %b
  store i32 100, ptr %limit
  br label %while.cond

while.cond:
  %a.val = load i32, ptr %a
  %limit.val = load i32, ptr %limit
  %guard = icmp slt i32 %a.val, %limit.val
  br i1 %guard, label %while.body, label %while.end

while.body:
  %b.val = load i32, ptr %b
  %limit.val2 = load i32, ptr %limit
  %cmp = icmp eq i32 %b.val, %limit.val2
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %limit.val3 = load i32, ptr %limit
  %inc = add nsw i32 %limit.val3, 1
  store i32 %inc, ptr %limit
  br label %if.end

if.end:
  %a.val2 = load i32, ptr %a
  %a.inc = add nsw i32 %a.val2, 1
  store i32 %a.inc, ptr %a
  %b.val2 = load i32, ptr %b
  %b.inc = add nsw i32 %b.val2, 1
  store i32 %b.inc, ptr %b
  br label %while.cond

while.end:
  %ret = load i32, ptr %b
  ret i32 %ret
}
