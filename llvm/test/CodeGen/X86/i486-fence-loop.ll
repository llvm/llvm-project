; RUN: llc -march=x86 -mcpu=i486 -o - %s | FileCheck %s

; Main test here was that ISelDAG could cope with a MachineNode in the chain
; from the first load to the "X86ISD::SUB". Previously it thought that meant no
; cycle could be formed so it tried to use "sub (%eax), [[RHS]]".

define void @gst_atomic_queue_push(i32* %addr) {
; CHECK-LABEL: gst_atomic_queue_push:
; CHECK: movl (%eax), [[LHS:%e[a-z]+]]
; CHECK: lock orl
; CHECK: movl (%eax), [[RHS:%e[a-z]+]]
; CHECK: cmpl [[LHS]], [[RHS]]

entry:
  br label %while.body

while.body:
  %0 = load volatile i32, i32* %addr, align 4
  fence seq_cst
  %1 = load volatile i32, i32* %addr, align 4
  %cmp = icmp sgt i32 %1, %0
  br i1 %cmp, label %while.body, label %if.then

if.then:
  ret void
}