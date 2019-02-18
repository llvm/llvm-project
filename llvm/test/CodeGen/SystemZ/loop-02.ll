; Test BRCTH.

; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu -mcpu=z196 \
; RUN:   -no-integrated-as | FileCheck %s

; Test a loop that should be converted into dbr form and then use BRCTH.
define void @f2(i32 *%src, i32 *%dest) {
; CHECK-LABEL: f2:
; CHECK: blah [[REG:%r[0-5]]]
; CHECK: [[LABEL:\.[^:]*]]:{{.*}} %loop
; CHECK: brcth [[REG]], [[LABEL]]
; CHECK: br %r14
entry:
  ; Force upper bound into a high register in order to encourage the
  ; register allocator to use a high register for the count variable.
  %top = call i32 asm sideeffect "blah $0", "=h"()
  br label %loop

loop:
  %count = phi i32 [ 0, %entry ], [ %next, %loop.next ]
  %next = add i32 %count, 1
  %val = load volatile i32, i32 *%src
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %loop.next, label %loop.store

loop.store:
  %add = add i32 %val, 1
  store volatile i32 %add, i32 *%dest
  br label %loop.next

loop.next:
  %cont = icmp ne i32 %next, %top
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}

