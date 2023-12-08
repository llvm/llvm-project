; RUN-disabled: llc < %s -verify-machineinstrs -mtriple=x86_64-apple-macosx -pre-RA-sched=ilp -debug-only=pre-RA-sched \
; RUN-disabled:     2>&1 | FileCheck %s
; RUN: true
; REQUIRES: asserts
;
; rdar:13279013: pre-RA-sched should not check all interferences and
; repush them on the ready queue after scheduling each instruction.
;
; CHECK: *** List Scheduling
; CHECK: Interfering reg EFLAGS
; CHECK: Repushing
; CHECK: Repushing
; CHECK: Repushing
; CHECK-NOT: Repushing
; CHECK: *** Final schedule
define i32 @test(ptr %pin) #0 {
  %l0 = load i8, ptr %pin, align 1

  %g1a = getelementptr inbounds i8, ptr %pin, i64 1
  %l1a = load i8, ptr %g1a, align 1
  %z1a = zext i8 %l1a to i32
  %g1b = getelementptr inbounds i8, ptr %pin, i64 2
  %l1b = load i8, ptr %g1b, align 1
  %z1b = zext i8 %l1b to i32
  %c1 = icmp ne i8 %l0, 0
  %x1 = xor i32 %z1a, %z1b
  %s1 = select i1 %c1, i32 %z1a, i32 %x1

  %g2a = getelementptr inbounds i8, ptr %pin, i64 3
  %l2a = load i8, ptr %g2a, align 1
  %z2a = zext i8 %l2a to i32
  %g2b = getelementptr inbounds i8, ptr %pin, i64 4
  %l2b = load i8, ptr %g2b, align 1
  %z2b = zext i8 %l2b to i32
  %x2 = xor i32 %z2a, %z2b
  %s2 = select i1 %c1, i32 %z2a, i32 %x2

  %g3a = getelementptr inbounds i8, ptr %pin, i64 5
  %l3a = load i8, ptr %g3a, align 1
  %z3a = zext i8 %l3a to i32
  %g3b = getelementptr inbounds i8, ptr %pin, i64 6
  %l3b = load i8, ptr %g3b, align 1
  %z3b = zext i8 %l3b to i32
  %x3 = xor i32 %z3a, %z3b
  %s3 = select i1 %c1, i32 %z3a, i32 %x3

  %c3 = icmp ne i8 %l1a, 0
  %c4 = icmp ne i8 %l2a, 0

  %s4 = select i1 %c3, i32 %s1, i32 %s2
  %s5 = select i1 %c4, i32 %s4, i32 %s3

  ret i32 %s5
}

attributes #0 = { nounwind ssp uwtable }
