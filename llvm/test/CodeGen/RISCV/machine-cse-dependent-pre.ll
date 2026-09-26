; RUN: llc -O2 -mtriple=riscv64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
; RUN: opt -S -mtriple=riscv64 -passes='default<O2>' %s | \
; RUN:   llc -O2 -mtriple=riscv64 -mattr=+m -verify-machineinstrs | FileCheck %s
;
; Division and remainder are distinct IR operations. Selection lowers both
; to MULHU followed by SRLI, so MachineCSE must expose the dependent shift
; during PRE. One invocation should share the whole sequence in the loop.
; This case also survives the ordinary O2 IR pipeline.
;
define void @divrem_loop(ptr %input, ptr %quotients, ptr %remainders, i64 %n) {
; CHECK-LABEL: divrem_loop:
; CHECK-NOT: mulhu
; CHECK-NOT: srli
; CHECK: mulhu [[HIGH:[a-z][a-z0-9]*]],
; CHECK-NOT: mulhu
; CHECK-NOT: srli
; CHECK: srli {{[a-z][a-z0-9]*}}, [[HIGH]], 3
; CHECK-NOT: mulhu
; CHECK-NOT: srli
; CHECK: .Lfunc_end{{[0-9]+}}:
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %ip = getelementptr i64, ptr %input, i64 %i
  %x = load i64, ptr %ip, align 8
  %bit = and i64 %x, 1
  %cond = icmp eq i64 %bit, 0
  br i1 %cond, label %then, label %else
then:
  %q = udiv i64 %x, 10
  %qp = getelementptr i64, ptr %quotients, i64 %i
  store i64 %q, ptr %qp, align 8
  br label %latch
else:
  %r = urem i64 %x, 10
  %rp = getelementptr i64, ptr %remainders, i64 %i
  store i64 %r, ptr %rp, align 8
  br label %latch
latch:
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; A deeper lowering needs three rounds of the original PRE/CSE sequence:
; SRLI x, 3 -> MULHU -> SRLI high, 4. Expose all three levels in one traversal.
; Quotient and remainder are distinct IR operations with a common lowering.
define void @divrem_1000_loop(ptr %input, ptr %quotients, ptr %remainders, i64 %n) {
; CHECK-LABEL: divrem_1000_loop:
; CHECK-NOT: mulhu
; CHECK-NOT: srli
; CHECK: srli [[INPUT:[a-z][a-z0-9]*]], {{[a-z][a-z0-9]*}}, 3
; CHECK-NOT: mulhu
; CHECK-NOT: srli
; CHECK: mulhu [[HIGH:[a-z][a-z0-9]*]], [[INPUT]],
; CHECK-NOT: mulhu
; CHECK-NOT: srli
; CHECK: srli {{[a-z][a-z0-9]*}}, [[HIGH]], 4
; CHECK-NOT: mulhu
; CHECK-NOT: srli
; CHECK: .Lfunc_end{{[0-9]+}}:
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %ip = getelementptr i64, ptr %input, i64 %i
  %x = load i64, ptr %ip, align 8
  %bit = and i64 %x, 1
  %cond = icmp eq i64 %bit, 0
  br i1 %cond, label %then, label %else
then:
  %q = udiv i64 %x, 1000
  %qp = getelementptr i64, ptr %quotients, i64 %i
  store i64 %q, ptr %qp, align 8
  br label %latch
else:
  %r = urem i64 %x, 1000
  %rp = getelementptr i64, ptr %remainders, i64 %i
  store i64 %r, ptr %rp, align 8
  br label %latch
latch:
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
