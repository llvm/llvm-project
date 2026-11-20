; RUN: opt -passes='default<O2>' -S %s | llc -mtriple=x86_64 -mattr=-lzcnt -verify-machineinstrs -o - | FileCheck %s --check-prefix=BSR
; RUN: opt -passes='default<O2>' -S %s | llc -mtriple=x86_64 -mattr=+lzcnt -verify-machineinstrs -o - | FileCheck %s --check-prefix=LZCNT
;
; Unrolling makes the second BSR reuse the loop-carried sum register. Its old
; value has already been saved and must not delay the independent bit scan.
; BSR-LABEL: scaled_log2_sum:
; BSR-NOT: lzcnt
; BSR: bsrq %r8, %r8
; BSR: movq %rax, %r10
; BSR-NEXT: addq %r8, %r10
; BSR-NEXT: orq $1, %r9
; BSR-NEXT: xorl %eax, %eax
; BSR-NEXT: bsrq %r9, %rax
; BSR: retq
; LZCNT-LABEL: scaled_log2_sum:
; LZCNT-NOT: bsr
; LZCNT: lzcntq
; LZCNT-NOT: bsr
; LZCNT: retq

target triple = "x86_64-unknown-linux-gnu"
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
define i64 @scaled_log2_sum(ptr %p, i64 %n) {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %sum = phi i64 [0, %entry], [%total, %loop]
  %addr = getelementptr i64, ptr %p, i64 %i
  %v = load i64, ptr %addr, align 8
  %nz = or i64 %v, 1
  %lz = call i64 @llvm.ctlz.i64(i64 %nz, i1 false)
  %log = xor i64 %lz, 63
  %mul = mul i64 %log, 9
  %add = add i64 %mul, 73
  %sz = lshr i64 %add, 6
  %total = add i64 %sum, %sz
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %result = phi i64 [0, %entry], [%total, %loop]
  ret i64 %result
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.count", i32 2}
