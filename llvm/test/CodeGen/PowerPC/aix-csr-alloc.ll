; REQUIRES: asserts
; RUN: llc -mtriple=powerpc64-aix-xcoff -debug-only=regalloc < %s 2>&1 | \
; RUN:   FileCheck --check-prefix=AIX-64 %s
; RUN: llc -mtriple=powerpc-aix-xcoff -debug-only=regalloc < %s 2>&1 | \
; RUN:   FileCheck --check-prefix=AIX-32 %s

define i32 @g(i32 %a, i32 %b) {
; AIX-64: AllocationOrder(G8RC_and_G8RC_NOX0) = [ $x3 $x4 $x5 $x6 $x7 $x8 $x9 $x10 $x11 $x12 $x31 $x30 $x29 $x28 $x27 $x26 $x25 $x24 $x23 $x22 $x21 $x20 $x19 $x18 $x17 $x16 $x15 $x14 ]
; AIX-64: AllocationOrder(G8RC) = [ $x3 $x4 $x5 $x6 $x7 $x8 $x9 $x10 $x11 $x12 $x0 $x31 $x30 $x29 $x28 $x27 $x26 $x25 $x24 $x23 $x22 $x21 $x20 $x19 $x18 $x17 $x16 $x15 $x14 ]
; AIX-32: AllocationOrder(GPRC) = [ $r3 $r4 $r5 $r6 $r7 $r8 $r9 $r10 $r11 $r12 $r0 $r31 $r30 $r29 $r28 $r27 $r26 $r25 $r24 $r23 $r22 $r21 $r20 $r19 $r18 $r17 $r16 $r15 $r14 $r13 ]
; AIX-32: AllocationOrder(GPRC_and_GPRC_NOR0) = [ $r3 $r4 $r5 $r6 $r7 $r8 $r9 $r10 $r11 $r12 $r31 $r30 $r29 $r28 $r27 $r26 $r25 $r24 $r23 $r22 $r21 $r20 $r19 $r18 $r17 $r16 $r15 $r14 $r13 ]
  %c = add i32 %a, %b
  %d = shl i32 %a, 4
  %cmp = icmp slt i32 %c, %d
  %e = select i1 %cmp, i32 %a, i32 %b
  ret i32 %e
}

define float @f(float %a, float %b) {
; AIX-32: AllocationOrder(F4RC) = [ $f0 $f1 $f2 $f3 $f4 $f5 $f6 $f7 $f8 $f9 $f10 $f11 $f12 $f13 $f31 $f30 $f29 $f28 $f27 $f26 $f25 $f24 $f23 $f22 $f21 $f20 $f19 $f18 $f17 $f16 $f15 $f14 ]
  %c = fadd float %a, %b
  ret float %c
}

define double @d(double %a, double %b) {
; AIX-64: AllocationOrder(VFRC) = [ $vf2 $vf3 $vf4 $vf5 $vf0 $vf1 $vf6 $vf7 $vf8 $vf9 $vf10 $vf11 $vf12 $vf13 $vf14 $vf15 $vf16 $vf17 $vf18 $vf19 $vf31 $vf30 $vf29 $vf28 $vf27 $vf26 $vf25 $vf24 $vf23 $vf22 $vf21 $vf20 ]
; AIX-64: AllocationOrder(F8RC) = [ $f0 $f1 $f2 $f3 $f4 $f5 $f6 $f7 $f8 $f9 $f10 $f11 $f12 $f13 $f31 $f30 $f29 $f28 $f27 $f26 $f25 $f24 $f23 $f22 $f21 $f20 $f19 $f18 $f17 $f16 $f15 $f14 ]
  %c = fadd double %a, %b
  ret double %c
}
