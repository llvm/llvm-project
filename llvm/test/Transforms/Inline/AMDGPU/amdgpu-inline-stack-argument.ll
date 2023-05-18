; REQUIRES: asserts
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline -inline-cost-full=true -inline-threshold=0 -inline-instr-cost=5 -inline-call-penalty=0 -debug-only=inline < %s 2>&1 | FileCheck %s

; CHECK:      NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %noinlinecall1 = call noundef i32 @non_inlining_call
; CHECK:      NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %noinlinecall2 = call noundef i32 @non_inlining_call
; CHECK-NOT:  NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %inlinecall1 = call noundef i32 @inlining_call
; CHECK-NOT:  NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %inlinecall2 = call noundef i32 @inlining_call

define noundef i32 @non_inlining_call(i32 noundef %a0, i32 noundef %b0, i32 noundef %c0, i32 noundef %d0, i32 noundef %e0, i32 noundef %f0, i32 noundef %g0, i32 noundef %h0, i32 noundef %i0, i32 noundef %j0, i32 noundef %k0, i32 noundef %l0, i32 noundef %m0, i32 noundef %n0, i32 noundef %o0, i32 noundef %p0, i32 noundef %q0, i32 noundef %r0, i32 noundef %s0, i32 noundef %t0, i32 noundef %u0, i32 noundef %v0, i32 noundef %w0, i32 noundef %x0, i32 noundef %y0, i32 noundef %z0, i32 noundef %a1, i32 noundef %b1, i32 noundef %c1, i32 noundef %d1, i32 noundef %e1, i32 noundef %f1) {
entry:
  %xor = xor i32 %a0, %b0
  %xor1 = xor i32 %xor, %c0
  %xor2 = xor i32 %xor1, %d0
  %xor3 = xor i32 %xor2, %e0
  %xor4 = xor i32 %xor3, %f0
  %xor5 = xor i32 %xor4, %g0
  %xor6 = xor i32 %xor5, %h0
  %xor7 = xor i32 %xor6, %i0
  %xor8 = xor i32 %xor7, %j0
  %xor9 = xor i32 %xor8, %k0
  %xor10 = xor i32 %xor9, %l0
  %xor11 = xor i32 %xor10, %m0
  %xor12 = xor i32 %xor11, %n0
  %xor13 = xor i32 %xor12, %o0
  %xor14 = xor i32 %xor13, %p0
  %xor15 = xor i32 %xor14, %q0
  %xor16 = xor i32 %xor15, %r0
  %xor17 = xor i32 %xor16, %s0
  %xor18 = xor i32 %xor17, %t0
  %xor19 = xor i32 %xor18, %u0
  %xor20 = xor i32 %xor19, %v0
  %xor21 = xor i32 %xor20, %w0
  %xor22 = xor i32 %xor21, %x0
  %xor23 = xor i32 %xor22, %y0
  %xor24 = xor i32 %xor23, %z0
  %xor25 = xor i32 %xor24, %a1
  %xor26 = xor i32 %xor25, %b1
  %xor27 = xor i32 %xor26, %c1
  %xor28 = xor i32 %xor27, %d1
  %xor29 = xor i32 %xor28, %e1
  %xor30 = xor i32 %xor29, %f1
  %xor31 = xor i32 %xor30, 1
  %xor32 = xor i32 %xor31, 2
  ret i32 %xor32
}

define noundef i32 @inlining_call(i32 noundef %a0, i32 noundef %b0, i32 noundef %c0, i32 noundef %d0, i32 noundef %e0, i32 noundef %f0, i32 noundef %g0, i32 noundef %h0, i32 noundef %i0, i32 noundef %j0, i32 noundef %k0, i32 noundef %l0, i32 noundef %m0, i32 noundef %n0, i32 noundef %o0, i32 noundef %p0, i32 noundef %q0, i32 noundef %r0, i32 noundef %s0, i32 noundef %t0, i32 noundef %u0, i32 noundef %v0, i32 noundef %w0, i32 noundef %x0, i32 noundef %y0, i32 noundef %z0, i32 noundef %a1, i32 noundef %b1, i32 noundef %c1, i32 noundef %d1, i32 noundef %e1, i32 noundef %f1, i32 noundef %g1) {
entry:
  %xor = xor i32 %a0, %b0
  %xor1 = xor i32 %xor, %c0
  %xor2 = xor i32 %xor1, %d0
  %xor3 = xor i32 %xor2, %e0
  %xor4 = xor i32 %xor3, %f0
  %xor5 = xor i32 %xor4, %g0
  %xor6 = xor i32 %xor5, %h0
  %xor7 = xor i32 %xor6, %i0
  %xor8 = xor i32 %xor7, %j0
  %xor9 = xor i32 %xor8, %k0
  %xor10 = xor i32 %xor9, %l0
  %xor11 = xor i32 %xor10, %m0
  %xor12 = xor i32 %xor11, %n0
  %xor13 = xor i32 %xor12, %o0
  %xor14 = xor i32 %xor13, %p0
  %xor15 = xor i32 %xor14, %q0
  %xor16 = xor i32 %xor15, %r0
  %xor17 = xor i32 %xor16, %s0
  %xor18 = xor i32 %xor17, %t0
  %xor19 = xor i32 %xor18, %u0
  %xor20 = xor i32 %xor19, %v0
  %xor21 = xor i32 %xor20, %w0
  %xor22 = xor i32 %xor21, %x0
  %xor23 = xor i32 %xor22, %y0
  %xor24 = xor i32 %xor23, %z0
  %xor25 = xor i32 %xor24, %a1
  %xor26 = xor i32 %xor25, %b1
  %xor27 = xor i32 %xor26, %c1
  %xor28 = xor i32 %xor27, %d1
  %xor29 = xor i32 %xor28, %e1
  %xor30 = xor i32 %xor29, %f1
  %xor31 = xor i32 %xor30, %g1
  %xor32 = xor i32 %xor30, 1
  %xor33 = xor i32 %xor31, 2
  ret i32 %xor33
}

; Calling each (non-)inlining function twice to make sure they won't get the sole call inlining cost bonus. 
define i32 @Caller(ptr noundef %in) {
entry:
  %arrayidx = getelementptr inbounds i32, ptr %in, i64 0
  %a0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %in, i64 1
  %b0 = load i32, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %in, i64 2
  %c0 = load i32, ptr %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %in, i64 3
  %d0 = load i32, ptr %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %in, i64 4
  %e0 = load i32, ptr %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i32, ptr %in, i64 5
  %f0 = load i32, ptr %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds i32, ptr %in, i64 6
  %g0 = load i32, ptr %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i32, ptr %in, i64 7
  %h0 = load i32, ptr %arrayidx7, align 4
  %arrayidx8 = getelementptr inbounds i32, ptr %in, i64 8
  %i0 = load i32, ptr %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32, ptr %in, i64 9
  %j0 = load i32, ptr %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds i32, ptr %in, i64 10
  %k0 = load i32, ptr %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds i32, ptr %in, i64 11
  %l0 = load i32, ptr %arrayidx11, align 4
  %arrayidx12 = getelementptr inbounds i32, ptr %in, i64 12
  %m0 = load i32, ptr %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds i32, ptr %in, i64 13
  %n0 = load i32, ptr %arrayidx13, align 4
  %arrayidx14 = getelementptr inbounds i32, ptr %in, i64 14
  %o0 = load i32, ptr %arrayidx14, align 4
  %arrayidx15 = getelementptr inbounds i32, ptr %in, i64 15
  %p0 = load i32, ptr %arrayidx15, align 4
  %arrayidx16 = getelementptr inbounds i32, ptr %in, i64 16
  %q0 = load i32, ptr %arrayidx16, align 4
  %arrayidx17 = getelementptr inbounds i32, ptr %in, i64 17
  %r0 = load i32, ptr %arrayidx17, align 4
  %arrayidx18 = getelementptr inbounds i32, ptr %in, i64 18
  %s0 = load i32, ptr %arrayidx18, align 4
  %arrayidx19 = getelementptr inbounds i32, ptr %in, i64 19
  %t0 = load i32, ptr %arrayidx19, align 4
  %arrayidx20 = getelementptr inbounds i32, ptr %in, i64 20
  %u0 = load i32, ptr %arrayidx20, align 4
  %arrayidx21 = getelementptr inbounds i32, ptr %in, i64 21
  %v0 = load i32, ptr %arrayidx21, align 4
  %arrayidx22 = getelementptr inbounds i32, ptr %in, i64 22
  %w0 = load i32, ptr %arrayidx22, align 4
  %arrayidx23 = getelementptr inbounds i32, ptr %in, i64 23
  %x0 = load i32, ptr %arrayidx23, align 4
  %arrayidx24 = getelementptr inbounds i32, ptr %in, i64 24
  %y0 = load i32, ptr %arrayidx24, align 4
  %arrayidx25 = getelementptr inbounds i32, ptr %in, i64 25
  %z0 = load i32, ptr %arrayidx25, align 4
  %arrayidx26 = getelementptr inbounds i32, ptr %in, i64 26
  %a1 = load i32, ptr %arrayidx26, align 4
  %arrayidx27 = getelementptr inbounds i32, ptr %in, i64 27
  %b1 = load i32, ptr %arrayidx27, align 4
  %arrayidx28 = getelementptr inbounds i32, ptr %in, i64 28
  %c1 = load i32, ptr %arrayidx28, align 4
  %arrayidx29 = getelementptr inbounds i32, ptr %in, i64 29
  %d1 = load i32, ptr %arrayidx29, align 4
  %arrayidx30 = getelementptr inbounds i32, ptr %in, i64 30
  %e1 = load i32, ptr %arrayidx30, align 4
  %arrayidx31 = getelementptr inbounds i32, ptr %in, i64 31
  %f1 = load i32, ptr %arrayidx31, align 4
  %arrayidx32 = getelementptr inbounds i32, ptr %in, i64 32
  %g1 = load i32, ptr %arrayidx32, align 4
  %noinlinecall1 = call noundef i32 @non_inlining_call(i32 noundef %a0, i32 noundef %b0, i32 noundef %c0, i32 noundef %d0, i32 noundef %e0, i32 noundef %f0, i32 noundef %g0, i32 noundef %h0, i32 noundef %i0, i32 noundef %j0, i32 noundef %k0, i32 noundef %l0, i32 noundef %m0, i32 noundef %n0, i32 noundef %o0, i32 noundef %p0, i32 noundef %q0, i32 noundef %r0, i32 noundef %s0, i32 noundef %t0, i32 noundef %u0, i32 noundef %v0, i32 noundef %w0, i32 noundef %x0, i32 noundef %y0, i32 noundef %z0, i32 noundef %a1, i32 noundef %b1, i32 noundef %c1, i32 noundef %d1, i32 noundef %e1, i32 noundef %f1)
  %add = add i32 0, %noinlinecall1
  %noinlinecall2 = call noundef i32 @non_inlining_call(i32 noundef %a0, i32 noundef %b0, i32 noundef %c0, i32 noundef %d0, i32 noundef %e0, i32 noundef %f0, i32 noundef %g0, i32 noundef %h0, i32 noundef %i0, i32 noundef %j0, i32 noundef %k0, i32 noundef %l0, i32 noundef %m0, i32 noundef %n0, i32 noundef %o0, i32 noundef %p0, i32 noundef %q0, i32 noundef %r0, i32 noundef %s0, i32 noundef %t0, i32 noundef %u0, i32 noundef %v0, i32 noundef %w0, i32 noundef %x0, i32 noundef %y0, i32 noundef %z0, i32 noundef %a1, i32 noundef %b1, i32 noundef %c1, i32 noundef %d1, i32 noundef %e1, i32 noundef %f1)
  %add2 = add i32 %add, %noinlinecall2
  %inlinecall1 = call noundef i32 @inlining_call(i32 noundef %a0, i32 noundef %b0, i32 noundef %c0, i32 noundef %d0, i32 noundef %e0, i32 noundef %f0, i32 noundef %g0, i32 noundef %h0, i32 noundef %i0, i32 noundef %j0, i32 noundef %k0, i32 noundef %l0, i32 noundef %m0, i32 noundef %n0, i32 noundef %o0, i32 noundef %p0, i32 noundef %q0, i32 noundef %r0, i32 noundef %s0, i32 noundef %t0, i32 noundef %u0, i32 noundef %v0, i32 noundef %w0, i32 noundef %x0, i32 noundef %y0, i32 noundef %z0, i32 noundef %a1, i32 noundef %b1, i32 noundef %c1, i32 noundef %d1, i32 noundef %e1, i32 noundef %f1, i32 noundef %g1)
  %add3 = add i32 %add2, %inlinecall1
  %inlinecall2 = call noundef i32 @inlining_call(i32 noundef %a0, i32 noundef %b0, i32 noundef %c0, i32 noundef %d0, i32 noundef %e0, i32 noundef %f0, i32 noundef %g0, i32 noundef %h0, i32 noundef %i0, i32 noundef %j0, i32 noundef %k0, i32 noundef %l0, i32 noundef %m0, i32 noundef %n0, i32 noundef %o0, i32 noundef %p0, i32 noundef %q0, i32 noundef %r0, i32 noundef %s0, i32 noundef %t0, i32 noundef %u0, i32 noundef %v0, i32 noundef %w0, i32 noundef %x0, i32 noundef %y0, i32 noundef %z0, i32 noundef %a1, i32 noundef %b1, i32 noundef %c1, i32 noundef %d1, i32 noundef %e1, i32 noundef %f1, i32 noundef %g1)
  %add4 = add i32 %add3, %inlinecall2
  ret i32 %add4
}
