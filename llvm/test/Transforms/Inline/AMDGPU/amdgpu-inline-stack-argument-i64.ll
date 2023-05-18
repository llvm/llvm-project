; REQUIRES: asserts
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline -inline-cost-full=true -inline-threshold=0 -inline-instr-cost=5 -inline-call-penalty=0 -debug-only=inline < %s 2>&1 | FileCheck %s

; CHECK:      NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %noinlinecall1 = call noundef i64 @non_inlining_call
; CHECK:      NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %noinlinecall2 = call noundef i64 @non_inlining_call
; CHECK-NOT:  NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %inlinecall1 = call noundef i64 @inlining_call
; CHECK-NOT:  NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %inlinecall2 = call noundef i64 @inlining_call

define noundef i64 @non_inlining_call(i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0) {
entry:
  %xor = xor i64 %a0, %b0
  %xor1 = xor i64 %xor, %c0
  %xor2 = xor i64 %xor1, %d0
  %xor3 = xor i64 %xor2, %e0
  %xor4 = xor i64 %xor3, %f0
  %xor5 = xor i64 %xor4, %g0
  %xor6 = xor i64 %xor5, %h0
  %xor7 = xor i64 %xor6, %i0
  %xor8 = xor i64 %xor7, %j0
  %xor9 = xor i64 %xor8, %k0
  %xor10 = xor i64 %xor9, %l0
  %xor11 = xor i64 %xor10, %m0
  %xor12 = xor i64 %xor11, %n0
  %xor13 = xor i64 %xor12, %o0
  %xor14 = xor i64 %xor13, %p0
  %xor15 = xor i64 %xor14, 1
  %xor16 = xor i64 %xor15, 2
  ret i64 %xor16
}

define noundef i64 @inlining_call(i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0, i64 noundef %q0) {
entry:
  %xor = xor i64 %a0, %b0
  %xor1 = xor i64 %xor, %c0
  %xor2 = xor i64 %xor1, %d0
  %xor3 = xor i64 %xor2, %e0
  %xor4 = xor i64 %xor3, %f0
  %xor5 = xor i64 %xor4, %g0
  %xor6 = xor i64 %xor5, %h0
  %xor7 = xor i64 %xor6, %i0
  %xor8 = xor i64 %xor7, %j0
  %xor9 = xor i64 %xor8, %k0
  %xor10 = xor i64 %xor9, %l0
  %xor11 = xor i64 %xor10, %m0
  %xor12 = xor i64 %xor11, %n0
  %xor13 = xor i64 %xor12, %o0
  %xor14 = xor i64 %xor13, %p0
  %xor15 = xor i64 %xor14, %q0
  %xor16 = xor i64 %xor15, 1
  %xor17 = xor i64 %xor16, 1
  ret i64 %xor17
}

; Calling each (non-)inlining function twice to make sure they won't get the sole call inlining cost bonus. 
define i64 @Caller(ptr noundef %in) {
entry:
  %arrayidx = getelementptr inbounds i64, ptr %in, i64 0
  %a0 = load i64, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i64, ptr %in, i64 1
  %b0 = load i64, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i64, ptr %in, i64 2
  %c0 = load i64, ptr %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i64, ptr %in, i64 3
  %d0 = load i64, ptr %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i64, ptr %in, i64 4
  %e0 = load i64, ptr %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i64, ptr %in, i64 5
  %f0 = load i64, ptr %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds i64, ptr %in, i64 6
  %g0 = load i64, ptr %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i64, ptr %in, i64 7
  %h0 = load i64, ptr %arrayidx7, align 4
  %arrayidx8 = getelementptr inbounds i64, ptr %in, i64 8
  %i0 = load i64, ptr %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i64, ptr %in, i64 9
  %j0 = load i64, ptr %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds i64, ptr %in, i64 10
  %k0 = load i64, ptr %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds i64, ptr %in, i64 11
  %l0 = load i64, ptr %arrayidx11, align 4
  %arrayidx12 = getelementptr inbounds i64, ptr %in, i64 12
  %m0 = load i64, ptr %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds i64, ptr %in, i64 13
  %n0 = load i64, ptr %arrayidx13, align 4
  %arrayidx14 = getelementptr inbounds i64, ptr %in, i64 14
  %o0 = load i64, ptr %arrayidx14, align 4
  %arrayidx15 = getelementptr inbounds i64, ptr %in, i64 15
  %p0 = load i64, ptr %arrayidx15, align 4
  %arrayidx16 = getelementptr inbounds i64, ptr %in, i64 16
  %q0 = load i64, ptr %arrayidx16, align 4
  %noinlinecall1 = call noundef i64 @non_inlining_call(i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0)
  %add = add i64 0, %noinlinecall1
  %noinlinecall2 = call noundef i64 @non_inlining_call(i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0)
  %add2 = add i64 %add, %noinlinecall2
  %inlinecall1 = call noundef i64 @inlining_call(i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0, i64 noundef %q0)
  %add3 = add i64 %add2, %inlinecall1
  %inlinecall2 = call noundef i64 @inlining_call(i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0, i64 noundef %q0)
  %add4 = add i64 %add3, %inlinecall2
  ret i64 %add4
}
