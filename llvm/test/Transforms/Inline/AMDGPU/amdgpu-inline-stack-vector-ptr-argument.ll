; REQUIRES: asserts
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline -inline-cost-full=true -inline-threshold=0 -inline-instr-cost=5 -inline-call-penalty=0 -debug-only=inline < %s 2>&1 | FileCheck %s

; CHECK:      NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %noinlinecall1 = call noundef i64 @non_inlining_call
; CHECK:      NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %noinlinecall2 = call noundef i64 @non_inlining_call
; CHECK-NOT:  NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %inlinecall1 = call noundef i64 @inlining_call
; CHECK-NOT:  NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %inlinecall2 = call noundef i64 @inlining_call

define noundef i64 @non_inlining_call(<2 x ptr> noundef %ptrvec, ptr noundef %ptrc0, ptr noundef %ptrd0, ptr noundef %ptre0, ptr noundef %ptrf0, ptr noundef %ptrg0, ptr noundef %ptrh0, ptr noundef %ptri0, ptr noundef %ptrj0, ptr noundef %ptrk0, ptr noundef %ptrl0, ptr noundef %ptrm0, ptr noundef %ptrn0, ptr noundef %ptro0, ptr noundef %ptrp0) {
entry:
  %ptra0 = extractelement <2 x ptr> %ptrvec, i32 0
  %ptrb0 = extractelement <2 x ptr> %ptrvec, i32 1
  %a0 = load i64, ptr %ptra0, align 8
  %b0 = load i64, ptr %ptrb0, align 8
  %c0 = load i64, ptr %ptrc0, align 8
  %d0 = load i64, ptr %ptrd0, align 8
  %e0 = load i64, ptr %ptre0, align 8
  %f0 = load i64, ptr %ptrf0, align 8
  %g0 = load i64, ptr %ptrg0, align 8
  %h0 = load i64, ptr %ptrh0, align 8
  %i0 = load i64, ptr %ptri0, align 8
  %j0 = load i64, ptr %ptrj0, align 8
  %k0 = load i64, ptr %ptrk0, align 8
  %l0 = load i64, ptr %ptrl0, align 8
  %m0 = load i64, ptr %ptrm0, align 8
  %n0 = load i64, ptr %ptrn0, align 8
  %o0 = load i64, ptr %ptro0, align 8
  %p0 = load i64, ptr %ptrp0, align 8
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
  ret i64 %xor14
}

define noundef i64 @inlining_call(<2 x ptr> noundef %ptrvec, ptr noundef %ptrc0, ptr noundef %ptrd0, ptr noundef %ptre0, ptr noundef %ptrf0, ptr noundef %ptrg0, ptr noundef %ptrh0, ptr noundef %ptri0, ptr noundef %ptrj0, ptr noundef %ptrk0, ptr noundef %ptrl0, ptr noundef %ptrm0, ptr noundef %ptrn0, ptr noundef %ptro0, ptr noundef %ptrp0, ptr noundef %ptrq0) {
entry:
  %ptra0 = extractelement <2 x ptr> %ptrvec, i32 0
  %ptrb0 = extractelement <2 x ptr> %ptrvec, i32 1
  %a0 = load i64, ptr %ptra0, align 8
  %b0 = load i64, ptr %ptrb0, align 8
  %c0 = load i64, ptr %ptrc0, align 8
  %d0 = load i64, ptr %ptrd0, align 8
  %e0 = load i64, ptr %ptre0, align 8
  %f0 = load i64, ptr %ptrf0, align 8
  %g0 = load i64, ptr %ptrg0, align 8
  %h0 = load i64, ptr %ptrh0, align 8
  %i0 = load i64, ptr %ptri0, align 8
  %j0 = load i64, ptr %ptrj0, align 8
  %k0 = load i64, ptr %ptrk0, align 8
  %l0 = load i64, ptr %ptrl0, align 8
  %m0 = load i64, ptr %ptrm0, align 8
  %n0 = load i64, ptr %ptrn0, align 8
  %o0 = load i64, ptr %ptro0, align 8
  %p0 = load i64, ptr %ptrp0, align 8
  %q0 = load i64, ptr %ptrq0, align 8
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
  ret i64 %xor15
}

; Calling each (non-)inlining function twice to make sure they won't get the sole call inlining cost bonus. 
define i64 @Caller(ptr noundef %in) {
entry:
  %a0 = getelementptr inbounds i64, ptr %in, i64 0
  %b0 = getelementptr inbounds i64, ptr %in, i64 1
  %vec0 = insertelement <2 x ptr> undef, ptr %a0, i32 0
  %vec1 = insertelement <2 x ptr> %vec0, ptr %b0, i32 0
  %c0 = getelementptr inbounds i64, ptr %in, i64 2
  %d0 = getelementptr inbounds i64, ptr %in, i64 3
  %e0 = getelementptr inbounds i64, ptr %in, i64 4
  %f0 = getelementptr inbounds i64, ptr %in, i64 5
  %g0 = getelementptr inbounds i64, ptr %in, i64 6
  %h0 = getelementptr inbounds i64, ptr %in, i64 7
  %i0 = getelementptr inbounds i64, ptr %in, i64 8
  %j0 = getelementptr inbounds i64, ptr %in, i64 9
  %k0 = getelementptr inbounds i64, ptr %in, i64 10
  %l0 = getelementptr inbounds i64, ptr %in, i64 11
  %m0 = getelementptr inbounds i64, ptr %in, i64 12
  %n0 = getelementptr inbounds i64, ptr %in, i64 13
  %o0 = getelementptr inbounds i64, ptr %in, i64 14
  %p0 = getelementptr inbounds i64, ptr %in, i64 15
  %q0 = getelementptr inbounds i64, ptr %in, i64 16
  %noinlinecall1 = call noundef i64 @non_inlining_call(<2 x ptr> noundef %vec1, ptr noundef %c0, ptr noundef %d0, ptr noundef %e0, ptr noundef %f0, ptr noundef %g0, ptr noundef %h0, ptr noundef %i0, ptr noundef %j0, ptr noundef %k0, ptr noundef %l0, ptr noundef %m0, ptr noundef %n0, ptr noundef %o0, ptr noundef %p0)
  %add = add i64 0, %noinlinecall1
  %noinlinecall2 = call noundef i64 @non_inlining_call(<2 x ptr> noundef %vec1, ptr noundef %c0, ptr noundef %d0, ptr noundef %e0, ptr noundef %f0, ptr noundef %g0, ptr noundef %h0, ptr noundef %i0, ptr noundef %j0, ptr noundef %k0, ptr noundef %l0, ptr noundef %m0, ptr noundef %n0, ptr noundef %o0, ptr noundef %p0)
  %add2 = add i64 %add, %noinlinecall2
  %inlinecall1 = call noundef i64 @inlining_call(<2 x ptr> noundef %vec1, ptr noundef %c0, ptr noundef %d0, ptr noundef %e0, ptr noundef %f0, ptr noundef %g0, ptr noundef %h0, ptr noundef %i0, ptr noundef %j0, ptr noundef %k0, ptr noundef %l0, ptr noundef %m0, ptr noundef %n0, ptr noundef %o0, ptr noundef %p0, ptr noundef %q0)
  %add3 = add i64 %add2, %inlinecall1
  %inlinecall2 = call noundef i64 @inlining_call(<2 x ptr> noundef %vec1, ptr noundef %c0, ptr noundef %d0, ptr noundef %e0, ptr noundef %f0, ptr noundef %g0, ptr noundef %h0, ptr noundef %i0, ptr noundef %j0, ptr noundef %k0, ptr noundef %l0, ptr noundef %m0, ptr noundef %n0, ptr noundef %o0, ptr noundef %p0, ptr noundef %q0)
  %add4 = add i64 %add3, %inlinecall2
  ret i64 %add4
}
