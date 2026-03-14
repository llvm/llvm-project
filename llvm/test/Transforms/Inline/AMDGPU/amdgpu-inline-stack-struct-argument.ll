; REQUIRES: asserts
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline -inline-cost-full=true -inline-threshold=0 -inline-instr-cost=5 -inline-call-penalty=0 -debug-only=inline < %s 2>&1 | FileCheck %s

; CHECK:      NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %noinlinecall1 = call noundef i64 @non_inlining_call
; CHECK:      NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %noinlinecall2 = call noundef i64 @non_inlining_call
; CHECK-NOT:  NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %inlinecall1 = call noundef i64 @inlining_call
; CHECK-NOT:  NOT Inlining (cost={{[0-9]+}}, threshold={{[0-9]+}}), Call:   %inlinecall2 = call noundef i64 @inlining_call

%noinlineT =  type {{ptr, ptr}, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64}
%inlineT =    type {{ptr, ptr}, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64}

define noundef i64 @non_inlining_call(%noinlineT noundef %struc) {
entry:
  %ptra0 = extractvalue %noinlineT %struc, 0, 0
  %ptrb0 = extractvalue %noinlineT %struc, 0, 1
  %ptrc0 = extractvalue %noinlineT %struc, 1
  %a0 = load i64, ptr %ptra0, align 8
  %b0 = load i64, ptr %ptrb0, align 8
  %c0 = load i64, ptr %ptrc0, align 8
  %d0 = extractvalue %noinlineT %struc, 2
  %e0 = extractvalue %noinlineT %struc, 3
  %f0 = extractvalue %noinlineT %struc, 4
  %g0 = extractvalue %noinlineT %struc, 5
  %h0 = extractvalue %noinlineT %struc, 6
  %i0 = extractvalue %noinlineT %struc, 7
  %j0 = extractvalue %noinlineT %struc, 8
  %k0 = extractvalue %noinlineT %struc, 9
  %l0 = extractvalue %noinlineT %struc, 10
  %m0 = extractvalue %noinlineT %struc, 11
  %n0 = extractvalue %noinlineT %struc, 12
  %o0 = extractvalue %noinlineT %struc, 13
  %p0 = extractvalue %noinlineT %struc, 14
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

define noundef i64 @inlining_call(%inlineT noundef %struc) {
entry:
  %ptra0 = extractvalue %inlineT %struc, 0, 0
  %ptrb0 = extractvalue %inlineT %struc, 0, 1
  %ptrc0 = extractvalue %inlineT %struc, 1
  %a0 = load i64, ptr %ptra0, align 8
  %b0 = load i64, ptr %ptrb0, align 8
  %c0 = load i64, ptr %ptrc0, align 8
  %d0 = extractvalue %inlineT %struc, 2
  %e0 = extractvalue %inlineT %struc, 3
  %f0 = extractvalue %inlineT %struc, 4
  %g0 = extractvalue %inlineT %struc, 5
  %h0 = extractvalue %inlineT %struc, 6
  %i0 = extractvalue %inlineT %struc, 7
  %j0 = extractvalue %inlineT %struc, 8
  %k0 = extractvalue %inlineT %struc, 9
  %l0 = extractvalue %inlineT %struc, 10
  %m0 = extractvalue %inlineT %struc, 11
  %n0 = extractvalue %inlineT %struc, 12
  %o0 = extractvalue %inlineT %struc, 13
  %p0 = extractvalue %inlineT %struc, 14
  %q0 = extractvalue %inlineT %struc, 15
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
  %ptra0 = getelementptr inbounds i64, ptr %in, i64 0
  %ptrb0 = getelementptr inbounds i64, ptr %in, i64 1
  %ptrc0 = getelementptr inbounds i64, ptr %in, i64 2
  %ptrd0 = getelementptr inbounds i64, ptr %in, i64 3
  %ptre0 = getelementptr inbounds i64, ptr %in, i64 4
  %ptrf0 = getelementptr inbounds i64, ptr %in, i64 5
  %ptrg0 = getelementptr inbounds i64, ptr %in, i64 6
  %ptrh0 = getelementptr inbounds i64, ptr %in, i64 7
  %ptri0 = getelementptr inbounds i64, ptr %in, i64 8
  %ptrj0 = getelementptr inbounds i64, ptr %in, i64 9
  %ptrk0 = getelementptr inbounds i64, ptr %in, i64 10
  %ptrl0 = getelementptr inbounds i64, ptr %in, i64 11
  %ptrm0 = getelementptr inbounds i64, ptr %in, i64 12
  %ptrn0 = getelementptr inbounds i64, ptr %in, i64 13
  %ptro0 = getelementptr inbounds i64, ptr %in, i64 14
  %ptrp0 = getelementptr inbounds i64, ptr %in, i64 15
  %ptrq0 = getelementptr inbounds i64, ptr %in, i64 16
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
  %noinlinestruc1 = insertvalue %noinlineT undef, ptr %ptra0, 0, 0
  %noinlinestruc2 = insertvalue %noinlineT %noinlinestruc1, ptr %ptrb0, 0, 1
  %noinlinestruc3 = insertvalue %noinlineT %noinlinestruc2, ptr %ptrc0, 1
  %noinlinestruc4 = insertvalue %noinlineT %noinlinestruc3, i64 %d0, 2
  %noinlinestruc5 = insertvalue %noinlineT %noinlinestruc4, i64 %e0, 3
  %noinlinestruc6 = insertvalue %noinlineT %noinlinestruc5, i64 %f0, 4
  %noinlinestruc7 = insertvalue %noinlineT %noinlinestruc6, i64 %g0, 5
  %noinlinestruc8 = insertvalue %noinlineT %noinlinestruc7, i64 %h0, 6
  %noinlinestruc9 = insertvalue %noinlineT %noinlinestruc8, i64 %i0, 7
  %noinlinestruc10 = insertvalue %noinlineT %noinlinestruc9, i64 %j0, 8
  %noinlinestruc11 = insertvalue %noinlineT %noinlinestruc10, i64 %k0, 9
  %noinlinestruc12 = insertvalue %noinlineT %noinlinestruc11, i64 %l0, 10
  %noinlinestruc13 = insertvalue %noinlineT %noinlinestruc12, i64 %m0, 11
  %noinlinestruc14 = insertvalue %noinlineT %noinlinestruc13, i64 %n0, 12
  %noinlinestruc15 = insertvalue %noinlineT %noinlinestruc14, i64 %o0, 13
  %noinlinestruc16 = insertvalue %noinlineT %noinlinestruc15, i64 %p0, 14
  %inlinestruc1 = insertvalue %inlineT undef, ptr %ptra0, 0, 0
  %inlinestruc2 = insertvalue %inlineT %inlinestruc1, ptr %ptrb0, 0, 1
  %inlinestruc3 = insertvalue %inlineT %inlinestruc2, ptr %ptrc0, 1
  %inlinestruc4 = insertvalue %inlineT %inlinestruc3, i64 %d0, 2
  %inlinestruc5 = insertvalue %inlineT %inlinestruc4, i64 %e0, 3
  %inlinestruc6 = insertvalue %inlineT %inlinestruc5, i64 %f0, 4
  %inlinestruc7 = insertvalue %inlineT %inlinestruc6, i64 %g0, 5
  %inlinestruc8 = insertvalue %inlineT %inlinestruc7, i64 %h0, 6
  %inlinestruc9 = insertvalue %inlineT %inlinestruc8, i64 %i0, 7
  %inlinestruc10 = insertvalue %inlineT %inlinestruc9, i64 %j0, 8
  %inlinestruc11 = insertvalue %inlineT %inlinestruc10, i64 %k0, 9
  %inlinestruc12 = insertvalue %inlineT %inlinestruc11, i64 %l0, 10
  %inlinestruc13 = insertvalue %inlineT %inlinestruc12, i64 %m0, 11
  %inlinestruc14 = insertvalue %inlineT %inlinestruc13, i64 %n0, 12
  %inlinestruc15 = insertvalue %inlineT %inlinestruc14, i64 %o0, 13
  %inlinestruc16 = insertvalue %inlineT %inlinestruc15, i64 %p0, 14
  %inlinestruc17 = insertvalue %inlineT %inlinestruc16, i64 %q0, 15
  %noinlinecall1 = call noundef i64 @non_inlining_call(%noinlineT noundef %noinlinestruc16)
  %add = add i64 0, %noinlinecall1
  %noinlinecall2 = call noundef i64 @non_inlining_call(%noinlineT noundef %noinlinestruc16)
  %add2 = add i64 %add, %noinlinecall2
  %inlinecall1 = call noundef i64 @inlining_call(%inlineT noundef %inlinestruc17)
  %add3 = add i64 %add2, %inlinecall1
  %inlinecall2 = call noundef i64 @inlining_call(%inlineT noundef %inlinestruc17)
  %add4 = add i64 %add3, %inlinecall2
  ret i64 %add4
}
