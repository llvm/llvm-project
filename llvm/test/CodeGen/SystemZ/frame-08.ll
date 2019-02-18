; Test the saving and restoring of GPRs in large frames.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This is the largest frame size that can use a plain LMG for %r6 and above.
; It is big enough to require two emergency spill slots at 160(%r15),
; so get a frame of size 524232 by allocating (524232 - 176) / 8 = 65507
; extra doublewords.
define void @f1(i32 *%ptr, i64 %x) {
; CHECK-LABEL: f1:
; CHECK: stmg %r6, %r15, 48(%r15)
; CHECK: .cfi_offset %r6, -112
; CHECK: .cfi_offset %r7, -104
; CHECK: .cfi_offset %r8, -96
; CHECK: .cfi_offset %r9, -88
; CHECK: .cfi_offset %r10, -80
; CHECK: .cfi_offset %r11, -72
; CHECK: .cfi_offset %r12, -64
; CHECK: .cfi_offset %r13, -56
; CHECK: .cfi_offset %r14, -48
; CHECK: .cfi_offset %r15, -40
; CHECK: agfi %r15, -524232
; CHECK: .cfi_def_cfa_offset 524392
; ...main function body...
; CHECK-NOT: ag
; CHECK: lmg %r6, %r15, 524280(%r15)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l6 = load volatile i32, i32 *%ptr
  %l7 = load volatile i32, i32 *%ptr
  %l8 = load volatile i32, i32 *%ptr
  %l9 = load volatile i32, i32 *%ptr
  %l10 = load volatile i32, i32 *%ptr
  %l11 = load volatile i32, i32 *%ptr
  %l12 = load volatile i32, i32 *%ptr
  %l13 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add4 = add i32 %l4, %add1
  %add5 = add i32 %l5, %add4
  %add6 = add i32 %l6, %add5
  %add7 = add i32 %l7, %add6
  %add8 = add i32 %l8, %add7
  %add9 = add i32 %l9, %add8
  %add10 = add i32 %l10, %add9
  %add11 = add i32 %l11, %add10
  %add12 = add i32 %l12, %add11
  %add13 = add i32 %l13, %add12
  %add14 = add i32 %l14, %add13
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  store volatile i32 %add5, i32 *%ptr
  store volatile i32 %add6, i32 *%ptr
  store volatile i32 %add7, i32 *%ptr
  store volatile i32 %add8, i32 *%ptr
  store volatile i32 %add9, i32 *%ptr
  store volatile i32 %add10, i32 *%ptr
  store volatile i32 %add11, i32 *%ptr
  store volatile i32 %add12, i32 *%ptr
  store volatile i32 %add13, i32 *%ptr
  store volatile i32 %add14, i32 *%ptr
  %y = alloca [65507 x i64], align 8
  %entry = getelementptr inbounds [65507 x i64], [65507 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %entry
  ret void
}

; This is the largest frame size that can use a plain LMG for %r14 and above
; It is big enough to require two emergency spill slots at 160(%r15),
; so get a frame of size 524168 by allocating (524168 - 176) / 8 = 65499
; extra doublewords.
define void @f2(i32 *%ptr, i64 %x) {
; CHECK-LABEL: f2:
; CHECK: stmg %r14, %r15, 112(%r15)
; CHECK: .cfi_offset %r14, -48
; CHECK: .cfi_offset %r15, -40
; CHECK: agfi %r15, -524168
; CHECK: .cfi_def_cfa_offset 524328
; ...main function body...
; CHECK-NOT: ag
; CHECK: lmg %r14, %r15, 524280(%r15)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add4 = add i32 %l4, %add1
  %add5 = add i32 %l5, %add4
  %add14 = add i32 %l14, %add5
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  store volatile i32 %add5, i32 *%ptr
  store volatile i32 %add14, i32 *%ptr
  %y = alloca [65499 x i64], align 8
  %entry = getelementptr inbounds [65499 x i64], [65499 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %entry
  ret void
}

; Like f1 but with a frame that is 8 bytes bigger.  This is the smallest
; frame size that needs two instructions to perform the final LMG for
; %r6 and above.
define void @f3(i32 *%ptr, i64 %x) {
; CHECK-LABEL: f3:
; CHECK: stmg %r6, %r15, 48(%r15)
; CHECK: .cfi_offset %r6, -112
; CHECK: .cfi_offset %r7, -104
; CHECK: .cfi_offset %r8, -96
; CHECK: .cfi_offset %r9, -88
; CHECK: .cfi_offset %r10, -80
; CHECK: .cfi_offset %r11, -72
; CHECK: .cfi_offset %r12, -64
; CHECK: .cfi_offset %r13, -56
; CHECK: .cfi_offset %r14, -48
; CHECK: .cfi_offset %r15, -40
; CHECK: agfi %r15, -524240
; CHECK: .cfi_def_cfa_offset 524400
; ...main function body...
; CHECK: aghi %r15, 8
; CHECK: lmg %r6, %r15, 524280(%r15)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l6 = load volatile i32, i32 *%ptr
  %l7 = load volatile i32, i32 *%ptr
  %l8 = load volatile i32, i32 *%ptr
  %l9 = load volatile i32, i32 *%ptr
  %l10 = load volatile i32, i32 *%ptr
  %l11 = load volatile i32, i32 *%ptr
  %l12 = load volatile i32, i32 *%ptr
  %l13 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add4 = add i32 %l4, %add1
  %add5 = add i32 %l5, %add4
  %add6 = add i32 %l6, %add5
  %add7 = add i32 %l7, %add6
  %add8 = add i32 %l8, %add7
  %add9 = add i32 %l9, %add8
  %add10 = add i32 %l10, %add9
  %add11 = add i32 %l11, %add10
  %add12 = add i32 %l12, %add11
  %add13 = add i32 %l13, %add12
  %add14 = add i32 %l14, %add13
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  store volatile i32 %add5, i32 *%ptr
  store volatile i32 %add6, i32 *%ptr
  store volatile i32 %add7, i32 *%ptr
  store volatile i32 %add8, i32 *%ptr
  store volatile i32 %add9, i32 *%ptr
  store volatile i32 %add10, i32 *%ptr
  store volatile i32 %add11, i32 *%ptr
  store volatile i32 %add12, i32 *%ptr
  store volatile i32 %add13, i32 *%ptr
  store volatile i32 %add14, i32 *%ptr
  %y = alloca [65508 x i64], align 8
  %entry = getelementptr inbounds [65508 x i64], [65508 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %entry
  ret void
}

; Like f2 but with a frame that is 8 bytes bigger.  This is the smallest
; frame size that needs two instructions to perform the final LMG for
; %r14 and %r15.
define void @f4(i32 *%ptr, i64 %x) {
; CHECK-LABEL: f4:
; CHECK: stmg %r14, %r15, 112(%r15)
; CHECK: .cfi_offset %r14, -48
; CHECK: .cfi_offset %r15, -40
; CHECK: agfi %r15, -524176
; CHECK: .cfi_def_cfa_offset 524336
; ...main function body...
; CHECK: aghi %r15, 8
; CHECK: lmg %r14, %r15, 524280(%r15)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add4 = add i32 %l4, %add1
  %add5 = add i32 %l5, %add4
  %add14 = add i32 %l14, %add5
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  store volatile i32 %add5, i32 *%ptr
  store volatile i32 %add14, i32 *%ptr
  %y = alloca [65500 x i64], align 8
  %entry = getelementptr inbounds [65500 x i64], [65500 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %entry
  ret void
}

; This is the largest frame size for which the preparatory increment for
; "lmg %r14, %r15, ..." can be done using AGHI.
define void @f5(i32 *%ptr, i64 %x) {
; CHECK-LABEL: f5:
; CHECK: stmg %r14, %r15, 112(%r15)
; CHECK: .cfi_offset %r14, -48
; CHECK: .cfi_offset %r15, -40
; CHECK: agfi %r15, -556928
; CHECK: .cfi_def_cfa_offset 557088
; ...main function body...
; CHECK: aghi %r15, 32760
; CHECK: lmg %r14, %r15, 524280(%r15)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add4 = add i32 %l4, %add1
  %add5 = add i32 %l5, %add4
  %add14 = add i32 %l14, %add5
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  store volatile i32 %add5, i32 *%ptr
  store volatile i32 %add14, i32 *%ptr
  %y = alloca [69594 x i64], align 8
  %entry = getelementptr inbounds [69594 x i64], [69594 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %entry
  ret void
}

; This is the smallest frame size for which the preparatory increment for
; "lmg %r14, %r15, ..." needs to be done using AGFI.
define void @f6(i32 *%ptr, i64 %x) {
; CHECK-LABEL: f6:
; CHECK: stmg %r14, %r15, 112(%r15)
; CHECK: .cfi_offset %r14, -48
; CHECK: .cfi_offset %r15, -40
; CHECK: agfi %r15, -556936
; CHECK: .cfi_def_cfa_offset 557096
; ...main function body...
; CHECK: agfi %r15, 32768
; CHECK: lmg %r14, %r15, 524280(%r15)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add4 = add i32 %l4, %add1
  %add5 = add i32 %l5, %add4
  %add14 = add i32 %l14, %add5
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  store volatile i32 %add5, i32 *%ptr
  store volatile i32 %add14, i32 *%ptr
  %y = alloca [69595 x i64], align 8
  %entry = getelementptr inbounds [69595 x i64], [69595 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %entry
  ret void
}
