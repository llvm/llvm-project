; Test that compares are omitted if CC already has the right value
; (z10 version).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -no-integrated-as \
; RUN:   -verify-machineinstrs| FileCheck %s

declare void @foo()

; Addition provides enough for comparisons with zero if we know no
; signed overflow happens, which is when the "nsw" flag is set.
; First test the EQ case.
define i32 @f1(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f1:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %res = add nsw i32 %a, 1000000
  %cmp = icmp eq i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and again with NE.
define i32 @f2(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f2:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %res = add nsw i32 %a, 1000000
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and again with SLT.
define i32 @f3(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f3:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: blr %r14
entry:
  %res = add nsw i32 %a, 1000000
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and again with SLE.
define i32 @f4(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f4:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: bler %r14
entry:
  %res = add nsw i32 %a, 1000000
  %cmp = icmp sle i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and again with SGT.
define i32 @f5(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f5:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: bhr %r14
entry:
  %res = add nsw i32 %a, 1000000
  %cmp = icmp sgt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and again with SGE.
define i32 @f6(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f6:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: bher %r14
entry:
  %res = add nsw i32 %a, 1000000
  %cmp = icmp sge i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; Subtraction provides in addition also enough for equality comparisons with
; zero even without "nsw".
define i32 @f7(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f7:
; CHECK: s %r2, 0(%r4)
; CHECK-NEXT: bner %r14
; CHECK: br %r14
entry:
  %cur = load i32, ptr %dest
  %res = sub i32 %a, %cur
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and again with SLT.
define i32 @f8(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f8:
; CHECK: s %r2, 0(%r4)
; CHECK-NEXT: blr %r14
entry:
  %cur = load i32, ptr %dest
  %res = sub nsw i32 %a, %cur
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; Logic register-register instructions also provide enough for equality
; comparisons with zero.
define i32 @f9(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f9:
; CHECK: nr %r2, %r3
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = and i32 %a, %b
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...but not for ordered comparisons.
define i32 @f10(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f10:
; CHECK: nr %r2, %r3
; CHECK-NEXT: cibl %r2, 0, 0(%r14)
; CHECK: br %r14
entry:
  %res = and i32 %a, %b
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; Logic register-immediate instructions also provide enough for equality
; comparisons with zero if the immediate covers the whole register.
define i32 @f11(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f11:
; CHECK: nilf %r2, 100000001
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = and i32 %a, 100000001
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; Partial logic register-immediate instructions do not provide simple
; zero results.
define i32 @f12(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f12:
; CHECK: nill %r2, 65436
; CHECK-NEXT: ciblh %r2, 0, 0(%r14)
; CHECK: br %r14
entry:
  %res = and i32 %a, -100
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; SRA provides the same CC result as a comparison with zero.
define i32 @f13(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f13:
; CHECK: sra %r2, 0(%r3)
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %res = ashr i32 %a, %b
  %cmp = icmp eq i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and again with NE.
define i32 @f14(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f14:
; CHECK: sra %r2, 0(%r3)
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %res = ashr i32 %a, %b
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and SLT.
define i32 @f15(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f15:
; CHECK: sra %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = ashr i32 %a, %b
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and SLE.
define i32 @f16(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f16:
; CHECK: sra %r2, 0(%r3)
; CHECK-NEXT: bler %r14
; CHECK: br %r14
entry:
  %res = ashr i32 %a, %b
  %cmp = icmp sle i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and SGT.
define i32 @f17(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f17:
; CHECK: sra %r2, 0(%r3)
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %res = ashr i32 %a, %b
  %cmp = icmp sgt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and SGE.
define i32 @f18(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f18:
; CHECK: sra %r2, 0(%r3)
; CHECK-NEXT: bher %r14
; CHECK: br %r14
entry:
  %res = ashr i32 %a, %b
  %cmp = icmp sge i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; RISBG provides the same result as a comparison against zero.
; Test the EQ case.
define i64 @f19(i64 %a, i64 %b, ptr %dest) {
; CHECK-LABEL: f19:
; CHECK: risbg %r2, %r3, 0, 190, 0
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %res = and i64 %b, -2
  %cmp = icmp eq i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %b, ptr %dest
  br label %exit

exit:
  ret i64 %res
}

; ...and the SLT case.
define i64 @f20(i64 %a, i64 %b, ptr %dest) {
; CHECK-LABEL: f20:
; CHECK: risbg %r2, %r3, 0, 190, 0
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = and i64 %b, -2
  %cmp = icmp slt i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %b, ptr %dest
  br label %exit

exit:
  ret i64 %res
}

; Test a case where the register we're testing is set by a non-CC-clobbering
; instruction.
define i32 @f21(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f21:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: cibe %r2, 0, 0(%r14)
; CHECK: br %r14
entry:
  %add = add i32 %a, 1000000
  %res = call i32 asm "blah $0", "=r,0" (i32 %add)
  %cmp = icmp eq i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; ...and again with a CC-clobbering instruction.
define i32 @f22(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f22:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: cibe %r2, 0, 0(%r14)
; CHECK: br %r14
entry:
  %add = add i32 %a, 1000000
  %res = call i32 asm "blah $0", "=r,0,~{cc}" (i32 %add)
  %cmp = icmp eq i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; Check that stores do not interfere.
define i32 @f23(i32 %a, i32 %b, ptr %dest1, ptr %dest2) {
; CHECK-LABEL: f23:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: st %r2, 0(%r4)
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %res = add nsw i32 %a, 1000000
  store i32 %res, ptr %dest1
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest2
  br label %exit

exit:
  ret i32 %res
}

; Check that calls do interfere.
define void @f24(ptr %ptr) {
; CHECK-LABEL: f24:
; CHECK: afi [[REG:%r[0-9]+]], 1000000
; CHECK-NEXT: brasl %r14, foo@PLT
; CHECK-NEXT: cijlh [[REG]], 0, .L{{.*}}
; CHECK: br %r14
entry:
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  %add = add i32 %xor, 1000000
  call void @foo()
  %cmp = icmp eq i32 %add, 0
  br i1 %cmp, label %store, label %exit, !prof !1

store:
  store i32 %add, ptr %ptr
  br label %exit

exit:
  ret void
}

; Check that inline asms don't interfere if they don't clobber CC.
define void @f25(i32 %a, ptr %ptr) {
; CHECK-LABEL: f25:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: #APP
; CHECK-NEXT: blah
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %add = add nsw i32 %a, 1000000
  call void asm sideeffect "blah", "r"(i32 %add)
  %cmp = icmp ne i32 %add, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %add, ptr %ptr
  br label %exit

exit:
  ret void
}

; ...but do interfere if they do clobber CC.
define void @f26(i32 %a, ptr %ptr) {
; CHECK-LABEL: f26:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: #APP
; CHECK-NEXT: blah
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: ciblh %r2, 0, 0(%r14)
; CHECK: br %r14
entry:
  %add = add i32 %a, 1000000
  call void asm sideeffect "blah", "r,~{cc}"(i32 %add)
  %cmp = icmp ne i32 %add, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %add, ptr %ptr
  br label %exit

exit:
  ret void
}

; Test a case where CC is set based on a different register from the
; compare input.
define i32 @f27(i32 %a, i32 %b, ptr %dest1, ptr %dest2) {
; CHECK-LABEL: f27:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: sr %r3, %r2
; CHECK-NEXT: st %r3, 0(%r4)
; CHECK-NEXT: cibe %r2, 0, 0(%r14)
; CHECK: br %r14
entry:
  %add = add nsw i32 %a, 1000000
  %sub = sub i32 %b, %add
  store i32 %sub, ptr %dest1
  %cmp = icmp eq i32 %add, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %sub, ptr %dest2
  br label %exit

exit:
  ret i32 %add
}

; Make sure that we don't confuse a base register for a destination.
define void @f28(i64 %a, ptr %dest) {
; CHECK-LABEL: f28:
; CHECK: xi 0(%r2), 15
; CHECK: cgibe %r2, 0, 0(%r14)
; CHECK: br %r14
entry:
  %ptr = inttoptr i64 %a to ptr
  %val = load i8, ptr %ptr
  %xor = xor i8 %val, 15
  store i8 %xor, ptr %ptr
  %cmp = icmp eq i64 %a, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %a, ptr %dest
  br label %exit

exit:
  ret void
}

; Test that L gets converted to LT where useful.
define i32 @f29(i64 %base, i64 %index, ptr %dest) {
; CHECK-LABEL: f29:
; CHECK: lt %r2, 0({{%r2,%r3|%r3,%r2}})
; CHECK-NEXT: bler %r14
; CHECK: br %r14
entry:
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to ptr
  %res = load i32, ptr %ptr
  %cmp = icmp sle i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %res, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; Test that LY gets converted to LT where useful.
define i32 @f30(i64 %base, i64 %index, ptr %dest) {
; CHECK-LABEL: f30:
; CHECK: lt %r2, 100000({{%r2,%r3|%r3,%r2}})
; CHECK-NEXT: bler %r14
; CHECK: br %r14
entry:
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 100000
  %ptr = inttoptr i64 %add2 to ptr
  %res = load i32, ptr %ptr
  %cmp = icmp sle i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %res, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; Test that LG gets converted to LTG where useful.
define i64 @f31(i64 %base, i64 %index, ptr %dest) {
; CHECK-LABEL: f31:
; CHECK: ltg %r2, 0({{%r2,%r3|%r3,%r2}})
; CHECK-NEXT: bher %r14
; CHECK: br %r14
entry:
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to ptr
  %res = load i64, ptr %ptr
  %cmp = icmp sge i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %res, ptr %dest
  br label %exit

exit:
  ret i64 %res
}

; Test that LGF gets converted to LTGF where useful.
define i64 @f32(i64 %base, i64 %index, ptr %dest) {
; CHECK-LABEL: f32:
; CHECK: ltgf %r2, 0({{%r2,%r3|%r3,%r2}})
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to ptr
  %val = load i32, ptr %ptr
  %res = sext i32 %val to i64
  %cmp = icmp sgt i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %res, ptr %dest
  br label %exit

exit:
  ret i64 %res
}

; Test that LR gets converted to LTR where useful.
define i32 @f33(i32 %dummy, i32 %val, ptr %dest) {
; CHECK-LABEL: f33:
; CHECK: ltr %r2, %r3
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  call void asm sideeffect "blah $0", "{r2}"(i32 %val)
  %cmp = icmp slt i32 %val, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %val, ptr %dest
  br label %exit

exit:
  ret i32 %val
}

; Test that LGR gets converted to LTGR where useful.
define i64 @f34(i64 %dummy, i64 %val, ptr %dest) {
; CHECK-LABEL: f34:
; CHECK: ltgr %r2, %r3
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  call void asm sideeffect "blah $0", "{r2}"(i64 %val)
  %cmp = icmp sgt i64 %val, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %val, ptr %dest
  br label %exit

exit:
  ret i64 %val
}

; Test that LGFR gets converted to LTGFR where useful.
define i64 @f35(i64 %dummy, i32 %val, ptr %dest) {
; CHECK-LABEL: f35:
; CHECK: ltgfr %r2, %r3
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %ext = sext i32 %val to i64
  call void asm sideeffect "blah $0", "{r2}"(i64 %ext)
  %cmp = icmp sgt i64 %ext, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %ext, ptr %dest
  br label %exit

exit:
  ret i64 %ext
}

; Test a case where it is the source rather than destination of LR that
; we need.
define i32 @f36(i32 %val, i32 %dummy, ptr %dest) {
; CHECK-LABEL: f36:
; CHECK: ltr %r3, %r2
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r3
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  call void asm sideeffect "blah $0", "{r3}"(i32 %val)
  %cmp = icmp slt i32 %val, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %val, ptr %dest
  br label %exit

exit:
  ret i32 %val
}

; Test a case where it is the source rather than destination of LGR that
; we need.
define i64 @f37(i64 %val, i64 %dummy, ptr %dest) {
; CHECK-LABEL: f37:
; CHECK: ltgr %r3, %r2
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r3
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  call void asm sideeffect "blah $0", "{r3}"(i64 %val)
  %cmp = icmp slt i64 %val, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %val, ptr %dest
  br label %exit

exit:
  ret i64 %val
}

; Test a case where it is the source rather than destination of LGFR that
; we need.
define i32 @f38(i32 %val, i64 %dummy, ptr %dest) {
; CHECK-LABEL: f38:
; CHECK: ltgfr %r3, %r2
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r3
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %ext = sext i32 %val to i64
  call void asm sideeffect "blah $0", "{r3}"(i64 %ext)
  %cmp = icmp slt i32 %val, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 %val, ptr %dest
  br label %exit

exit:
  ret i32 %val
}

; Test f35 for in-register extensions.
define i64 @f39(i64 %dummy, i64 %a, ptr %dest) {
; CHECK-LABEL: f39:
; CHECK: ltgfr %r2, %r3
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %val = trunc i64 %a to i32
  %ext = sext i32 %val to i64
  call void asm sideeffect "blah $0", "{r2}"(i64 %ext)
  %cmp = icmp sgt i64 %ext, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %ext, ptr %dest
  br label %exit

exit:
  ret i64 %ext
}

; ...and again with what InstCombine would produce for f40.
define i64 @f40(i64 %dummy, i64 %a, ptr %dest) {
; CHECK-LABEL: f40:
; CHECK: ltgfr %r2, %r3
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %r2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %shl = shl i64 %a, 32
  %ext = ashr i64 %shl, 32
  call void asm sideeffect "blah $0", "{r2}"(i64 %ext)
  %cmp = icmp sgt i64 %shl, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %ext, ptr %dest
  br label %exit

exit:
  ret i64 %ext
}

; Try a form of f7 in which the subtraction operands are compared directly.
define i32 @f41(i32 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f41:
; CHECK: s %r2, 0(%r4)
; CHECK-NEXT: bner %r14
; CHECK: br %r14
entry:
  %cur = load i32, ptr %dest
  %res = sub i32 %a, %cur
  %cmp = icmp ne i32 %a, %cur
  br i1 %cmp, label %exit, label %store

store:
  store i32 %b, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; A version of f32 that tests the unextended value.
define i64 @f42(i64 %base, i64 %index, ptr %dest) {
; CHECK-LABEL: f42:
; CHECK: ltgf %r2, 0({{%r2,%r3|%r3,%r2}})
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to ptr
  %val = load i32, ptr %ptr
  %res = sext i32 %val to i64
  %cmp = icmp sgt i32 %val, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 %res, ptr %dest
  br label %exit

exit:
  ret i64 %res
}

!1 = !{!"branch_weights", i32 2, i32 1}
