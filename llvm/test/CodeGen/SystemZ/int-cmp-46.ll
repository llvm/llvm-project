; Test the use of TEST UNDER MASK for 32-bit operations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

@g = global i32 0

; Check the lowest useful TMLL value.
define void @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: tmll %r2, 1
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 1
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the high end of the TMLL range.
define void @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: tmll %r2, 65535
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 65535
  %cmp = icmp ne i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the lowest useful TMLH value, which is the next value up.
define void @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: tmlh %r2, 1
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 65536
  %cmp = icmp ne i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the next value up again, which cannot use TM.
define void @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK-NOT: {{tm[lh].}}
; CHECK: br %r14
entry:
  %and = and i32 %a, 4294901759
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the high end of the TMLH range.
define void @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: tmlh %r2, 65535
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 4294901760
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for LT comparisons that are equivalent to
; an equality comparison with zero.
define void @f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK: tmll %r2, 240
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 240
  %cmp = icmp slt i32 %and, 16
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; ...same again with LE.
define void @f7(i32 %a) {
; CHECK-LABEL: f7:
; CHECK: tmll %r2, 240
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 240
  %cmp = icmp sle i32 %and, 15
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for GE comparisons that are equivalent to
; an inequality comparison with zero.
define void @f8(i32 %a) {
; CHECK-LABEL: f8:
; CHECK: tmll %r2, 240
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 240
  %cmp = icmp uge i32 %and, 16
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; ...same again with GT.
define void @f9(i32 %a) {
; CHECK-LABEL: f9:
; CHECK: tmll %r2, 240
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 240
  %cmp = icmp ugt i32 %and, 15
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for LT comparisons that effectively
; test whether the top bit is clear.
define void @f10(i32 %a) {
; CHECK-LABEL: f10:
; CHECK: tmll %r2, 35
; CHECK: bler %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 35
  %cmp = icmp ult i32 %and, 8
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; ...same again with LE.
define void @f11(i32 %a) {
; CHECK-LABEL: f11:
; CHECK: tmll %r2, 35
; CHECK: bler %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 35
  %cmp = icmp ule i32 %and, 31
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for GE comparisons that effectively test
; whether the top bit is set.
define void @f12(i32 %a) {
; CHECK-LABEL: f12:
; CHECK: tmll %r2, 140
; CHECK: bnler %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 140
  %cmp = icmp uge i32 %and, 128
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; ...same again for GT.
define void @f13(i32 %a) {
; CHECK-LABEL: f13:
; CHECK: tmll %r2, 140
; CHECK: bnler %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 140
  %cmp = icmp ugt i32 %and, 126
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for equality comparisons with the mask.
define void @f14(i32 %a) {
; CHECK-LABEL: f14:
; CHECK: tmll %r2, 101
; CHECK: bor %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 101
  %cmp = icmp eq i32 %and, 101
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for inequality comparisons with the mask.
define void @f15(i32 %a) {
; CHECK-LABEL: f15:
; CHECK: tmll %r2, 65519
; CHECK: bnor %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 65519
  %cmp = icmp ne i32 %and, 65519
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for LT comparisons that are equivalent
; to inequality comparisons with the mask.
define void @f16(i32 %a) {
; CHECK-LABEL: f16:
; CHECK: tmll %r2, 130
; CHECK: bnor %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 130
  %cmp = icmp ult i32 %and, 129
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; ...same again with LE.
define void @f17(i32 %a) {
; CHECK-LABEL: f17:
; CHECK: tmll %r2, 130
; CHECK: bnor %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 130
  %cmp = icmp ule i32 %and, 128
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for GE comparisons that are equivalent
; to equality comparisons with the mask.
define void @f18(i32 %a) {
; CHECK-LABEL: f18:
; CHECK: tmll %r2, 194
; CHECK: bor %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 194
  %cmp = icmp uge i32 %and, 193
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; ...same again for GT.
define void @f19(i32 %a) {
; CHECK-LABEL: f19:
; CHECK: tmll %r2, 194
; CHECK: bor %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 194
  %cmp = icmp ugt i32 %and, 192
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for equality comparisons for the low bit
; when the mask has two bits.
define void @f20(i32 %a) {
; CHECK-LABEL: f20:
; CHECK: tmll %r2, 20
; CHECK: blr %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 20
  %cmp = icmp eq i32 %and, 4
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for inequality comparisons for the low bit
; when the mask has two bits.
define void @f21(i32 %a) {
; CHECK-LABEL: f21:
; CHECK: tmll %r2, 20
; CHECK: bnlr %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 20
  %cmp = icmp ne i32 %and, 4
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for equality comparisons for the high bit
; when the mask has two bits.
define void @f22(i32 %a) {
; CHECK-LABEL: f22:
; CHECK: tmll %r2, 20
; CHECK: bhr %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 20
  %cmp = icmp eq i32 %and, 16
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can use TMLL for inequality comparisons for the high bit
; when the mask has two bits.
define void @f23(i32 %a) {
; CHECK-LABEL: f23:
; CHECK: tmll %r2, 20
; CHECK: bnhr %r14
; CHECK: br %r14
entry:
  %and = and i32 %a, 20
  %cmp = icmp ne i32 %and, 16
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can fold an SHL into a TMxx mask.
define void @f24(i32 %a) {
; CHECK-LABEL: f24:
; CHECK: tmll %r2, 255
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %shl = shl i32 %a, 12
  %and = and i32 %shl, 1044480
  %cmp = icmp ne i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can fold an SHR into a TMxx mask.
define void @f25(i32 %a) {
; CHECK-LABEL: f25:
; CHECK: tmlh %r2, 512
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %shr = lshr i32 %a, 25
  %and = and i32 %shr, 1
  %cmp = icmp ne i32 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}
