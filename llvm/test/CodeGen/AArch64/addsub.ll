; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-linux-gnu | FileCheck %s

; Note that this should be refactored (for efficiency if nothing else)
; when the PCS is implemented so we don't have to worry about the
; loads and stores.

@var_i32 = global i32 42
@var2_i32 = global i32 43
@var_i64 = global i64 0

; Add pure 12-bit immediates:
define void @add_small() {
; CHECK-LABEL: add_small:

; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, #4095
  %val32 = load i32, i32* @var_i32
  %newval32 = add i32 %val32, 4095
  store i32 %newval32, i32* @var_i32

; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, #52
  %val64 = load i64, i64* @var_i64
  %newval64 = add i64 %val64, 52
  store i64 %newval64, i64* @var_i64

  ret void
}

; Make sure we grab the imm variant when the register operand
; can be implicitly zero-extend.
; We used to generate something horrible like this:
; wA = ldrb
; xB = ldimm 12
; xC = add xB, wA, uxtb
; whereas this can be achieved with:
; wA = ldrb
; xC = add xA, #12 ; <- xA implicitly zero extend wA.
define void @add_small_imm(i8* %p, i64* %q, i32 %b, i32* %addr) {
; CHECK-LABEL: add_small_imm:
entry:

; CHECK: ldrb w[[LOAD32:[0-9]+]], [x0]
  %t = load i8, i8* %p
  %promoted = zext i8 %t to i64
  %zextt = zext i8 %t to i32
  %add = add nuw i32 %zextt, %b

; CHECK: add [[ADD2:x[0-9]+]], x[[LOAD32]], #12
  %add2 = add nuw i64 %promoted, 12
  store i32 %add, i32* %addr

; CHECK: str [[ADD2]], [x1]
  store i64 %add2, i64* %q
  ret void
}

; Add 12-bit immediates, shifted left by 12 bits
define void @add_med() {
; CHECK-LABEL: add_med:

; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{#3567, lsl #12|#14610432}}
  %val32 = load i32, i32* @var_i32
  %newval32 = add i32 %val32, 14610432 ; =0xdef000
  store i32 %newval32, i32* @var_i32

; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{#4095, lsl #12|#16773120}}
  %val64 = load i64, i64* @var_i64
  %newval64 = add i64 %val64, 16773120 ; =0xfff000
  store i64 %newval64, i64* @var_i64

  ret void
}

; Subtract 12-bit immediates
define void @sub_small() {
; CHECK-LABEL: sub_small:

; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, #4095
  %val32 = load i32, i32* @var_i32
  %newval32 = sub i32 %val32, 4095
  store i32 %newval32, i32* @var_i32

; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, #52
  %val64 = load i64, i64* @var_i64
  %newval64 = sub i64 %val64, 52
  store i64 %newval64, i64* @var_i64

  ret void
}

; Subtract 12-bit immediates, shifted left by 12 bits
define void @sub_med() {
; CHECK-LABEL: sub_med:

; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{#3567, lsl #12|#14610432}}
  %val32 = load i32, i32* @var_i32
  %newval32 = sub i32 %val32, 14610432 ; =0xdef000
  store i32 %newval32, i32* @var_i32

; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{#4095, lsl #12|#16773120}}
  %val64 = load i64, i64* @var_i64
  %newval64 = sub i64 %val64, 16773120 ; =0xfff000
  store i64 %newval64, i64* @var_i64

  ret void
}

define void @testing() {
; CHECK-LABEL: testing:
  %val = load i32, i32* @var_i32
  %val2 = load i32, i32* @var2_i32

; CHECK: cmp {{w[0-9]+}}, #4095
; CHECK: b.ne [[RET:.?LBB[0-9]+_[0-9]+]]
  %cmp_pos_small = icmp ne i32 %val, 4095
  br i1 %cmp_pos_small, label %ret, label %test2

test2:
; CHECK: cmp {{w[0-9]+}}, {{#3567, lsl #12|#14610432}}
; CHECK: b.lo [[RET]]
  %newval2 = add i32 %val, 1
  store i32 %newval2, i32* @var_i32
  %cmp_pos_big = icmp ult i32 %val2, 14610432
  br i1 %cmp_pos_big, label %ret, label %test3

test3:
; CHECK: cmp {{w[0-9]+}}, #123
; CHECK: b.lt [[RET]]
  %newval3 = add i32 %val, 2
  store i32 %newval3, i32* @var_i32
  %cmp_pos_slt = icmp slt i32 %val, 123
  br i1 %cmp_pos_slt, label %ret, label %test4

test4:
; CHECK: cmp {{w[0-9]+}}, #321
; CHECK: b.gt [[RET]]
  %newval4 = add i32 %val, 3
  store i32 %newval4, i32* @var_i32
  %cmp_pos_sgt = icmp sgt i32 %val2, 321
  br i1 %cmp_pos_sgt, label %ret, label %test5

test5:
; CHECK: cmn {{w[0-9]+}}, #444
; CHECK: b.gt [[RET]]
  %newval5 = add i32 %val, 4
  store i32 %newval5, i32* @var_i32
  %cmp_neg_uge = icmp sgt i32 %val2, -444
  br i1 %cmp_neg_uge, label %ret, label %test6

test6:
  %newval6 = add i32 %val, 5
  store i32 %newval6, i32* @var_i32
  ret void

ret:
  ret void
}
; TODO: adds/subs
