; RUN: llc -verify-machineinstrs %s -o - -mtriple=aarch64-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0

define void @addsub_i8rhs() minsize {
; CHECK-LABEL: addsub_i8rhs:
    %val8_tmp = load i8, i8* @var8
    %lhs32 = load i32, i32* @var32
    %lhs64 = load i64, i64* @var64

    ; Need this to prevent extension upon load and give a vanilla i8 operand.
    %val8 = add i8 %val8_tmp, 123


; Zero-extending to 32-bits
    %rhs32_zext = zext i8 %val8 to i32
    %res32_zext = add i32 %lhs32, %rhs32_zext
    store volatile i32 %res32_zext, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb

   %rhs32_zext_shift = shl i32 %rhs32_zext, 3
   %res32_zext_shift = add i32 %lhs32, %rhs32_zext_shift
   store volatile i32 %res32_zext_shift, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb #3


; Zero-extending to 64-bits
    %rhs64_zext = zext i8 %val8 to i64
    %res64_zext = add i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb

   %rhs64_zext_shift = shl i64 %rhs64_zext, 1
   %res64_zext_shift = add i64 %lhs64, %rhs64_zext_shift
   store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb #1

; Sign-extending to 32-bits
    %rhs32_sext = sext i8 %val8 to i32
    %res32_sext = add i32 %lhs32, %rhs32_sext
    store volatile i32 %res32_sext, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, sxtb

   %rhs32_sext_shift = shl i32 %rhs32_sext, 1
   %res32_sext_shift = add i32 %lhs32, %rhs32_sext_shift
   store volatile i32 %res32_sext_shift, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, sxtb #1

; Sign-extending to 64-bits
    %rhs64_sext = sext i8 %val8 to i64
    %res64_sext = add i64 %lhs64, %rhs64_sext
    store volatile i64 %res64_sext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtb

   %rhs64_sext_shift = shl i64 %rhs64_sext, 4
   %res64_sext_shift = add i64 %lhs64, %rhs64_sext_shift
   store volatile i64 %res64_sext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtb #4


; CMP variants
    %tst = icmp slt i32 %lhs32, %rhs32_zext
    br i1 %tst, label %end, label %test2
; CHECK: cmp {{w[0-9]+}}, {{w[0-9]+}}, uxtb

test2:
    %cmp_sext = sext i8 %val8 to i64
    %tst2 = icmp eq i64 %lhs64, %cmp_sext
    br i1 %tst2, label %other, label %end
; CHECK: cmp {{x[0-9]+}}, {{w[0-9]+}}, sxtb

other:
    store volatile i32 %lhs32, i32* @var32
    ret void

end:
    ret void
}

define void @sub_i8rhs() minsize {
; CHECK-LABEL: sub_i8rhs:
    %val8_tmp = load i8, i8* @var8
    %lhs32 = load i32, i32* @var32
    %lhs64 = load i64, i64* @var64

    ; Need this to prevent extension upon load and give a vanilla i8 operand.
    %val8 = add i8 %val8_tmp, 123


; Zero-extending to 32-bits
    %rhs32_zext = zext i8 %val8 to i32
    %res32_zext = sub i32 %lhs32, %rhs32_zext
    store volatile i32 %res32_zext, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb

   %rhs32_zext_shift = shl i32 %rhs32_zext, 3
   %res32_zext_shift = sub i32 %lhs32, %rhs32_zext_shift
   store volatile i32 %res32_zext_shift, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb #3


; Zero-extending to 64-bits
    %rhs64_zext = zext i8 %val8 to i64
    %res64_zext = sub i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb

   %rhs64_zext_shift = shl i64 %rhs64_zext, 1
   %res64_zext_shift = sub i64 %lhs64, %rhs64_zext_shift
   store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb #1

; Sign-extending to 32-bits
    %rhs32_sext = sext i8 %val8 to i32
    %res32_sext = sub i32 %lhs32, %rhs32_sext
    store volatile i32 %res32_sext, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, sxtb

   %rhs32_sext_shift = shl i32 %rhs32_sext, 1
   %res32_sext_shift = sub i32 %lhs32, %rhs32_sext_shift
   store volatile i32 %res32_sext_shift, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, sxtb #1

; Sign-extending to 64-bits
    %rhs64_sext = sext i8 %val8 to i64
    %res64_sext = sub i64 %lhs64, %rhs64_sext
    store volatile i64 %res64_sext, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtb

   %rhs64_sext_shift = shl i64 %rhs64_sext, 4
   %res64_sext_shift = sub i64 %lhs64, %rhs64_sext_shift
   store volatile i64 %res64_sext_shift, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtb #4

    ret void
}

define void @addsub_i16rhs() minsize {
; CHECK-LABEL: addsub_i16rhs:
    %val16_tmp = load i16, i16* @var16
    %lhs32 = load i32, i32* @var32
    %lhs64 = load i64, i64* @var64

    ; Need this to prevent extension upon load and give a vanilla i16 operand.
    %val16 = add i16 %val16_tmp, 123


; Zero-extending to 32-bits
    %rhs32_zext = zext i16 %val16 to i32
    %res32_zext = add i32 %lhs32, %rhs32_zext
    store volatile i32 %res32_zext, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth

   %rhs32_zext_shift = shl i32 %rhs32_zext, 3
   %res32_zext_shift = add i32 %lhs32, %rhs32_zext_shift
   store volatile i32 %res32_zext_shift, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth #3


; Zero-extending to 64-bits
    %rhs64_zext = zext i16 %val16 to i64
    %res64_zext = add i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth

   %rhs64_zext_shift = shl i64 %rhs64_zext, 1
   %res64_zext_shift = add i64 %lhs64, %rhs64_zext_shift
   store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth #1

; Sign-extending to 32-bits
    %rhs32_sext = sext i16 %val16 to i32
    %res32_sext = add i32 %lhs32, %rhs32_sext
    store volatile i32 %res32_sext, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, sxth

   %rhs32_sext_shift = shl i32 %rhs32_sext, 1
   %res32_sext_shift = add i32 %lhs32, %rhs32_sext_shift
   store volatile i32 %res32_sext_shift, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, sxth #1

; Sign-extending to 64-bits
    %rhs64_sext = sext i16 %val16 to i64
    %res64_sext = add i64 %lhs64, %rhs64_sext
    store volatile i64 %res64_sext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxth

   %rhs64_sext_shift = shl i64 %rhs64_sext, 4
   %res64_sext_shift = add i64 %lhs64, %rhs64_sext_shift
   store volatile i64 %res64_sext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxth #4


; CMP variants
    %tst = icmp slt i32 %lhs32, %rhs32_zext
    br i1 %tst, label %end, label %test2
; CHECK: cmp {{w[0-9]+}}, {{w[0-9]+}}, uxth

test2:
    %cmp_sext = sext i16 %val16 to i64
    %tst2 = icmp eq i64 %lhs64, %cmp_sext
    br i1 %tst2, label %other, label %end
; CHECK: cmp {{x[0-9]+}}, {{w[0-9]+}}, sxth

other:
    store volatile i32 %lhs32, i32* @var32
    ret void

end:
    ret void
}

define void @sub_i16rhs() minsize {
; CHECK-LABEL: sub_i16rhs:
    %val16_tmp = load i16, i16* @var16
    %lhs32 = load i32, i32* @var32
    %lhs64 = load i64, i64* @var64

    ; Need this to prevent extension upon load and give a vanilla i16 operand.
    %val16 = add i16 %val16_tmp, 123


; Zero-extending to 32-bits
    %rhs32_zext = zext i16 %val16 to i32
    %res32_zext = sub i32 %lhs32, %rhs32_zext
    store volatile i32 %res32_zext, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth

   %rhs32_zext_shift = shl i32 %rhs32_zext, 3
   %res32_zext_shift = sub i32 %lhs32, %rhs32_zext_shift
   store volatile i32 %res32_zext_shift, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth #3


; Zero-extending to 64-bits
    %rhs64_zext = zext i16 %val16 to i64
    %res64_zext = sub i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth

   %rhs64_zext_shift = shl i64 %rhs64_zext, 1
   %res64_zext_shift = sub i64 %lhs64, %rhs64_zext_shift
   store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth #1

; Sign-extending to 32-bits
    %rhs32_sext = sext i16 %val16 to i32
    %res32_sext = sub i32 %lhs32, %rhs32_sext
    store volatile i32 %res32_sext, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, sxth

   %rhs32_sext_shift = shl i32 %rhs32_sext, 1
   %res32_sext_shift = sub i32 %lhs32, %rhs32_sext_shift
   store volatile i32 %res32_sext_shift, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, sxth #1

; Sign-extending to 64-bits
    %rhs64_sext = sext i16 %val16 to i64
    %res64_sext = sub i64 %lhs64, %rhs64_sext
    store volatile i64 %res64_sext, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxth

   %rhs64_sext_shift = shl i64 %rhs64_sext, 4
   %res64_sext_shift = sub i64 %lhs64, %rhs64_sext_shift
   store volatile i64 %res64_sext_shift, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxth #4

    ret void
}

; N.b. we could probably check more here ("add w2, w3, w1, uxtw" for
; example), but the remaining instructions are probably not idiomatic
; in the face of "add/sub (shifted register)" so I don't intend to.
define void @addsub_i32rhs() minsize {
; CHECK-LABEL: addsub_i32rhs:
    %val32_tmp = load i32, i32* @var32
    %lhs64 = load i64, i64* @var64

    %val32 = add i32 %val32_tmp, 123

    %rhs64_zext = zext i32 %val32 to i64
    %res64_zext = add i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw

    %rhs64_zext_shift = shl i64 %rhs64_zext, 2
    %res64_zext_shift = add i64 %lhs64, %rhs64_zext_shift
    store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw #2

    %rhs64_sext = sext i32 %val32 to i64
    %res64_sext = add i64 %lhs64, %rhs64_sext
    store volatile i64 %res64_sext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtw

    %rhs64_sext_shift = shl i64 %rhs64_sext, 2
    %res64_sext_shift = add i64 %lhs64, %rhs64_sext_shift
    store volatile i64 %res64_sext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtw #2

    ret void
}

define void @sub_i32rhs() minsize {
; CHECK-LABEL: sub_i32rhs:
    %val32_tmp = load i32, i32* @var32
    %lhs64 = load i64, i64* @var64

    %val32 = add i32 %val32_tmp, 123

    %rhs64_zext = zext i32 %val32 to i64
    %res64_zext = sub i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw

    %rhs64_zext_shift = shl i64 %rhs64_zext, 2
    %res64_zext_shift = sub i64 %lhs64, %rhs64_zext_shift
    store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw #2

    %rhs64_sext = sext i32 %val32 to i64
    %res64_sext = sub i64 %lhs64, %rhs64_sext
    store volatile i64 %res64_sext, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtw

    %rhs64_sext_shift = shl i64 %rhs64_sext, 2
    %res64_sext_shift = sub i64 %lhs64, %rhs64_sext_shift
    store volatile i64 %res64_sext_shift, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtw #2

    ret void
}
