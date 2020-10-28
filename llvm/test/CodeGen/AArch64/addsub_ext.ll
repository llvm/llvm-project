; RUN: llc -debugify-and-strip-all-safe -enable-machine-outliner=never -verify-machineinstrs %s -o - -mtriple=aarch64-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s
; RUN: llc -debugify-and-strip-all-safe -global-isel -enable-machine-outliner=never -verify-machineinstrs %s -o - -mtriple=aarch64-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s --check-prefix=GISEL

; FIXME: GISel only knows how to handle explicit G_SEXT instructions. So when
; G_SEXT is lowered to anything else, it won't fold in a stx*.
; FIXME: GISel doesn't currently handle folding the addressing mode into a cmp.

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0

define void @addsub_i8rhs() minsize {
; CHECK-LABEL: addsub_i8rhs:
; GISEL-LABEL: addsub_i8rhs:
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
; GISEL: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb

   %rhs32_zext_shift = shl i32 %rhs32_zext, 3
   %res32_zext_shift = add i32 %lhs32, %rhs32_zext_shift
   store volatile i32 %res32_zext_shift, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb #3
; GISEL: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb #3

; Zero-extending to 64-bits
    %rhs64_zext = zext i8 %val8 to i64
    %res64_zext = add i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb
; GISEL: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb

   %rhs64_zext_shift = shl i64 %rhs64_zext, 1
   %res64_zext_shift = add i64 %lhs64, %rhs64_zext_shift
   store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb #1
; GISEL: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb #1

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
; GISEL: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb

   %rhs32_zext_shift = shl i32 %rhs32_zext, 3
   %res32_zext_shift = sub i32 %lhs32, %rhs32_zext_shift
   store volatile i32 %res32_zext_shift, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb #3
; GISEL: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxtb #3

; Zero-extending to 64-bits
    %rhs64_zext = zext i8 %val8 to i64
    %res64_zext = sub i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb
; GISEL: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb

   %rhs64_zext_shift = shl i64 %rhs64_zext, 1
   %res64_zext_shift = sub i64 %lhs64, %rhs64_zext_shift
   store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb #1
; GISEL: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtb #1

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
; GISEL-LABEL: addsub_i16rhs:
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
; GISEL: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth

   %rhs32_zext_shift = shl i32 %rhs32_zext, 3
   %res32_zext_shift = add i32 %lhs32, %rhs32_zext_shift
   store volatile i32 %res32_zext_shift, i32* @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth #3
; GISEL: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth #3

; Zero-extending to 64-bits
    %rhs64_zext = zext i16 %val16 to i64
    %res64_zext = add i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth
; GISEL: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth

   %rhs64_zext_shift = shl i64 %rhs64_zext, 1
   %res64_zext_shift = add i64 %lhs64, %rhs64_zext_shift
   store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth #1
; GISEL: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth #1

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
; GISEL-LABEL: sub_i16rhs:
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
; GISEL: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth

   %rhs32_zext_shift = shl i32 %rhs32_zext, 3
   %res32_zext_shift = sub i32 %lhs32, %rhs32_zext_shift
   store volatile i32 %res32_zext_shift, i32* @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth #3
; GISEL: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, uxth #3

; Zero-extending to 64-bits
    %rhs64_zext = zext i16 %val16 to i64
    %res64_zext = sub i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth
; GISEL: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth

   %rhs64_zext_shift = shl i64 %rhs64_zext, 1
   %res64_zext_shift = sub i64 %lhs64, %rhs64_zext_shift
   store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth #1
; GISEL: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxth #1

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
define void @addsub_i32rhs(i32 %in32) minsize {
; CHECK-LABEL: addsub_i32rhs:
; GISEL-LABEL: addsub_i32rhs:
    %val32_tmp = load i32, i32* @var32
    %lhs64 = load i64, i64* @var64

    %val32 = add i32 %val32_tmp, 123

    %rhs64_zext = zext i32 %in32 to i64
    %res64_zext = add i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw
; GISEL: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw

    %rhs64_zext2 = zext i32 %val32 to i64
    %rhs64_zext_shift = shl i64 %rhs64_zext2, 2
    %res64_zext_shift = add i64 %lhs64, %rhs64_zext_shift
    store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw #2
; GISEL: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw #2

    %rhs64_sext = sext i32 %val32 to i64
    %res64_sext = add i64 %lhs64, %rhs64_sext
    store volatile i64 %res64_sext, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtw
; GISEL: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtw

    %rhs64_sext_shift = shl i64 %rhs64_sext, 2
    %res64_sext_shift = add i64 %lhs64, %rhs64_sext_shift
    store volatile i64 %res64_sext_shift, i64* @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtw #2
; GISEL: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtw #2

    ret void
}

define void @sub_i32rhs(i32 %in32) minsize {
; CHECK-LABEL: sub_i32rhs:
    %val32_tmp = load i32, i32* @var32
    %lhs64 = load i64, i64* @var64

    %val32 = add i32 %val32_tmp, 123

    %rhs64_zext = zext i32 %in32 to i64
    %res64_zext = sub i64 %lhs64, %rhs64_zext
    store volatile i64 %res64_zext, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw
; GISEL: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw

    %rhs64_zext2 = zext i32 %val32 to i64
    %rhs64_zext_shift = shl i64 %rhs64_zext2, 2
    %res64_zext_shift = sub i64 %lhs64, %rhs64_zext_shift
    store volatile i64 %res64_zext_shift, i64* @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw #2
; GISEL: sub {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, uxtw #2

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

; Check that implicit zext from w reg write is used instead of uxtw form of add.
define i64 @add_fold_uxtw(i32 %x, i64 %y) {
; CHECK-LABEL: add_fold_uxtw:
; GISEL-LABEL: add_fold_uxtw:
entry:
; CHECK: and w[[TMP:[0-9]+]], w0, #0x3
; GISEL: and w[[TMP:[0-9]+]], w0, #0x3
; FIXME: Global ISel produces an unncessary ubfx here.
  %m = and i32 %x, 3
  %ext = zext i32 %m to i64
; CHECK-NEXT: add x0, x1, x[[TMP]]
; GISEL: add x0, x1, x[[TMP]]
  %ret = add i64 %y, %ext
  ret i64 %ret
}

; Check that implicit zext from w reg write is used instead of uxtw
; form of sub and that mov WZR is folded to form a neg instruction.
define i64 @sub_fold_uxtw_xzr(i32 %x)  {
; CHECK-LABEL: sub_fold_uxtw_xzr:
; GISEL-LABEL: sub_fold_uxtw_xzr:
entry:
; CHECK: and w[[TMP:[0-9]+]], w0, #0x3
; GISEL: and w[[TMP:[0-9]+]], w0, #0x3
  %m = and i32 %x, 3
  %ext = zext i32 %m to i64
; CHECK-NEXT: neg x0, x[[TMP]]
; GISEL: neg x0, x[[TMP]]
  %ret = sub i64 0, %ext
  ret i64 %ret
}

; Check that implicit zext from w reg write is used instead of uxtw form of subs/cmp.
define i1 @cmp_fold_uxtw(i32 %x, i64 %y) {
; CHECK-LABEL: cmp_fold_uxtw:
entry:
; CHECK: and w[[TMP:[0-9]+]], w0, #0x3
  %m = and i32 %x, 3
  %ext = zext i32 %m to i64
; CHECK-NEXT: cmp x1, x[[TMP]]
; CHECK-NEXT: cset
  %ret = icmp eq i64 %y, %ext
  ret i1 %ret
}

; Check that implicit zext from w reg write is used instead of uxtw
; form of add, leading to madd selection.
define i64 @madd_fold_uxtw(i32 %x, i64 %y) {
; CHECK-LABEL: madd_fold_uxtw:
; GISEL-LABEL: madd_fold_uxtw:
entry:
; CHECK: and w[[TMP:[0-9]+]], w0, #0x3
; GISEL: and w[[TMP:[0-9]+]], w0, #0x3
  %m = and i32 %x, 3
  %ext = zext i32 %m to i64
; GISEL: madd x0, x1, x1, x[[TMP]]
; CHECK-NEXT: madd x0, x1, x1, x[[TMP]]
  %mul = mul i64 %y, %y
  %ret = add i64 %mul, %ext
  ret i64 %ret
}

; Check that implicit zext from w reg write is used instead of uxtw
; form of sub, leading to sub/cmp folding.
; Check that implicit zext from w reg write is used instead of uxtw form of subs/cmp.
define i1 @cmp_sub_fold_uxtw(i32 %x, i64 %y, i64 %z) {
; CHECK-LABEL: cmp_sub_fold_uxtw:
entry:
; CHECK: and w[[TMP:[0-9]+]], w0, #0x3
  %m = and i32 %x, 3
  %ext = zext i32 %m to i64
; CHECK-NEXT: cmp x[[TMP2:[0-9]+]], x[[TMP]]
; CHECK-NEXT: cset
  %sub = sub i64 %z, %ext
  %ret = icmp eq i64 %sub, 0
  ret i1 %ret
}

; Check that implicit zext from w reg write is used instead of uxtw
; form of add and add of -1 gets selected as sub.
define i64 @add_imm_fold_uxtw(i32 %x) {
; CHECK-LABEL: add_imm_fold_uxtw:
; GISEL-LABEL: add_imm_fold_uxtw:
entry:
; CHECK: and w[[TMP:[0-9]+]], w0, #0x3
; GISEL: and w[[TMP:[0-9]+]], w0, #0x3
  %m = and i32 %x, 3
  %ext = zext i32 %m to i64
; CHECK-NEXT: sub x0, x[[TMP]], #1
; GISEL: sub x0, x[[TMP]], #1
  %ret = add i64 %ext, -1
  ret i64 %ret
}

; Check that implicit zext from w reg write is used instead of uxtw
; form of add and add lsl form gets selected.
define i64 @add_lsl_fold_uxtw(i32 %x, i64 %y) {
; CHECK-LABEL: add_lsl_fold_uxtw:
; GISEL-LABEL: add_lsl_fold_uxtw:
entry:
; CHECK: orr w[[TMP:[0-9]+]], w0, #0x3
; GISEL: orr w[[TMP:[0-9]+]], w0, #0x3
  %m = or i32 %x, 3
  %ext = zext i32 %m to i64
  %shift = shl i64 %y, 3
; CHECK-NEXT: add x0, x[[TMP]], x1, lsl #3
; GISEL: add x0, x[[TMP]], x1, lsl #3
  %ret = add i64 %ext, %shift
  ret i64 %ret
}
