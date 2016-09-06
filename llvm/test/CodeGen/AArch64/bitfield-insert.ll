; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s

; First, a simple example from Clang. The registers could plausibly be
; different, but probably won't be.

%struct.foo = type { i8, [2 x i8], i8 }

define [1 x i64] @from_clang([1 x i64] %f.coerce, i32 %n) nounwind readnone {
; CHECK-LABEL: from_clang:
; CHECK: bfi {{w[0-9]+}}, {{w[0-9]+}}, #3, #4

entry:
  %f.coerce.fca.0.extract = extractvalue [1 x i64] %f.coerce, 0
  %tmp.sroa.0.0.extract.trunc = trunc i64 %f.coerce.fca.0.extract to i32
  %bf.value = shl i32 %n, 3
  %0 = and i32 %bf.value, 120
  %f.sroa.0.0.insert.ext.masked = and i32 %tmp.sroa.0.0.extract.trunc, 135
  %1 = or i32 %f.sroa.0.0.insert.ext.masked, %0
  %f.sroa.0.0.extract.trunc = zext i32 %1 to i64
  %tmp1.sroa.1.1.insert.insert = and i64 %f.coerce.fca.0.extract, 4294967040
  %tmp1.sroa.0.0.insert.insert = or i64 %f.sroa.0.0.extract.trunc, %tmp1.sroa.1.1.insert.insert
  %.fca.0.insert = insertvalue [1 x i64] undef, i64 %tmp1.sroa.0.0.insert.insert, 0
  ret [1 x i64] %.fca.0.insert
}

define void @test_whole32(i32* %existing, i32* %new) {
; CHECK-LABEL: test_whole32:

; CHECK: bfi {{w[0-9]+}}, {{w[0-9]+}}, #26, #5

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 2214592511 ; =0x83ffffff

  %newval = load volatile i32, i32* %new
  %newval_shifted = shl i32 %newval, 26
  %newval_masked = and i32 %newval_shifted, 2080374784 ; = 0x7c000000

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing

  ret void
}

define void @test_whole64(i64* %existing, i64* %new) {
; CHECK-LABEL: test_whole64:
; CHECK: bfi {{x[0-9]+}}, {{x[0-9]+}}, #26, #14
; CHECK-NOT: and
; CHECK: ret

  %oldval = load volatile i64, i64* %existing
  %oldval_keep = and i64 %oldval, 18446742974265032703 ; = 0xffffff0003ffffffL

  %newval = load volatile i64, i64* %new
  %newval_shifted = shl i64 %newval, 26
  %newval_masked = and i64 %newval_shifted, 1099444518912 ; = 0xfffc000000

  %combined = or i64 %oldval_keep, %newval_masked
  store volatile i64 %combined, i64* %existing

  ret void
}

define void @test_whole32_from64(i64* %existing, i64* %new) {
; CHECK-LABEL: test_whole32_from64:


; CHECK: bfxil {{x[0-9]+}}, {{x[0-9]+}}, #0, #16

; CHECK: ret

  %oldval = load volatile i64, i64* %existing
  %oldval_keep = and i64 %oldval, 4294901760 ; = 0xffff0000

  %newval = load volatile i64, i64* %new
  %newval_masked = and i64 %newval, 65535 ; = 0xffff

  %combined = or i64 %oldval_keep, %newval_masked
  store volatile i64 %combined, i64* %existing

  ret void
}

define void @test_32bit_masked(i32 *%existing, i32 *%new) {
; CHECK-LABEL: test_32bit_masked:

; CHECK: and
; CHECK: bfi [[INSERT:w[0-9]+]], {{w[0-9]+}}, #3, #4

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 135 ; = 0x87

  %newval = load volatile i32, i32* %new
  %newval_shifted = shl i32 %newval, 3
  %newval_masked = and i32 %newval_shifted, 120 ; = 0x78

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing

  ret void
}

define void @test_64bit_masked(i64 *%existing, i64 *%new) {
; CHECK-LABEL: test_64bit_masked:
; CHECK: and
; CHECK: bfi [[INSERT:x[0-9]+]], {{x[0-9]+}}, #40, #8

  %oldval = load volatile i64, i64* %existing
  %oldval_keep = and i64 %oldval, 1095216660480 ; = 0xff_0000_0000

  %newval = load volatile i64, i64* %new
  %newval_shifted = shl i64 %newval, 40
  %newval_masked = and i64 %newval_shifted, 280375465082880 ; = 0xff00_0000_0000

  %combined = or i64 %newval_masked, %oldval_keep
  store volatile i64 %combined, i64* %existing

  ret void
}

; Mask is too complicated for literal ANDwwi, make sure other avenues are tried.
define void @test_32bit_complexmask(i32 *%existing, i32 *%new) {
; CHECK-LABEL: test_32bit_complexmask:

; CHECK: and
; CHECK: bfi {{w[0-9]+}}, {{w[0-9]+}}, #3, #4

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 647 ; = 0x287

  %newval = load volatile i32, i32* %new
  %newval_shifted = shl i32 %newval, 3
  %newval_masked = and i32 %newval_shifted, 120 ; = 0x278

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing

  ret void
}

; Neither mask is is a contiguous set of 1s. BFI can't be used
define void @test_32bit_badmask(i32 *%existing, i32 *%new) {
; CHECK-LABEL: test_32bit_badmask:
; CHECK-NOT: bfi
; CHECK-NOT: bfm
; CHECK: ret

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 135 ; = 0x87

  %newval = load volatile i32, i32* %new
  %newval_shifted = shl i32 %newval, 3
  %newval_masked = and i32 %newval_shifted, 632 ; = 0x278

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing

  ret void
}

; Ditto
define void @test_64bit_badmask(i64 *%existing, i64 *%new) {
; CHECK-LABEL: test_64bit_badmask:
; CHECK-NOT: bfi
; CHECK-NOT: bfm
; CHECK: ret

  %oldval = load volatile i64, i64* %existing
  %oldval_keep = and i64 %oldval, 135 ; = 0x87

  %newval = load volatile i64, i64* %new
  %newval_shifted = shl i64 %newval, 3
  %newval_masked = and i64 %newval_shifted, 664 ; = 0x278

  %combined = or i64 %oldval_keep, %newval_masked
  store volatile i64 %combined, i64* %existing

  ret void
}

; Bitfield insert where there's a left-over shr needed at the beginning
; (e.g. result of str.bf1 = str.bf2)
define void @test_32bit_with_shr(i32* %existing, i32* %new) {
; CHECK-LABEL: test_32bit_with_shr:

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 2214592511 ; =0x83ffffff

  %newval = load i32, i32* %new
  %newval_shifted = shl i32 %newval, 12
  %newval_masked = and i32 %newval_shifted, 2080374784 ; = 0x7c000000

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing
; CHECK: lsr [[BIT:w[0-9]+]], {{w[0-9]+}}, #14
; CHECK: bfi {{w[0-9]+}}, [[BIT]], #26, #5

  ret void
}

; Bitfield insert where the second or operand is a better match to be folded into the BFM
define void @test_32bit_opnd1_better(i32* %existing, i32* %new) {
; CHECK-LABEL: test_32bit_opnd1_better:

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 65535 ; 0x0000ffff

  %newval = load i32, i32* %new
  %newval_shifted = shl i32 %newval, 16
  %newval_masked = and i32 %newval_shifted, 16711680 ; 0x00ff0000

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing
; CHECK: and [[BIT:w[0-9]+]], {{w[0-9]+}}, #0xffff
; CHECK: bfi [[BIT]], {{w[0-9]+}}, #16, #8

  ret void
}

; Tests when all the bits from one operand are not useful
define i32 @test_nouseful_bits(i8 %a, i32 %b) {
; CHECK-LABEL: test_nouseful_bits:
; CHECK: bfi
; CHECK: bfi
; CHECK: bfi
; CHECK-NOT: bfi
; CHECK-NOT: or
; CHECK: lsl
  %conv = zext i8 %a to i32     ;   0  0  0  A
  %shl = shl i32 %b, 8          ;   B2 B1 B0 0
  %or = or i32 %conv, %shl      ;   B2 B1 B0 A
  %shl.1 = shl i32 %or, 8       ;   B1 B0 A 0
  %or.1 = or i32 %conv, %shl.1  ;   B1 B0 A A
  %shl.2 = shl i32 %or.1, 8     ;   B0 A A 0
  %or.2 = or i32 %conv, %shl.2  ;   B0 A A A
  %shl.3 = shl i32 %or.2, 8     ;   A A A 0
  %or.3 = or i32 %conv, %shl.3  ;   A A A A
  %shl.4 = shl i32 %or.3, 8     ;   A A A 0
  ret i32 %shl.4
}

define void @test_nouseful_strb(i32* %ptr32, i8* %ptr8, i32 %x)  {
entry:
; CHECK-LABEL: @test_nouseful_strb
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NOT:  and {{w[0-9]+}}, {{w[0-9]+}}, #0xf8
; CHECK-NEXT: bfxil [[REG1]], w2, #16, #3
; CHECK-NEXT: strb [[REG1]],
; CHECK-NEXT: ret
  %0 = load i32, i32* %ptr32, align 8
  %and = and i32 %0, -8
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 7
  %or = or i32 %and, %and1
  %trunc = trunc i32 %or to i8
  store i8 %trunc, i8* %ptr8
  ret void
}

define void @test_nouseful_strh(i32* %ptr32, i16* %ptr16, i32 %x)  {
entry:
; CHECK-LABEL: @test_nouseful_strh
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NOT:  and {{w[0-9]+}}, {{w[0-9]+}}, #0xfff0
; CHECK-NEXT: bfxil [[REG1]], w2, #16, #4
; CHECK-NEXT: strh [[REG1]],
; CHECK-NEXT: ret
  %0 = load i32, i32* %ptr32, align 8
  %and = and i32 %0, -16
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 15
  %or = or i32 %and, %and1
  %trunc = trunc i32 %or to i16
  store i16 %trunc, i16* %ptr16
  ret void
}

define void @test_nouseful_sturb(i32* %ptr32, i8* %ptr8, i32 %x)  {
entry:
; CHECK-LABEL: @test_nouseful_sturb
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NOT:  and {{w[0-9]+}}, {{w[0-9]+}}, #0xf8
; CHECK-NEXT: bfxil [[REG1]], w2, #16, #3
; CHECK-NEXT: sturb [[REG1]],
; CHECK-NEXT: ret
  %0 = load i32, i32* %ptr32, align 8
  %and = and i32 %0, -8
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 7
  %or = or i32 %and, %and1
  %trunc = trunc i32 %or to i8
  %gep = getelementptr i8, i8* %ptr8, i64 -1
  store i8 %trunc, i8* %gep
  ret void
}

define void @test_nouseful_sturh(i32* %ptr32, i16* %ptr16, i32 %x)  {
entry:
; CHECK-LABEL: @test_nouseful_sturh
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NOT:  and {{w[0-9]+}}, {{w[0-9]+}}, #0xfff0
; CHECK-NEXT: bfxil [[REG1]], w2, #16, #4
; CHECK-NEXT: sturh [[REG1]],
; CHECK-NEXT: ret
  %0 = load i32, i32* %ptr32, align 8
  %and = and i32 %0, -16
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 15
  %or = or i32 %and, %and1
  %trunc = trunc i32 %or to i16
  %gep = getelementptr i16, i16* %ptr16, i64 -1
  store i16 %trunc, i16* %gep
  ret void
}

; The next set of tests generate a BFXIL from 'or (and X, Mask0Imm),
; (and Y, Mask1Imm)' iff Mask0Imm and ~Mask1Imm are equivalent and one of the
; MaskImms is a shifted mask (e.g., 0x000ffff0).

; CHECK-LABEL: @test_or_and_and1
; CHECK: lsr w8, w1, #4
; CHECK: bfi w0, w8, #4, #12
define i32 @test_or_and_and1(i32 %a, i32 %b) {
entry:
  %and = and i32 %a, -65521 ; 0xffff000f
  %and1 = and i32 %b, 65520 ; 0x0000fff0
  %or = or i32 %and1, %and
  ret i32 %or
}

; CHECK-LABEL: @test_or_and_and2
; CHECK: lsr w8, w0, #4
; CHECK: bfi w1, w8, #4, #12
define i32 @test_or_and_and2(i32 %a, i32 %b) {
entry:
  %and = and i32 %a, 65520   ; 0x0000fff0
  %and1 = and i32 %b, -65521 ; 0xffff000f
  %or = or i32 %and1, %and
  ret i32 %or
}

; CHECK-LABEL: @test_or_and_and3
; CHECK: lsr x8, x1, #16
; CHECK: bfi x0, x8, #16, #32
define i64 @test_or_and_and3(i64 %a, i64 %b) {
entry:
  %and = and i64 %a, -281474976645121 ; 0xffff00000000ffff
  %and1 = and i64 %b, 281474976645120 ; 0x0000ffffffff0000
  %or = or i64 %and1, %and
  ret i64 %or
}

; Don't convert 'and' with multiple uses.
; CHECK-LABEL: @test_or_and_and4
; CHECK: and w8, w0, #0xffff000f
; CHECK: and w9, w1, #0xfff0
; CHECK: orr w0, w9, w8
; CHECK: str w8, [x2
define i32 @test_or_and_and4(i32 %a, i32 %b, i32* %ptr) {
entry:
  %and = and i32 %a, -65521
  store i32 %and, i32* %ptr, align 4
  %and2 = and i32 %b, 65520
  %or = or i32 %and2, %and
  ret i32 %or
}

; Don't convert 'and' with multiple uses.
; CHECK-LABEL: @test_or_and_and5
; CHECK: and w8, w1, #0xfff0
; CHECK: and w9, w0, #0xffff000f
; CHECK: orr w0, w8, w9
; CHECK: str w8, [x2]
define i32 @test_or_and_and5(i32 %a, i32 %b, i32* %ptr) {
entry:
  %and = and i32 %b, 65520
  store i32 %and, i32* %ptr, align 4
  %and1 = and i32 %a, -65521
  %or = or i32 %and, %and1
  ret i32 %or
}

; CHECK-LABEL: @test1
; CHECK: mov [[REG:w[0-9]+]], #5
; CHECK: bfxil w0, [[REG]], #0, #4
define i32 @test1(i32 %a) {
  %1 = and i32 %a, -16 ; 0xfffffff0
  %2 = or i32 %1, 5    ; 0x00000005
  ret i32 %2
}

; CHECK-LABEL: @test2
; CHECK: mov [[REG:w[0-9]+]], #10
; CHECK: bfi w0, [[REG]], #22, #4
define i32 @test2(i32 %a) {
  %1 = and i32 %a, -62914561 ; 0xfc3fffff
  %2 = or i32 %1, 41943040   ; 0x06400000
  ret i32 %2
}

; CHECK-LABEL: @test3
; CHECK: mov [[REG:x[0-9]+]], #5
; CHECK: bfxil x0, [[REG]], #0, #3
define i64 @test3(i64 %a) {
  %1 = and i64 %a, -8 ; 0xfffffffffffffff8
  %2 = or i64 %1, 5   ; 0x0000000000000005
  ret i64 %2
}

; CHECK-LABEL: @test4
; CHECK: mov [[REG:x[0-9]+]], #9
; CHECK: bfi x0, [[REG]], #1, #7
define i64 @test4(i64 %a) {
  %1 = and i64 %a, -255 ; 0xffffffffffffff01
  %2 = or i64 %1,  18   ; 0x0000000000000012
  ret i64 %2
}

; Don't generate BFI/BFXIL if the immediate can be encoded in the ORR.
; CHECK-LABEL: @test5
; CHECK: and [[REG:w[0-9]+]], w0, #0xfffffff0
; CHECK: orr w0, [[REG]], #0x6
define i32 @test5(i32 %a) {
  %1 = and i32 %a, 4294967280 ; 0xfffffff0
  %2 = or i32 %1, 6           ; 0x00000006
  ret i32 %2
}

; BFXIL will use the same constant as the ORR, so we don't care how the constant
; is materialized (it's an equal cost either way).
; CHECK-LABEL: @test6
; CHECK: mov [[REG:w[0-9]+]], #720896
; CHECK: movk [[REG]], #23250
; CHECK: bfxil w0, [[REG]], #0, #20
define i32 @test6(i32 %a) {
  %1 = and i32 %a, 4293918720 ; 0xfff00000
  %2 = or i32 %1, 744146      ; 0x000b5ad2
  ret i32 %2
}

; BFIs that require the same number of instruction to materialize the constant
; as the original ORR are okay.
; CHECK-LABEL: @test7
; CHECK: mov [[REG:w[0-9]+]], #327680
; CHECK: movk [[REG]], #44393
; CHECK: bfi w0, [[REG]], #1, #19
define i32 @test7(i32 %a) {
  %1 = and i32 %a, 4293918721 ; 0xfff00001
  %2 = or i32 %1, 744146      ; 0x000b5ad2
  ret i32 %2
}

; BFIs that require more instructions to materialize the constant as compared
; to the original ORR are not okay.  In this case we would be replacing the
; 'and' with a 'movk', which would decrease ILP while using the same number of
; instructions.
; CHECK-LABEL: @test8
; CHECK: mov [[REG2:x[0-9]+]], #157599529959424
; CHECK: and [[REG1:x[0-9]+]], x0, #0xff000000000000ff
; CHECK: movk [[REG2]], #31059, lsl #16
; CHECK: orr x0, [[REG1]], [[REG2]]
define i64 @test8(i64 %a) {
  %1 = and i64 %a, -72057594037927681 ; 0xff000000000000ff
  %2 = or i64 %1, 157601565442048     ; 0x00008f5679530000
  ret i64 %2
}

; This test exposed an issue with an overly aggressive assert.  The bit of code
; that is expected to catch this case is unable to deal with the trunc, which
; results in a failing check due to a mismatch between the BFI opcode and
; the expected value type of the OR.
; CHECK-LABEL: @test9
; CHECK: lsr x0, x0, #12
; CHECK: lsr [[REG:w[0-9]+]], w1, #23
; CHECK: bfi w0, [[REG]], #23, #9
define i32 @test9(i64 %b, i32 %e) {
  %c = lshr i64 %b, 12
  %d = trunc i64 %c to i32
  %f = and i32 %d, 8388607
  %g = and i32 %e, -8388608
  %h = or i32 %g, %f
  ret i32 %h
}
