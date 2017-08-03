; bswap should be constant folded when it is passed a constant argument

; RUN: llc < %s -march=x86 -mcpu=i686 | FileCheck %s
; RUN: llc < %s -march=x86-64 | FileCheck %s --check-prefix=CHECK64

declare i16 @llvm.bswap.i16(i16)

declare i32 @llvm.bswap.i32(i32)

declare i64 @llvm.bswap.i64(i64)

define i16 @W(i16 %A) {
; CHECK-LABEL: W:
; CHECK: rolw $8, %ax

; CHECK64-LABEL: W:
; CHECK64: rolw $8, %
        %Z = call i16 @llvm.bswap.i16( i16 %A )         ; <i16> [#uses=1]
        ret i16 %Z
}

define i32 @X(i32 %A) {
; CHECK-LABEL: X:
; CHECK: bswapl %eax

; CHECK64-LABEL: X:
; CHECK64: bswapl %
        %Z = call i32 @llvm.bswap.i32( i32 %A )         ; <i32> [#uses=1]
        ret i32 %Z
}

define i64 @Y(i64 %A) {
; CHECK-LABEL: Y:
; CHECK: bswapl %eax
; CHECK: bswapl %edx

; CHECK64-LABEL: Y:
; CHECK64: bswapq %
        %Z = call i64 @llvm.bswap.i64( i64 %A )         ; <i64> [#uses=1]
        ret i64 %Z
}

; rdar://9164521
define i32 @test1(i32 %a) nounwind readnone {
entry:
; CHECK-LABEL: test1:
; CHECK: bswapl [[REG:%.*]]
; CHECK: shrl $16, [[REG]]

; CHECK64-LABEL: test1:
; CHECK64: bswapl [[REG:%.*]]
; CHECK64: shrl $16, [[REG]]
  %and = lshr i32 %a, 8
  %shr3 = and i32 %and, 255
  %and2 = shl i32 %a, 8
  %shl = and i32 %and2, 65280
  %or = or i32 %shr3, %shl
  ret i32 %or
}

define i32 @test2(i32 %a) nounwind readnone {
entry:
; CHECK-LABEL: test2:
; CHECK: bswapl [[REG:%.*]]
; CHECK: sarl $16, [[REG]]

; CHECK64-LABEL: test2:
; CHECK64: bswapl [[REG:%.*]]
; CHECK64: sarl $16, [[REG]]
  %and = lshr i32 %a, 8
  %shr4 = and i32 %and, 255
  %and2 = shl i32 %a, 8
  %or = or i32 %shr4, %and2
  %sext = shl i32 %or, 16
  %conv3 = ashr exact i32 %sext, 16
  ret i32 %conv3
}

@var8 = global i8 0
@var16 = global i16 0

; The "shl" below can move bits into the high parts of the value, so the
; operation is not a "bswap, shr" pair.

; rdar://problem/14814049
define i64 @not_bswap() {
; CHECK-LABEL: not_bswap:
; CHECK-NOT: bswapl
; CHECK: ret

; CHECK64-LABEL: not_bswap:
; CHECK64-NOT: bswapq
; CHECK64: ret
  %init = load i16, i16* @var16
  %big = zext i16 %init to i64

  %hishifted = lshr i64 %big, 8
  %loshifted = shl i64 %big, 8

  %notswapped = or i64 %hishifted, %loshifted

  ret i64 %notswapped
}

; This time, the lshr (and subsequent or) is completely useless. While it's
; technically correct to convert this into a "bswap, shr", it's suboptimal. A
; simple shl works better.

define i64 @not_useful_bswap() {
; CHECK-LABEL: not_useful_bswap:
; CHECK-NOT: bswapl
; CHECK: ret

; CHECK64-LABEL: not_useful_bswap:
; CHECK64-NOT: bswapq
; CHECK64: ret

  %init = load i8, i8* @var8
  %big = zext i8 %init to i64

  %hishifted = lshr i64 %big, 8
  %loshifted = shl i64 %big, 8

  %notswapped = or i64 %hishifted, %loshifted

  ret i64 %notswapped
}

; Finally, it *is* OK to just mask off the shl if we know that the value is zero
; beyond 16 bits anyway. This is a legitimate bswap.

define i64 @finally_useful_bswap() {
; CHECK-LABEL: finally_useful_bswap:
; CHECK: bswapl [[REG:%.*]]
; CHECK: shrl $16, [[REG]]
; CHECK: ret

; CHECK64-LABEL: finally_useful_bswap:
; CHECK64: bswapq [[REG:%.*]]
; CHECK64: shrq $48, [[REG]]
; CHECK64: ret

  %init = load i16, i16* @var16
  %big = zext i16 %init to i64

  %hishifted = lshr i64 %big, 8
  %lomasked = and i64 %big, 255
  %loshifted = shl i64 %lomasked, 8

  %swapped = or i64 %hishifted, %loshifted

  ret i64 %swapped
}

