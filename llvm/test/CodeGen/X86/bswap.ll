; bswap should be constant folded when it is passed a constant argument

; RUN: llc < %s -march=x86 -mcpu=i686 | FileCheck %s

declare i16 @llvm.bswap.i16(i16)

declare i32 @llvm.bswap.i32(i32)

declare i64 @llvm.bswap.i64(i64)

define i16 @W(i16 %A) {
; CHECK: W:
; CHECK: rolw $8, %ax
        %Z = call i16 @llvm.bswap.i16( i16 %A )         ; <i16> [#uses=1]
        ret i16 %Z
}

define i32 @X(i32 %A) {
; CHECK: X:
; CHECK: bswapl %eax
        %Z = call i32 @llvm.bswap.i32( i32 %A )         ; <i32> [#uses=1]
        ret i32 %Z
}

define i64 @Y(i64 %A) {
; CHECK: Y:
; CHECK: bswapl %eax
; CHECK: bswapl %edx
        %Z = call i64 @llvm.bswap.i64( i64 %A )         ; <i64> [#uses=1]
        ret i64 %Z
}

; rdar://9164521
define i32 @test1(i32 %a) nounwind readnone {
entry:
; CHECK: test1
; CHECK: bswapl %eax
; CHECK: shrl $16, %eax
  %and = lshr i32 %a, 8
  %shr3 = and i32 %and, 255
  %and2 = shl i32 %a, 8
  %shl = and i32 %and2, 65280
  %or = or i32 %shr3, %shl
  ret i32 %or
}

define i32 @test2(i32 %a) nounwind readnone {
entry:
; CHECK: test2
; CHECK: bswapl %eax
; CHECK: sarl $16, %eax
  %and = lshr i32 %a, 8
  %shr4 = and i32 %and, 255
  %and2 = shl i32 %a, 8
  %or = or i32 %shr4, %and2
  %sext = shl i32 %or, 16
  %conv3 = ashr exact i32 %sext, 16
  ret i32 %conv3
}
