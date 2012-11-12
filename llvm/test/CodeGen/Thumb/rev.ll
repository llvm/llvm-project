; RUN: llc < %s -march=thumb -mattr=+v6 | FileCheck %s

define i32 @test1(i32 %X) nounwind {
; CHECK: test1
; CHECK: rev16 r0, r0
        %tmp1 = lshr i32 %X, 8
        %X15 = bitcast i32 %X to i32
        %tmp4 = shl i32 %X15, 8
        %tmp2 = and i32 %tmp1, 16711680
        %tmp5 = and i32 %tmp4, -16777216
        %tmp9 = and i32 %tmp1, 255
        %tmp13 = and i32 %tmp4, 65280
        %tmp6 = or i32 %tmp5, %tmp2
        %tmp10 = or i32 %tmp6, %tmp13
        %tmp14 = or i32 %tmp10, %tmp9
        ret i32 %tmp14
}

define i32 @test2(i32 %X) nounwind {
; CHECK: test2
; CHECK: revsh r0, r0
        %tmp1 = lshr i32 %X, 8
        %tmp1.upgrd.1 = trunc i32 %tmp1 to i16
        %tmp3 = trunc i32 %X to i16
        %tmp2 = and i16 %tmp1.upgrd.1, 255
        %tmp4 = shl i16 %tmp3, 8
        %tmp5 = or i16 %tmp2, %tmp4
        %tmp5.upgrd.2 = sext i16 %tmp5 to i32
        ret i32 %tmp5.upgrd.2
}

; rdar://9147637
define i32 @test3(i16 zeroext %a) nounwind {
entry:
; CHECK: test3:
; CHECK: revsh r0, r0
  %0 = tail call i16 @llvm.bswap.i16(i16 %a)
  %1 = sext i16 %0 to i32
  ret i32 %1
}

declare i16 @llvm.bswap.i16(i16) nounwind readnone

define i32 @test4(i16 zeroext %a) nounwind {
entry:
; CHECK: test4:
; CHECK: revsh r0, r0
  %conv = zext i16 %a to i32
  %shr9 = lshr i16 %a, 8
  %conv2 = zext i16 %shr9 to i32
  %shl = shl nuw nsw i32 %conv, 8
  %or = or i32 %conv2, %shl
  %sext = shl i32 %or, 16
  %conv8 = ashr exact i32 %sext, 16
  ret i32 %conv8
}
