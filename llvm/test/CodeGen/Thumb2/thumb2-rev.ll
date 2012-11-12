; RUN: llc < %s -march=thumb -mattr=+thumb2,+v7,+t2xtpk | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: rev r0, r0
    %tmp = tail call i32 @llvm.bswap.i32(i32 %a)
    ret i32 %tmp
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone

define i32 @f2(i32 %X) {
; CHECK: f2:
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
