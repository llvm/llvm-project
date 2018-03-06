; RUN: llc -march=hexagon -verify-machineinstrs=true < %s | FileCheck %s
; Testing for these 5 variants of circular store:
;   Q6_circ_store_update_B(inputLR, pDelay, -1, nConvLength, 4);
;   Q6_circ_store_update_D(inputLR, pDelay, -1, nConvLength, 4);
;   Q6_circ_store_update_HL(inputLR, pDelay, -1, nConvLength, 4);
;   Q6_circ_store_update_HH(inputLR, pDelay, -1, nConvLength, 4);
;   Q6_circ_store_update_W(inputLR, pDelay, -1, nConvLength, 4);
; producing these
;   memb(r1++#-1:circ(m0)) = r3
;   memd(r1++#-8:circ(m0)) = r1:0
;   memh(r1++#-2:circ(m0)) = r3
;   memh(r1++#-2:circ(m0)) = r3.h
;   memw(r1++#-4:circ(m0)) = r0

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define zeroext i8 @foo1(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr2 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %or = or i32 %shr2, 33554432
; CHECK: memb(r{{[0-9]+}}++#-1:circ(m{{[0-1]}}))
  %1 = tail call i8* @llvm.hexagon.circ.stb(i8* %0, i32 0, i32 %or, i32 -1)
  ret i8 0
}

declare i8* @llvm.hexagon.circ.stb(i8*, i32, i32, i32) nounwind

define i64 @foo2(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %shl = shl nuw nsw i32 %shr1, 3
  %or = or i32 %shl, 83886080
; CHECK: memd(r{{[0-9]+}}++#-8:circ(m{{[0-1]}}))
  %1 = tail call i8* @llvm.hexagon.circ.std(i8* %0, i64 undef, i32 %or, i32 -8)
  ret i64 0
}

declare i8* @llvm.hexagon.circ.std(i8*, i64, i32, i32) nounwind

define signext i16 @foo3(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr2 = and i32 %conv, 65534
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %or = or i32 %shr2, 50331648
; CHECK: memh(r{{[0-9]+}}++#-2:circ(m{{[0-1]}}))
  %1 = tail call i8* @llvm.hexagon.circ.sth(i8* %0, i32 0, i32 %or, i32 -2)
  ret i16 0
}

declare i8* @llvm.hexagon.circ.sth(i8*, i32, i32, i32) nounwind

define signext i16 @foo5(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr2 = and i32 %conv, 65534
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %or = or i32 %shr2, 50331648
; CHECK: memh(r{{[0-9]+}}++#-2:circ(m{{[0-1]}})) = r{{[0-9]*}}.h
  %1 = tail call i8* @llvm.hexagon.circ.sthhi(i8* %0, i32 0, i32 %or, i32 -2)
  ret i16 0
}

declare i8* @llvm.hexagon.circ.sthhi(i8*, i32, i32, i32) nounwind

define i32 @foo6(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %shl = shl nuw nsw i32 %shr1, 2
  %or = or i32 %shl, 67108864
; CHECK: memw(r{{[0-9]+}}++#-4:circ(m{{[0-1]}}))
  %1 = tail call i8* @llvm.hexagon.circ.stw(i8* %0, i32 undef, i32 %or, i32 -4)
  ret i32 0
}

declare i8* @llvm.hexagon.circ.stw(i8*, i32, i32, i32) nounwind

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
!2 = !{!"short", !0}
!3 = !{!"int", !0}
