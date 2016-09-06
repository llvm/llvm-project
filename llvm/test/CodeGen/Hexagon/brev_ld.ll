; RUN: llc -march=hexagon < %s | FileCheck %s
; RUN: llc -march=hexagon -verify-machineinstrs=true < %s | FileCheck %s
; Testing bitreverse load intrinsics:
;   Q6_bitrev_load_update_D(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_W(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_H(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_UH(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_UB(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_B(inputLR, pDelay, nConvLength);
; producing these instructions:
;   r3:2 = memd(r0++m0:brev)
;   r1 = memw(r0++m0:brev)
;   r1 = memh(r0++m0:brev)
;   r1 = memuh(r0++m0:brev)
;   r1 = memub(r0++m0:brev)
;   r1 = memb(r0++m0:brev)

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define i64 @foo(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i64, align 8
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %1 = bitcast i64* %inputLR to i8*
  %sub = sub i32 13, %shr1
  %shl = shl i32 1, %sub
; CHECK: = memd(r{{[0-9]*}} ++ m{{[0-1]}}:brev)
  %2 = call i8* @llvm.hexagon.brev.ldd(i8* %0, i8* %1, i32 %shl)
  %3 = bitcast i8* %1 to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !0
  ret i64 %4
}

declare i8* @llvm.hexagon.brev.ldd(i8*, i8*, i32) nounwind

define i32 @foo1(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i32, align 4
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %1 = bitcast i32* %inputLR to i8*
  %sub = sub i32 14, %shr1
  %shl = shl i32 1, %sub
; CHECK: = memw(r{{[0-9]*}} ++ m{{[0-1]}}:brev)
  %2 = call i8* @llvm.hexagon.brev.ldw(i8* %0, i8* %1, i32 %shl)
  %3 = bitcast i8* %1 to i32*
  %4 = load i32, i32* %3, align 4, !tbaa !2
  ret i32 %4
}

declare i8* @llvm.hexagon.brev.ldw(i8*, i8*, i32) nounwind

define signext i16 @foo2(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i16, align 2
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %1 = bitcast i16* %inputLR to i8*
  %sub = sub i32 15, %shr1
  %shl = shl i32 1, %sub
; CHECK: = memh(r{{[0-9]*}} ++ m0:brev)
  %2 = call i8* @llvm.hexagon.brev.ldh(i8* %0, i8* %1, i32 %shl)
  %3 = bitcast i8* %1 to i16*
  %4 = load i16, i16* %3, align 2, !tbaa !3
  ret i16 %4
}

declare i8* @llvm.hexagon.brev.ldh(i8*, i8*, i32) nounwind

define zeroext i16 @foo3(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i16, align 2
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %1 = bitcast i16* %inputLR to i8*
  %sub = sub i32 15, %shr1
  %shl = shl i32 1, %sub
; CHECK: = memuh(r{{[0-9]*}} ++ m0:brev)
  %2 = call i8* @llvm.hexagon.brev.lduh(i8* %0, i8* %1, i32 %shl)
  %3 = bitcast i8* %1 to i16*
  %4 = load i16, i16* %3, align 2, !tbaa !3
  ret i16 %4
}

declare i8* @llvm.hexagon.brev.lduh(i8*, i8*, i32) nounwind

define zeroext i8 @foo4(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i8, align 1
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %sub = sub nsw i32 16, %shr1
  %shl = shl i32 1, %sub
; CHECK: = memub(r{{[0-9]*}} ++ m{{[0-1]}}:brev)
  %1 = call i8* @llvm.hexagon.brev.ldub(i8* %0, i8* %inputLR, i32 %shl)
  %2 = load i8, i8* %inputLR, align 1, !tbaa !0
  ret i8 %2
}

declare i8* @llvm.hexagon.brev.ldub(i8*, i8*, i32) nounwind

define signext i8 @foo5(i16 zeroext %filtMemLen, i16* %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i8, align 1
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, i16* %filtMemLR, i32 %idxprom
  %0 = bitcast i16* %arrayidx to i8*
  %sub = sub nsw i32 16, %shr1
  %shl = shl i32 1, %sub
; CHECK: = memb(r{{[0-9]*}} ++ m{{[0-1]}}:brev)
  %1 = call i8* @llvm.hexagon.brev.ldb(i8* %0, i8* %inputLR, i32 %shl)
  %2 = load i8, i8* %inputLR, align 1, !tbaa !0
  ret i8 %2
}

declare i8* @llvm.hexagon.brev.ldb(i8*, i8*, i32) nounwind

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
!2 = !{!"int", !0}
!3 = !{!"short", !0}
