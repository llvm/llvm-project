; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -verify-machineinstrs=true < %s | FileCheck %s
; Test these 5 bitreverse store intrinsics:
;   Q6_bitrev_store_update_D(inputLR, pDelay, nConvLength);
;   Q6_bitrev_store_update_W(inputLR, pDelay, nConvLength);
;   Q6_bitrev_store_update_HL(inputLR, pDelay, nConvLength);
;   Q6_bitrev_store_update_HH(inputLR, pDelay, nConvLength);
;   Q6_bitrev_store_update_B(inputLR, pDelay, nConvLength);
; producing these instructions:
;   memd(r0++m0:brev) = r1:0
;   memw(r0++m0:brev) = r0
;   memh(r0++m0:brev) = r3
;   memh(r0++m0:brev) = r3.h
;   memb(r0++m0:brev) = r3

; ModuleID = 'brev_st.i'
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define i64 @foo(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr2 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %sub = sub i32 13, %shr2
  %shl = shl i32 1, %sub
; CHECK: memd(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call ptr @llvm.hexagon.S2.storerd.pbr(ptr %arrayidx, i64 undef, i32 %shl)
  ret i64 0
}

declare ptr @llvm.hexagon.S2.storerd.pbr(ptr, i64, i32) nounwind

define i32 @foo1(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %sub = sub i32 14, %shr1
  %shl = shl i32 1, %sub
; CHECK: memw(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call ptr @llvm.hexagon.S2.storeri.pbr(ptr %arrayidx, i32 undef, i32 %shl)
  ret i32 0
}

declare ptr @llvm.hexagon.S2.storeri.pbr(ptr, i32, i32) nounwind

define signext i16 @foo2(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr2 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %sub = sub i32 15, %shr2
  %shl = shl i32 1, %sub
; CHECK: memh(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call ptr @llvm.hexagon.S2.storerh.pbr(ptr %arrayidx, i32 0, i32 %shl)
  ret i16 0
}

declare ptr @llvm.hexagon.S2.storerh.pbr(ptr, i32, i32) nounwind

define signext i16 @foo3(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr2 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %sub = sub i32 15, %shr2
  %shl = shl i32 1, %sub
; CHECK: memh(r{{[0-9]*}}++m{{[0-1]}}:brev) = r{{[0-9]*}}.h
  %0 = tail call ptr @llvm.hexagon.S2.storerf.pbr(ptr %arrayidx, i32 0, i32 %shl)
  ret i16 0
}

declare ptr @llvm.hexagon.S2.storerf.pbr(ptr, i32, i32) nounwind

define zeroext i8 @foo5(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %conv = zext i16 %filtMemLen to i32
  %shr2 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %sub = sub nsw i32 16, %shr2
  ; CHECK: memb(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %shl = shl i32 1, %sub
  %0 = tail call ptr @llvm.hexagon.S2.storerb.pbr(ptr %arrayidx, i32 0, i32 %shl)
  ret i8 0
}

declare ptr @llvm.hexagon.S2.storerb.pbr(ptr, i32, i32) nounwind

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
!2 = !{!"int", !0}
!3 = !{!"short", !0}
