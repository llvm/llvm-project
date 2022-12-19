; RUN: llc -march=hexagon < %s | FileCheck %s
; Testing for these 6 variants of circular load:
;   Q6_circ_load_update_B(inputLR, pDelay, -1, nConvLength, 4);
;   Q6_circ_load_update_D(inputLR, pDelay, -1, nConvLength, 4);
;   Q6_circ_load_update_H(inputLR, pDelay, -1, nConvLength, 4);
;   Q6_circ_load_update_UB(inputLR, pDelay, -1, nConvLength, 4);
;   Q6_circ_load_update_UH(inputLR, pDelay, -1, nConvLength, 4);
;   Q6_circ_load_update_W(inputLR, pDelay, -1, nConvLength, 4);
; producing these:
;   r0 = memb(r1++#-1:circ(m0))
;   r3:2 = memd(r1++#-8:circ(m0))
;   r0 = memh(r1++#-2:circ(m0))
;   r0 = memub(r1++#-1:circ(m0))
;   r0 = memuh(r1++#-2:circ(m0))
;   r0 = memw(r1++#-4:circ(m0))

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define signext i8 @foo1(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i8, align 1
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %or = or i32 %shr1, 33554432
; CHECK: = memb(r{{[0-9]+}}++#-1:circ(m{{[0-1]}}))
  %0 = call ptr @llvm.hexagon.circ.ldb(ptr %arrayidx, ptr %inputLR, i32 %or, i32 -1)
  %1 = load i8, ptr %inputLR, align 1, !tbaa !0
  ret i8 %1
}

declare ptr @llvm.hexagon.circ.ldb(ptr, ptr, i32, i32) nounwind

define i64 @foo2(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i64, align 8
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %shl = shl nuw nsw i32 %shr1, 3
  %or = or i32 %shl, 83886080
; CHECK: = memd(r{{[0-9]+}}++#-8:circ(m{{[0-1]}}))
  %0 = call ptr @llvm.hexagon.circ.ldd(ptr %arrayidx, ptr %inputLR, i32 %or, i32 -8)
  %1 = load i64, ptr %inputLR, align 8, !tbaa !0
  ret i64 %1
}

declare ptr @llvm.hexagon.circ.ldd(ptr, ptr, i32, i32) nounwind

define signext i16 @foo3(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i16, align 2
  %conv = zext i16 %filtMemLen to i32
  %shr1 = and i32 %conv, 65534
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %or = or i32 %shr1, 50331648
; CHECK: = memh(r{{[0-9]+}}++#-2:circ(m{{[0-1]}}))
  %0 = call ptr @llvm.hexagon.circ.ldh(ptr %arrayidx, ptr %inputLR, i32 %or, i32 -2)
  %1 = load i16, ptr %inputLR, align 2, !tbaa !2
  ret i16 %1
}

declare ptr @llvm.hexagon.circ.ldh(ptr, ptr, i32, i32) nounwind

define zeroext i8 @foo4(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i8, align 1
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %or = or i32 %shr1, 33554432
; CHECK: = memub(r{{[0-9]+}}++#-1:circ(m{{[0-1]}}))
  %0 = call ptr @llvm.hexagon.circ.ldub(ptr %arrayidx, ptr %inputLR, i32 %or, i32 -1)
  %1 = load i8, ptr %inputLR, align 1, !tbaa !0
  ret i8 %1
}

declare ptr @llvm.hexagon.circ.ldub(ptr, ptr, i32, i32) nounwind

define zeroext i16 @foo5(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i16, align 2
  %conv = zext i16 %filtMemLen to i32
  %shr1 = and i32 %conv, 65534
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %or = or i32 %shr1, 50331648
; CHECK: = memuh(r{{[0-9]+}}++#-2:circ(m{{[0-1]}}))
  %0 = call ptr @llvm.hexagon.circ.lduh(ptr %arrayidx, ptr %inputLR, i32 %or, i32 -2)
  %1 = load i16, ptr %inputLR, align 2, !tbaa !2
  ret i16 %1
}

declare ptr @llvm.hexagon.circ.lduh(ptr, ptr, i32, i32) nounwind

define i32 @foo6(i16 zeroext %filtMemLen, ptr %filtMemLR, i16 signext %filtMemIndex) nounwind {
entry:
  %inputLR = alloca i32, align 4
  %conv = zext i16 %filtMemLen to i32
  %shr1 = lshr i32 %conv, 1
  %idxprom = sext i16 %filtMemIndex to i32
  %arrayidx = getelementptr inbounds i16, ptr %filtMemLR, i32 %idxprom
  %shl = shl nuw nsw i32 %shr1, 2
  %or = or i32 %shl, 67108864
; CHECK: = memw(r{{[0-9]+}}++#-4:circ(m{{[0-1]}}))
  %0 = call ptr @llvm.hexagon.circ.ldw(ptr %arrayidx, ptr %inputLR, i32 %or, i32 -4)
  %1 = load i32, ptr %inputLR, align 4, !tbaa !3
  ret i32 %1
}

declare ptr @llvm.hexagon.circ.ldw(ptr, ptr, i32, i32) nounwind

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
!2 = !{!"short", !0}
!3 = !{!"int", !0}
