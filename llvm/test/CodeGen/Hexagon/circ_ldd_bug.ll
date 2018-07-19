; RUN: llc -O2 < %s
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

; We would fail on this file with:
; Unimplemented
; UNREACHABLE executed at llvm/lib/Target/Hexagon/HexagonInstrInfo.cpp:615!
; This happened because after unrolling a loop with a ldd_circ instruction we
; would have several TFCR and ldd_circ instruction sequences.
; %0 (CRRegs) = TFCR %0 (IntRegs)
;                 = ldd_circ( , , %0)
; %1 (CRRegs) = TFCR %1 (IntRegs)
;                 = ldd_circ( , , %0)
; The scheduler would move the CRRegs to the top of the loop. The allocator
; would try to spill the CRRegs after running out of them. We don't have code to
; spill CRRegs and the above assertion would be triggered.
declare i8* @llvm.hexagon.circ.ldd(i8*, i8*, i32, i32) nounwind

define i32 @test(i16 zeroext %var0, i16* %var1, i16 signext %var2, i16* nocapture %var3) nounwind {
entry:
  %var4 = alloca i64, align 8
  %conv = zext i16 %var0 to i32
  %shr5 = lshr i32 %conv, 1
  %idxprom = sext i16 %var2 to i32
  %arrayidx = getelementptr inbounds i16, i16* %var1, i32 %idxprom
  %0 = bitcast i16* %var3 to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !1
  %2 = bitcast i16* %arrayidx to i8*
  %3 = bitcast i64* %var4 to i8*
  %shl = shl nuw nsw i32 %shr5, 3
  %or = or i32 %shl, 83886080
  %4 = call i8* @llvm.hexagon.circ.ldd(i8* %2, i8* %3, i32 %or, i32 -8)
  %sub = add nsw i32 %shr5, -1
  %cmp6 = icmp sgt i32 %sub, 0
  %5 = load i64, i64* %var4, align 8, !tbaa !1
  %6 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 0, i64 %1, i64 %5)
  br i1 %cmp6, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %incdec.ptr = getelementptr inbounds i16, i16* %var3, i32 4
  %7 = bitcast i16* %incdec.ptr to i64*
  %8 = zext i16 %var0 to i32
  %9 = lshr i32 %8, 1
  %10 = add i32 %9, -1
  %xtraiter = urem i32 %10, 8
  %lcmp = icmp ne i32 %xtraiter, 0
  br i1 %lcmp, label %unr.cmp60, label %for.body.lr.ph.split.split

unr.cmp60:                                        ; preds = %for.body.lr.ph
  %un.tmp61 = icmp eq i32 %xtraiter, 1
  br i1 %un.tmp61, label %for.body.unr53, label %unr.cmp51

unr.cmp51:                                        ; preds = %unr.cmp60
  %un.tmp52 = icmp eq i32 %xtraiter, 2
  br i1 %un.tmp52, label %for.body.unr44, label %unr.cmp42

unr.cmp42:                                        ; preds = %unr.cmp51
  %un.tmp43 = icmp eq i32 %xtraiter, 3
  br i1 %un.tmp43, label %for.body.unr35, label %unr.cmp33

unr.cmp33:                                        ; preds = %unr.cmp42
  %un.tmp34 = icmp eq i32 %xtraiter, 4
  br i1 %un.tmp34, label %for.body.unr26, label %unr.cmp24

unr.cmp24:                                        ; preds = %unr.cmp33
  %un.tmp25 = icmp eq i32 %xtraiter, 5
  br i1 %un.tmp25, label %for.body.unr17, label %unr.cmp

unr.cmp:                                          ; preds = %unr.cmp24
  %un.tmp = icmp eq i32 %xtraiter, 6
  br i1 %un.tmp, label %for.body.unr13, label %for.body.unr

for.body.unr:                                     ; preds = %unr.cmp
  %11 = call i8* @llvm.hexagon.circ.ldd(i8* %4, i8* %3, i32 %or, i32 -8)
  %12 = load i64, i64* %7, align 8, !tbaa !1
  %inc.unr = add nsw i32 0, 1
  %incdec.ptr4.unr = getelementptr inbounds i64, i64* %7, i32 1
  %cmp.unr = icmp slt i32 %inc.unr, %sub
  %13 = load i64, i64* %var4, align 8, !tbaa !1
  %14 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %6, i64 %12, i64 %13)
  br label %for.body.unr13

for.body.unr13:                                   ; preds = %for.body.unr, %unr.cmp
  %15 = phi i64 [ %6, %unr.cmp ], [ %14, %for.body.unr ]
  %pvar6.09.unr = phi i64* [ %7, %unr.cmp ], [ %incdec.ptr4.unr, %for.body.unr ]
  %var8.0.in8.unr = phi i8* [ %4, %unr.cmp ], [ %11, %for.body.unr ]
  %i.07.unr = phi i32 [ 0, %unr.cmp ], [ %inc.unr, %for.body.unr ]
  %16 = call i8* @llvm.hexagon.circ.ldd(i8* %var8.0.in8.unr, i8* %3, i32 %or, i32 -8)
  %17 = load i64, i64* %pvar6.09.unr, align 8, !tbaa !1
  %inc.unr14 = add nsw i32 %i.07.unr, 1
  %incdec.ptr4.unr15 = getelementptr inbounds i64, i64* %pvar6.09.unr, i32 1
  %cmp.unr16 = icmp slt i32 %inc.unr14, %sub
  %18 = load i64, i64* %var4, align 8, !tbaa !1
  %19 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %15, i64 %17, i64 %18)
  br label %for.body.unr17

for.body.unr17:                                   ; preds = %for.body.unr13, %unr.cmp24
  %20 = phi i64 [ %6, %unr.cmp24 ], [ %19, %for.body.unr13 ]
  %pvar6.09.unr18 = phi i64* [ %7, %unr.cmp24 ], [ %incdec.ptr4.unr15, %for.body.unr13 ]
  %var8.0.in8.unr19 = phi i8* [ %4, %unr.cmp24 ], [ %16, %for.body.unr13 ]
  %i.07.unr20 = phi i32 [ 0, %unr.cmp24 ], [ %inc.unr14, %for.body.unr13 ]
  %21 = call i8* @llvm.hexagon.circ.ldd(i8* %var8.0.in8.unr19, i8* %3, i32 %or, i32 -8)
  %22 = load i64, i64* %pvar6.09.unr18, align 8, !tbaa !1
  %inc.unr21 = add nsw i32 %i.07.unr20, 1
  %incdec.ptr4.unr22 = getelementptr inbounds i64, i64* %pvar6.09.unr18, i32 1
  %cmp.unr23 = icmp slt i32 %inc.unr21, %sub
  %23 = load i64, i64* %var4, align 8, !tbaa !1
  %24 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %20, i64 %22, i64 %23)
  br label %for.body.unr26

for.body.unr26:                                   ; preds = %for.body.unr17, %unr.cmp33
  %25 = phi i64 [ %6, %unr.cmp33 ], [ %24, %for.body.unr17 ]
  %pvar6.09.unr27 = phi i64* [ %7, %unr.cmp33 ], [ %incdec.ptr4.unr22, %for.body.unr17 ]
  %var8.0.in8.unr28 = phi i8* [ %4, %unr.cmp33 ], [ %21, %for.body.unr17 ]
  %i.07.unr29 = phi i32 [ 0, %unr.cmp33 ], [ %inc.unr21, %for.body.unr17 ]
  %26 = call i8* @llvm.hexagon.circ.ldd(i8* %var8.0.in8.unr28, i8* %3, i32 %or, i32 -8)
  %27 = load i64, i64* %pvar6.09.unr27, align 8, !tbaa !1
  %inc.unr30 = add nsw i32 %i.07.unr29, 1
  %incdec.ptr4.unr31 = getelementptr inbounds i64, i64* %pvar6.09.unr27, i32 1
  %cmp.unr32 = icmp slt i32 %inc.unr30, %sub
  %28 = load i64, i64* %var4, align 8, !tbaa !1
  %29 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %25, i64 %27, i64 %28)
  br label %for.body.unr35

for.body.unr35:                                   ; preds = %for.body.unr26, %unr.cmp42
  %30 = phi i64 [ %6, %unr.cmp42 ], [ %29, %for.body.unr26 ]
  %pvar6.09.unr36 = phi i64* [ %7, %unr.cmp42 ], [ %incdec.ptr4.unr31, %for.body.unr26 ]
  %var8.0.in8.unr37 = phi i8* [ %4, %unr.cmp42 ], [ %26, %for.body.unr26 ]
  %i.07.unr38 = phi i32 [ 0, %unr.cmp42 ], [ %inc.unr30, %for.body.unr26 ]
  %31 = call i8* @llvm.hexagon.circ.ldd(i8* %var8.0.in8.unr37, i8* %3, i32 %or, i32 -8)
  %32 = load i64, i64* %pvar6.09.unr36, align 8, !tbaa !1
  %inc.unr39 = add nsw i32 %i.07.unr38, 1
  %incdec.ptr4.unr40 = getelementptr inbounds i64, i64* %pvar6.09.unr36, i32 1
  %cmp.unr41 = icmp slt i32 %inc.unr39, %sub
  %33 = load i64, i64* %var4, align 8, !tbaa !1
  %34 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %30, i64 %32, i64 %33)
  br label %for.body.unr44

for.body.unr44:                                   ; preds = %for.body.unr35, %unr.cmp51
  %35 = phi i64 [ %6, %unr.cmp51 ], [ %34, %for.body.unr35 ]
  %pvar6.09.unr45 = phi i64* [ %7, %unr.cmp51 ], [ %incdec.ptr4.unr40, %for.body.unr35 ]
  %var8.0.in8.unr46 = phi i8* [ %4, %unr.cmp51 ], [ %31, %for.body.unr35 ]
  %i.07.unr47 = phi i32 [ 0, %unr.cmp51 ], [ %inc.unr39, %for.body.unr35 ]
  %36 = call i8* @llvm.hexagon.circ.ldd(i8* %var8.0.in8.unr46, i8* %3, i32 %or, i32 -8)
  %37 = load i64, i64* %pvar6.09.unr45, align 8, !tbaa !1
  %inc.unr48 = add nsw i32 %i.07.unr47, 1
  %incdec.ptr4.unr49 = getelementptr inbounds i64, i64* %pvar6.09.unr45, i32 1
  %cmp.unr50 = icmp slt i32 %inc.unr48, %sub
  %38 = load i64, i64* %var4, align 8, !tbaa !1
  %39 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %35, i64 %37, i64 %38)
  br label %for.body.unr53

for.body.unr53:                                   ; preds = %for.body.unr44, %unr.cmp60
  %40 = phi i64 [ %6, %unr.cmp60 ], [ %39, %for.body.unr44 ]
  %pvar6.09.unr54 = phi i64* [ %7, %unr.cmp60 ], [ %incdec.ptr4.unr49, %for.body.unr44 ]
  %var8.0.in8.unr55 = phi i8* [ %4, %unr.cmp60 ], [ %36, %for.body.unr44 ]
  %i.07.unr56 = phi i32 [ 0, %unr.cmp60 ], [ %inc.unr48, %for.body.unr44 ]
  %41 = call i8* @llvm.hexagon.circ.ldd(i8* %var8.0.in8.unr55, i8* %3, i32 %or, i32 -8)
  %42 = load i64, i64* %pvar6.09.unr54, align 8, !tbaa !1
  %inc.unr57 = add nsw i32 %i.07.unr56, 1
  %incdec.ptr4.unr58 = getelementptr inbounds i64, i64* %pvar6.09.unr54, i32 1
  %cmp.unr59 = icmp slt i32 %inc.unr57, %sub
  %43 = load i64, i64* %var4, align 8, !tbaa !1
  %44 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %40, i64 %42, i64 %43)
  br label %for.body.lr.ph.split

for.body.lr.ph.split:                             ; preds = %for.body.unr53
  %45 = icmp ult i32 %10, 8
  br i1 %45, label %for.end.loopexit, label %for.body.lr.ph.split.split

for.body.lr.ph.split.split:                       ; preds = %for.body.lr.ph.split, %for.body.lr.ph
  %.unr = phi i64 [ %44, %for.body.lr.ph.split ], [ %6, %for.body.lr.ph ]
  %pvar6.09.unr62 = phi i64* [ %incdec.ptr4.unr58, %for.body.lr.ph.split ], [ %7, %for.body.lr.ph ]
  %var8.0.in8.unr63 = phi i8* [ %41, %for.body.lr.ph.split ], [ %4, %for.body.lr.ph ]
  %i.07.unr64 = phi i32 [ %inc.unr57, %for.body.lr.ph.split ], [ 0, %for.body.lr.ph ]
  %.lcssa12.unr = phi i64 [ %44, %for.body.lr.ph.split ], [ 0, %for.body.lr.ph ]
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph.split.split
  %46 = phi i64 [ %.unr, %for.body.lr.ph.split.split ], [ %78, %for.body ]
  %pvar6.09 = phi i64* [ %pvar6.09.unr62, %for.body.lr.ph.split.split ], [ %scevgep71, %for.body ]
  %var8.0.in8 = phi i8* [ %var8.0.in8.unr63, %for.body.lr.ph.split.split ], [ %75, %for.body ]
  %i.07 = phi i32 [ %i.07.unr64, %for.body.lr.ph.split.split ], [ %inc.7, %for.body ]
  %47 = call i8* @llvm.hexagon.circ.ldd(i8* %var8.0.in8, i8* %3, i32 %or, i32 -8)
  %48 = load i64, i64* %pvar6.09, align 8, !tbaa !1
  %inc = add nsw i32 %i.07, 1
  %49 = load i64, i64* %var4, align 8, !tbaa !1
  %50 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %46, i64 %48, i64 %49)
  %51 = call i8* @llvm.hexagon.circ.ldd(i8* %47, i8* %3, i32 %or, i32 -8)
  %scevgep = getelementptr i64, i64* %pvar6.09, i32 1
  %52 = load i64, i64* %scevgep, align 8, !tbaa !1
  %inc.1 = add nsw i32 %inc, 1
  %53 = load i64, i64* %var4, align 8, !tbaa !1
  %54 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %50, i64 %52, i64 %53)
  %55 = call i8* @llvm.hexagon.circ.ldd(i8* %51, i8* %3, i32 %or, i32 -8)
  %scevgep65 = getelementptr i64, i64* %scevgep, i32 1
  %56 = load i64, i64* %scevgep65, align 8, !tbaa !1
  %inc.2 = add nsw i32 %inc.1, 1
  %57 = load i64, i64* %var4, align 8, !tbaa !1
  %58 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %54, i64 %56, i64 %57)
  %59 = call i8* @llvm.hexagon.circ.ldd(i8* %55, i8* %3, i32 %or, i32 -8)
  %scevgep66 = getelementptr i64, i64* %scevgep65, i32 1
  %60 = load i64, i64* %scevgep66, align 8, !tbaa !1
  %inc.3 = add nsw i32 %inc.2, 1
  %61 = load i64, i64* %var4, align 8, !tbaa !1
  %62 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %58, i64 %60, i64 %61)
  %63 = call i8* @llvm.hexagon.circ.ldd(i8* %59, i8* %3, i32 %or, i32 -8)
  %scevgep67 = getelementptr i64, i64* %scevgep66, i32 1
  %64 = load i64, i64* %scevgep67, align 8, !tbaa !1
  %inc.4 = add nsw i32 %inc.3, 1
  %65 = load i64, i64* %var4, align 8, !tbaa !1
  %66 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %62, i64 %64, i64 %65)
  %67 = call i8* @llvm.hexagon.circ.ldd(i8* %63, i8* %3, i32 %or, i32 -8)
  %scevgep68 = getelementptr i64, i64* %scevgep67, i32 1
  %68 = load i64, i64* %scevgep68, align 8, !tbaa !1
  %inc.5 = add nsw i32 %inc.4, 1
  %69 = load i64, i64* %var4, align 8, !tbaa !1
  %70 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %66, i64 %68, i64 %69)
  %71 = call i8* @llvm.hexagon.circ.ldd(i8* %67, i8* %3, i32 %or, i32 -8)
  %scevgep69 = getelementptr i64, i64* %scevgep68, i32 1
  %72 = load i64, i64* %scevgep69, align 8, !tbaa !1
  %inc.6 = add nsw i32 %inc.5, 1
  %73 = load i64, i64* %var4, align 8, !tbaa !1
  %74 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %70, i64 %72, i64 %73)
  %75 = call i8* @llvm.hexagon.circ.ldd(i8* %71, i8* %3, i32 %or, i32 -8)
  %scevgep70 = getelementptr i64, i64* %scevgep69, i32 1
  %76 = load i64, i64* %scevgep70, align 8, !tbaa !1
  %inc.7 = add nsw i32 %inc.6, 1
  %77 = load i64, i64* %var4, align 8, !tbaa !1
  %78 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %74, i64 %76, i64 %77)
  %cmp.7 = icmp slt i32 %inc.7, %sub
  %scevgep71 = getelementptr i64, i64* %scevgep70, i32 1
  br i1 %cmp.7, label %for.body, label %for.end.loopexit.unr-lcssa

for.end.loopexit.unr-lcssa:                       ; preds = %for.body
  %.lcssa12.ph = phi i64 [ %78, %for.body ]
  br label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.end.loopexit.unr-lcssa, %for.body.lr.ph.split
  %.lcssa12 = phi i64 [ %44, %for.body.lr.ph.split ], [ %.lcssa12.ph, %for.end.loopexit.unr-lcssa ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %.lcssa = phi i64 [ %6, %entry ], [ %.lcssa12, %for.end.loopexit ]
  %79 = call i32 @llvm.hexagon.S2.vrndpackwhs(i64 %.lcssa)
  ret i32 %79
}

declare i64 @llvm.hexagon.M2.vdmacs.s1(i64, i64, i64) nounwind readnone

declare i32 @llvm.hexagon.S2.vrndpackwhs(i64) nounwind readnone

!0 = !{!"long long", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
