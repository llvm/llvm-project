; RUN: llc -mtriple=hexagon -O2 < %s

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
declare ptr @llvm.hexagon.circ.ldd(ptr, ptr, i32, i32) nounwind

define i32 @test(i16 zeroext %var0, ptr %var1, i16 signext %var2, ptr nocapture %var3) nounwind {
entry:
  %var4 = alloca i64, align 8
  %conv = zext i16 %var0 to i32
  %shr5 = lshr i32 %conv, 1
  %idxprom = sext i16 %var2 to i32
  %arrayidx = getelementptr inbounds i16, ptr %var1, i32 %idxprom
  %0 = load i64, ptr %var3, align 8
  %shl = shl nuw nsw i32 %shr5, 3
  %or = or i32 %shl, 83886080
  %1 = call ptr @llvm.hexagon.circ.ldd(ptr %arrayidx, ptr %var4, i32 %or, i32 -8)
  %sub = add nsw i32 %shr5, -1
  %cmp6 = icmp sgt i32 %sub, 0
  %2 = load i64, ptr %var4, align 8
  %3 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 0, i64 %0, i64 %2)
  br i1 %cmp6, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %incdec.ptr = getelementptr inbounds i16, ptr %var3, i32 4
  %4 = zext i16 %var0 to i32
  %5 = lshr i32 %4, 1
  %6 = add i32 %5, -1
  %xtraiter = urem i32 %6, 8
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
  %7 = call ptr @llvm.hexagon.circ.ldd(ptr %1, ptr %var4, i32 %or, i32 -8)
  %8 = load i64, ptr %incdec.ptr, align 8
  %inc.unr = add nsw i32 0, 1
  %incdec.ptr4.unr = getelementptr inbounds i64, ptr %incdec.ptr, i32 1
  %cmp.unr = icmp slt i32 %inc.unr, %sub
  %9 = load i64, ptr %var4, align 8
  %10 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %3, i64 %8, i64 %9)
  br label %for.body.unr13

for.body.unr13:                                   ; preds = %for.body.unr, %unr.cmp
  %11 = phi i64 [ %3, %unr.cmp ], [ %10, %for.body.unr ]
  %pvar6.09.unr = phi ptr [ %incdec.ptr, %unr.cmp ], [ %incdec.ptr4.unr, %for.body.unr ]
  %var8.0.in8.unr = phi ptr [ %1, %unr.cmp ], [ %7, %for.body.unr ]
  %i.07.unr = phi i32 [ 0, %unr.cmp ], [ %inc.unr, %for.body.unr ]
  %12 = call ptr @llvm.hexagon.circ.ldd(ptr %var8.0.in8.unr, ptr %var4, i32 %or, i32 -8)
  %13 = load i64, ptr %pvar6.09.unr, align 8
  %inc.unr14 = add nsw i32 %i.07.unr, 1
  %incdec.ptr4.unr15 = getelementptr inbounds i64, ptr %pvar6.09.unr, i32 1
  %cmp.unr16 = icmp slt i32 %inc.unr14, %sub
  %14 = load i64, ptr %var4, align 8
  %15 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %11, i64 %13, i64 %14)
  br label %for.body.unr17

for.body.unr17:                                   ; preds = %for.body.unr13, %unr.cmp24
  %16 = phi i64 [ %3, %unr.cmp24 ], [ %15, %for.body.unr13 ]
  %pvar6.09.unr18 = phi ptr [ %incdec.ptr, %unr.cmp24 ], [ %incdec.ptr4.unr15, %for.body.unr13 ]
  %var8.0.in8.unr19 = phi ptr [ %1, %unr.cmp24 ], [ %12, %for.body.unr13 ]
  %i.07.unr20 = phi i32 [ 0, %unr.cmp24 ], [ %inc.unr14, %for.body.unr13 ]
  %17 = call ptr @llvm.hexagon.circ.ldd(ptr %var8.0.in8.unr19, ptr %var4, i32 %or, i32 -8)
  %18 = load i64, ptr %pvar6.09.unr18, align 8
  %inc.unr21 = add nsw i32 %i.07.unr20, 1
  %incdec.ptr4.unr22 = getelementptr inbounds i64, ptr %pvar6.09.unr18, i32 1
  %cmp.unr23 = icmp slt i32 %inc.unr21, %sub
  %19 = load i64, ptr %var4, align 8
  %20 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %16, i64 %18, i64 %19)
  br label %for.body.unr26

for.body.unr26:                                   ; preds = %for.body.unr17, %unr.cmp33
  %21 = phi i64 [ %3, %unr.cmp33 ], [ %20, %for.body.unr17 ]
  %pvar6.09.unr27 = phi ptr [ %incdec.ptr, %unr.cmp33 ], [ %incdec.ptr4.unr22, %for.body.unr17 ]
  %var8.0.in8.unr28 = phi ptr [ %1, %unr.cmp33 ], [ %17, %for.body.unr17 ]
  %i.07.unr29 = phi i32 [ 0, %unr.cmp33 ], [ %inc.unr21, %for.body.unr17 ]
  %22 = call ptr @llvm.hexagon.circ.ldd(ptr %var8.0.in8.unr28, ptr %var4, i32 %or, i32 -8)
  %23 = load i64, ptr %pvar6.09.unr27, align 8
  %inc.unr30 = add nsw i32 %i.07.unr29, 1
  %incdec.ptr4.unr31 = getelementptr inbounds i64, ptr %pvar6.09.unr27, i32 1
  %cmp.unr32 = icmp slt i32 %inc.unr30, %sub
  %24 = load i64, ptr %var4, align 8
  %25 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %21, i64 %23, i64 %24)
  br label %for.body.unr35

for.body.unr35:                                   ; preds = %for.body.unr26, %unr.cmp42
  %26 = phi i64 [ %3, %unr.cmp42 ], [ %25, %for.body.unr26 ]
  %pvar6.09.unr36 = phi ptr [ %incdec.ptr, %unr.cmp42 ], [ %incdec.ptr4.unr31, %for.body.unr26 ]
  %var8.0.in8.unr37 = phi ptr [ %1, %unr.cmp42 ], [ %22, %for.body.unr26 ]
  %i.07.unr38 = phi i32 [ 0, %unr.cmp42 ], [ %inc.unr30, %for.body.unr26 ]
  %27 = call ptr @llvm.hexagon.circ.ldd(ptr %var8.0.in8.unr37, ptr %var4, i32 %or, i32 -8)
  %28 = load i64, ptr %pvar6.09.unr36, align 8
  %inc.unr39 = add nsw i32 %i.07.unr38, 1
  %incdec.ptr4.unr40 = getelementptr inbounds i64, ptr %pvar6.09.unr36, i32 1
  %cmp.unr41 = icmp slt i32 %inc.unr39, %sub
  %29 = load i64, ptr %var4, align 8
  %30 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %26, i64 %28, i64 %29)
  br label %for.body.unr44

for.body.unr44:                                   ; preds = %for.body.unr35, %unr.cmp51
  %31 = phi i64 [ %3, %unr.cmp51 ], [ %30, %for.body.unr35 ]
  %pvar6.09.unr45 = phi ptr [ %incdec.ptr, %unr.cmp51 ], [ %incdec.ptr4.unr40, %for.body.unr35 ]
  %var8.0.in8.unr46 = phi ptr [ %1, %unr.cmp51 ], [ %27, %for.body.unr35 ]
  %i.07.unr47 = phi i32 [ 0, %unr.cmp51 ], [ %inc.unr39, %for.body.unr35 ]
  %32 = call ptr @llvm.hexagon.circ.ldd(ptr %var8.0.in8.unr46, ptr %var4, i32 %or, i32 -8)
  %33 = load i64, ptr %pvar6.09.unr45, align 8
  %inc.unr48 = add nsw i32 %i.07.unr47, 1
  %incdec.ptr4.unr49 = getelementptr inbounds i64, ptr %pvar6.09.unr45, i32 1
  %cmp.unr50 = icmp slt i32 %inc.unr48, %sub
  %34 = load i64, ptr %var4, align 8
  %35 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %31, i64 %33, i64 %34)
  br label %for.body.unr53

for.body.unr53:                                   ; preds = %for.body.unr44, %unr.cmp60
  %36 = phi i64 [ %3, %unr.cmp60 ], [ %35, %for.body.unr44 ]
  %pvar6.09.unr54 = phi ptr [ %incdec.ptr, %unr.cmp60 ], [ %incdec.ptr4.unr49, %for.body.unr44 ]
  %var8.0.in8.unr55 = phi ptr [ %1, %unr.cmp60 ], [ %32, %for.body.unr44 ]
  %i.07.unr56 = phi i32 [ 0, %unr.cmp60 ], [ %inc.unr48, %for.body.unr44 ]
  %37 = call ptr @llvm.hexagon.circ.ldd(ptr %var8.0.in8.unr55, ptr %var4, i32 %or, i32 -8)
  %38 = load i64, ptr %pvar6.09.unr54, align 8
  %inc.unr57 = add nsw i32 %i.07.unr56, 1
  %incdec.ptr4.unr58 = getelementptr inbounds i64, ptr %pvar6.09.unr54, i32 1
  %cmp.unr59 = icmp slt i32 %inc.unr57, %sub
  %39 = load i64, ptr %var4, align 8
  %40 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %36, i64 %38, i64 %39)
  br label %for.body.lr.ph.split

for.body.lr.ph.split:                             ; preds = %for.body.unr53
  %41 = icmp ult i32 %6, 8
  br i1 %41, label %for.end.loopexit, label %for.body.lr.ph.split.split

for.body.lr.ph.split.split:                       ; preds = %for.body.lr.ph.split, %for.body.lr.ph
  %.unr = phi i64 [ %40, %for.body.lr.ph.split ], [ %3, %for.body.lr.ph ]
  %pvar6.09.unr62 = phi ptr [ %incdec.ptr4.unr58, %for.body.lr.ph.split ], [ %incdec.ptr, %for.body.lr.ph ]
  %var8.0.in8.unr63 = phi ptr [ %37, %for.body.lr.ph.split ], [ %1, %for.body.lr.ph ]
  %i.07.unr64 = phi i32 [ %inc.unr57, %for.body.lr.ph.split ], [ 0, %for.body.lr.ph ]
  %.lcssa12.unr = phi i64 [ %40, %for.body.lr.ph.split ], [ 0, %for.body.lr.ph ]
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph.split.split
  %42 = phi i64 [ %.unr, %for.body.lr.ph.split.split ], [ %74, %for.body ]
  %pvar6.09 = phi ptr [ %pvar6.09.unr62, %for.body.lr.ph.split.split ], [ %scevgep71, %for.body ]
  %var8.0.in8 = phi ptr [ %var8.0.in8.unr63, %for.body.lr.ph.split.split ], [ %71, %for.body ]
  %i.07 = phi i32 [ %i.07.unr64, %for.body.lr.ph.split.split ], [ %inc.7, %for.body ]
  %43 = call ptr @llvm.hexagon.circ.ldd(ptr %var8.0.in8, ptr %var4, i32 %or, i32 -8)
  %44 = load i64, ptr %pvar6.09, align 8
  %inc = add nsw i32 %i.07, 1
  %45 = load i64, ptr %var4, align 8
  %46 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %42, i64 %44, i64 %45)
  %47 = call ptr @llvm.hexagon.circ.ldd(ptr %43, ptr %var4, i32 %or, i32 -8)
  %scevgep = getelementptr i64, ptr %pvar6.09, i32 1
  %48 = load i64, ptr %scevgep, align 8
  %inc.1 = add nsw i32 %inc, 1
  %49 = load i64, ptr %var4, align 8
  %50 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %46, i64 %48, i64 %49)
  %51 = call ptr @llvm.hexagon.circ.ldd(ptr %47, ptr %var4, i32 %or, i32 -8)
  %scevgep65 = getelementptr i64, ptr %scevgep, i32 1
  %52 = load i64, ptr %scevgep65, align 8
  %inc.2 = add nsw i32 %inc.1, 1
  %53 = load i64, ptr %var4, align 8
  %54 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %50, i64 %52, i64 %53)
  %55 = call ptr @llvm.hexagon.circ.ldd(ptr %51, ptr %var4, i32 %or, i32 -8)
  %scevgep66 = getelementptr i64, ptr %scevgep65, i32 1
  %56 = load i64, ptr %scevgep66, align 8
  %inc.3 = add nsw i32 %inc.2, 1
  %57 = load i64, ptr %var4, align 8
  %58 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %54, i64 %56, i64 %57)
  %59 = call ptr @llvm.hexagon.circ.ldd(ptr %55, ptr %var4, i32 %or, i32 -8)
  %scevgep67 = getelementptr i64, ptr %scevgep66, i32 1
  %60 = load i64, ptr %scevgep67, align 8
  %inc.4 = add nsw i32 %inc.3, 1
  %61 = load i64, ptr %var4, align 8
  %62 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %58, i64 %60, i64 %61)
  %63 = call ptr @llvm.hexagon.circ.ldd(ptr %59, ptr %var4, i32 %or, i32 -8)
  %scevgep68 = getelementptr i64, ptr %scevgep67, i32 1
  %64 = load i64, ptr %scevgep68, align 8
  %inc.5 = add nsw i32 %inc.4, 1
  %65 = load i64, ptr %var4, align 8
  %66 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %62, i64 %64, i64 %65)
  %67 = call ptr @llvm.hexagon.circ.ldd(ptr %63, ptr %var4, i32 %or, i32 -8)
  %scevgep69 = getelementptr i64, ptr %scevgep68, i32 1
  %68 = load i64, ptr %scevgep69, align 8
  %inc.6 = add nsw i32 %inc.5, 1
  %69 = load i64, ptr %var4, align 8
  %70 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %66, i64 %68, i64 %69)
  %71 = call ptr @llvm.hexagon.circ.ldd(ptr %67, ptr %var4, i32 %or, i32 -8)
  %scevgep70 = getelementptr i64, ptr %scevgep69, i32 1
  %72 = load i64, ptr %scevgep70, align 8
  %inc.7 = add nsw i32 %inc.6, 1
  %73 = load i64, ptr %var4, align 8
  %74 = call i64 @llvm.hexagon.M2.vdmacs.s1(i64 %70, i64 %72, i64 %73)
  %cmp.7 = icmp slt i32 %inc.7, %sub
  %scevgep71 = getelementptr i64, ptr %scevgep70, i32 1
  br i1 %cmp.7, label %for.body, label %for.end.loopexit.unr-lcssa

for.end.loopexit.unr-lcssa:                       ; preds = %for.body
  %.lcssa12.ph = phi i64 [ %74, %for.body ]
  br label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.end.loopexit.unr-lcssa, %for.body.lr.ph.split
  %.lcssa12 = phi i64 [ %40, %for.body.lr.ph.split ], [ %.lcssa12.ph, %for.end.loopexit.unr-lcssa ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %.lcssa = phi i64 [ %3, %entry ], [ %.lcssa12, %for.end.loopexit ]
  %75 = call i32 @llvm.hexagon.S2.vrndpackwhs(i64 %.lcssa)
  ret i32 %75
}

declare i64 @llvm.hexagon.M2.vdmacs.s1(i64, i64, i64) nounwind readnone

declare i32 @llvm.hexagon.S2.vrndpackwhs(i64) nounwind readnone
