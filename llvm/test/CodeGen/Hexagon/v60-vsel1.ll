; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK: if (p{{[0-3]}}) v{{[0-9]+}} = v{{[0-9]+}}

target triple = "hexagon"

; Function Attrs: nounwind
define void @fast9_detect_coarse(i8* nocapture readnone %img, i32 %xsize, i32 %stride, i32 %barrier, i32* nocapture %bitmask, i32 %boundary) #0 {
entry:
  %0 = bitcast i32* %bitmask to <16 x i32>*
  %1 = mul i32 %boundary, -2
  %sub = add i32 %1, %xsize
  %rem = and i32 %boundary, 63
  %add = add i32 %sub, %rem
  %2 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 -1)
  %3 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %4 = tail call <512 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %add)
  %5 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32> %3, <512 x i1> %4, i32 12)
  %and4 = and i32 %add, 511
  %cmp = icmp eq i32 %and4, 0
  %sMaskR.0 = select i1 %cmp, <16 x i32> %2, <16 x i32> %5
  %cmp547 = icmp sgt i32 %add, 0
  br i1 %cmp547, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %6 = tail call <512 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %boundary)
  %7 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<512 x i1> %6, i32 16843009)
  %8 = tail call <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32> %7)
  %9 = add i32 %rem, %xsize
  %10 = add i32 %9, -1
  %11 = add i32 %10, %1
  %12 = lshr i32 %11, 9
  %13 = mul i32 %12, 16
  %14 = add nuw nsw i32 %13, 16
  %scevgep = getelementptr i32, i32* %bitmask, i32 %14
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.050 = phi i32 [ %add, %for.body.lr.ph ], [ %sub6, %for.body ]
  %sMask.049 = phi <16 x i32> [ %8, %for.body.lr.ph ], [ %2, %for.body ]
  %optr.048 = phi <16 x i32>* [ %0, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %15 = tail call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> undef, <16 x i32> %sMask.049)
  %incdec.ptr = getelementptr inbounds <16 x i32>, <16 x i32>* %optr.048, i32 1
  store <16 x i32> %15, <16 x i32>* %optr.048, align 64
  %sub6 = add nsw i32 %i.050, -512
  %cmp5 = icmp sgt i32 %sub6, 0
  br i1 %cmp5, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %scevgep51 = bitcast i32* %scevgep to <16 x i32>*
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %optr.0.lcssa = phi <16 x i32>* [ %scevgep51, %for.cond.for.end_crit_edge ], [ %0, %entry ]
  %16 = load <16 x i32>, <16 x i32>* %optr.0.lcssa, align 64
  %17 = tail call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> %16, <16 x i32> %sMaskR.0)
  store <16 x i32> %17, <16 x i32>* %optr.0.lcssa, align 64
  ret void
}

declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1
declare <512 x i1> @llvm.hexagon.V6.pred.scalar2(i32) #1
declare <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32>, <512 x i1>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<512 x i1>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.vand(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
