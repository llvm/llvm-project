; RUN: opt -loop-vectorize -riscv-v-vector-bits-min=128 -scalable-vectorization=on -force-target-instruction-cost=1 -S < %s | FileCheck %s

target triple = "riscv64"

define void @trip5_i8(i8* noalias nocapture noundef %dst, i8* noalias nocapture noundef readonly %src) #0 {
; CHECK-LABEL: @trip5_i8(
; CHECK:       vector.body:
; CHECK:         [[ACTIVE_LANE_MASK:%.*]] = icmp ule <vscale x 8 x i64> {{%.*}}, shufflevector (<vscale x 8 x i64> insertelement (<vscale x 8 x i64> poison, i64 4, i32 0), <vscale x 8 x i64> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK:         {{%.*}} = call <vscale x 8 x i8> @llvm.masked.load.nxv8i8.p0nxv8i8(<vscale x 8 x i8>* {{%.*}}, i32 1, <vscale x 8 x i1> [[ACTIVE_LANE_MASK]], <vscale x 8 x i8> poison)
; CHECK:         {{%.*}} = call <vscale x 8 x i8> @llvm.masked.load.nxv8i8.p0nxv8i8(<vscale x 8 x i8>* {{%.*}}, i32 1, <vscale x 8 x i1> [[ACTIVE_LANE_MASK]], <vscale x 8 x i8> poison)
; CHECK:         call void @llvm.masked.store.nxv8i8.p0nxv8i8(<vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8>* {{%.*}}, i32 1, <vscale x 8 x i1> [[ACTIVE_LANE_MASK]])
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %src, i64 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %mul = shl i8 %0, 1
  %arrayidx1 = getelementptr inbounds i8, i8* %dst, i64 %i.08
  %1 = load i8, i8* %arrayidx1, align 1
  %add = add i8 %mul, %1
  store i8 %add, i8* %arrayidx1, align 1
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, 5
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { "target-features"="+v,+d" }
