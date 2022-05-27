; This is the loop in c++ being vectorize in this file with
;experimental.vector.reverse
;  #pragma clang loop vectorize_width(4, scalable)
;  for (int i = N-1; i >= 0; --i)
;    a[i] = b[i] + 1.0;

; REQUIRES: asserts
; RUN: opt -loop-vectorize -dce -instcombine -mtriple riscv64-linux-gnu \
; RUN:   -mattr=+v -debug-only=loop-vectorize -scalable-vectorization=on \
; RUN:   -riscv-v-vector-bits-min=128 -S < %s 2>&1 | FileCheck %s

; CHECK-LABEL: vector_reverse_i64
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 1 For instruction: %{{.*}} = load i32, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 1 For instruction: store i32 %{{.*}}, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 2 For instruction: %{{.*}} = load i32, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 2 For instruction: store i32 %{{.*}}, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 4 For instruction: %{{.*}} = load i32, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 4 For instruction: store i32 %{{.*}}, ptr %{{.*}}, align 4
; CHECK: LV: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2, vscale x 4): load   %1 = load i32, ptr %arrayidx, align 4
; CHECK: remark: <unknown>:0:0: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2, vscale x 4): load
; CHECK: LV: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2, vscale x 4): store   store i32 %add9, ptr %arrayidx3, align 4
; CHECK: remark: <unknown>:0:0: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2, vscale x 4): store
; CHECK: LV: Selecting VF: 4.
define void @vector_reverse_i64(ptr nocapture noundef writeonly %A, ptr nocapture noundef readonly %B, i32 noundef signext %n) {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
  %i.0 = add nsw i32 %i.0.in8, -1
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  %add9 = add i32 %1, 1
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %idxprom
  store i32 %add9, ptr %arrayidx3, align 4
  %cmp = icmp ugt i64 %indvars.iv, 1
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

; CHECK-LABEL: vector_reverse_f32
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 1 For instruction: %{{.*}} = load float, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 1 For instruction: store float %{{.*}}, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 2 For instruction: %{{.*}} = load float, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 2 For instruction: store float %{{.*}}, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 4 For instruction: %{{.*}} = load float, ptr %{{.*}}, align 4
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 4 For instruction: store float %{{.*}}, ptr %{{.*}}, align 4
; CHECK: LV: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2, vscale x 4): load   %1 = load float, ptr %arrayidx, align 4
; CHECK: remark: <unknown>:0:0: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2, vscale x 4): load
; CHECK: LV: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2, vscale x 4): store   store float %conv1, ptr %arrayidx3, align 4
; CHECK: remark: <unknown>:0:0: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2, vscale x 4): store
; CHECK: LV: Selecting VF: 4.
define void @vector_reverse_f32(ptr nocapture noundef writeonly %A, ptr nocapture noundef readonly %B, i32 noundef signext %n) {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
  %i.0 = add nsw i32 %i.0.in8, -1
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds float, ptr %B, i64 %idxprom
  %1 = load float, ptr %arrayidx, align 4
  %conv1 = fadd float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr %A, i64 %idxprom
  store float %conv1, ptr %arrayidx3, align 4
  %cmp = icmp ugt i64 %indvars.iv, 1
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
