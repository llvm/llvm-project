; RUN: llc -verify-machineinstrs -mtriple=s390x-ibm-linux -mcpu=z13 -O3 -hoist-const-stores < %s | FileCheck %s

@b = dso_local local_unnamed_addr global i32 15, align 4
@e = dso_local local_unnamed_addr global i32 -1, align 4
@f = common dso_local global i32 0, align 4
@g = dso_local local_unnamed_addr global ptr @f, align 8
@c = common dso_local local_unnamed_addr global i32 0, align 4
@a = common dso_local local_unnamed_addr global [6 x i32] zeroinitializer, align 4
@d = common dso_local local_unnamed_addr global i32 0, align 4
@h = common dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [15 x i8] c"checksum = %X\0A\00", align 2

; Function Attrs: nounwind
define dso_local signext i32 @main()  {
entry:
  %i = alloca i32, align 4
  %.pr = load i32, ptr @c, align 4, !tbaa !2
  %cmp6 = icmp slt i32 %.pr, 6
  br i1 %cmp6, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  store i32 5, ptr getelementptr inbounds ([6 x i32], ptr @a, i64 0, i64 1), align 4, !tbaa !2
  store i32 6, ptr @c, align 4, !tbaa !2
  br label %for.end

for.end:                                          ; preds = %for.body.preheader, %entry
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %i)
  store i32 14, ptr %i, align 4, !tbaa !2
  %.pr2 = load i32, ptr @d, align 4, !tbaa !2
  %cmp25 = icmp sgt i32 %.pr2, -1
  br i1 %cmp25, label %for.cond4thread-pre-split.lr.ph, label %for.end.for.end11_crit_edge

for.end.for.end11_crit_edge:                      ; preds = %for.end
  %.pre10 = load i32, ptr @b, align 4, !tbaa !2
  br label %for.end11

; CHECK: # %for.cond4thread-pre-split.lr.ph
; CHECK-NOT: mvhi    164(%r15), 0
; CHECK: # %for.end9
; CHECK: mvhi    164(%r15), 0

for.cond4thread-pre-split.lr.ph:                  ; preds = %for.end
  %0 = ptrtoint ptr %i to i64
  %1 = trunc i64 %0 to i32
  %2 = load ptr, ptr @g, align 8
  %.pr3.pre = load i32, ptr @e, align 4, !tbaa !2
  br label %for.cond4thread-pre-split

for.cond4thread-pre-split:                        ; preds = %for.cond4thread-pre-split.lr.ph, %for.end9
  %3 = phi i32 [ %.pr2, %for.cond4thread-pre-split.lr.ph ], [ %dec, %for.end9 ]
  %4 = phi i32 [ 14, %for.cond4thread-pre-split.lr.ph ], [ 0, %for.end9 ]
  %.pr3 = phi i32 [ %.pr3.pre, %for.cond4thread-pre-split.lr.ph ], [ %.pr37, %for.end9 ]
  %cmp54 = icmp slt i32 %.pr3, 1
  br i1 %cmp54, label %for.body6.preheader, label %for.end9

for.body6.preheader:                              ; preds = %for.cond4thread-pre-split
  store i32 %4, ptr %2, align 4, !tbaa !2
  %5 = load i32, ptr @e, align 4, !tbaa !2
  %inc811 = add nsw i32 %5, 1
  store i32 %inc811, ptr @e, align 4, !tbaa !2
  %cmp512 = icmp slt i32 %5, 0
  br i1 %cmp512, label %for.body6.for.body6_crit_edge, label %for.end9.loopexit

for.body6.for.body6_crit_edge:                    ; preds = %for.body6.preheader, %for.body6.for.body6_crit_edge.3
  %.pre = load i32, ptr %i, align 4, !tbaa !2
  store i32 %.pre, ptr %2, align 4, !tbaa !2
  %6 = load i32, ptr @e, align 4, !tbaa !2
  %inc8 = add nsw i32 %6, 1
  store i32 %inc8, ptr @e, align 4, !tbaa !2
  %cmp5 = icmp slt i32 %6, 0
  br i1 %cmp5, label %for.body6.for.body6_crit_edge.1, label %for.end9.loopexit

for.end9.loopexit:                                ; preds = %for.body6.for.body6_crit_edge, %for.body6.for.body6_crit_edge.1, %for.body6.for.body6_crit_edge.2, %for.body6.for.body6_crit_edge.3, %for.body6.preheader
  %inc8.lcssa = phi i32 [ %inc811, %for.body6.preheader ], [ %inc8, %for.body6.for.body6_crit_edge ], [ %inc8.1, %for.body6.for.body6_crit_edge.1 ], [ %inc8.2, %for.body6.for.body6_crit_edge.2 ], [ %inc8.3, %for.body6.for.body6_crit_edge.3 ]
  %.pre9 = load i32, ptr @d, align 4, !tbaa !2
  br label %for.end9

for.end9:                                         ; preds = %for.end9.loopexit, %for.cond4thread-pre-split
  %7 = phi i32 [ %.pre9, %for.end9.loopexit ], [ %3, %for.cond4thread-pre-split ]
  %.pr37 = phi i32 [ %inc8.lcssa, %for.end9.loopexit ], [ %.pr3, %for.cond4thread-pre-split ]
  store i32 %1, ptr @h, align 4, !tbaa !2
  store i32 0, ptr %i, align 4, !tbaa !2
  %8 = load i32, ptr @b, align 4, !tbaa !2
  %9 = load i32, ptr @f, align 4, !tbaa !2
  %xor = xor i32 %9, %8
  %idxprom = sext i32 %xor to i64
  %arrayidx = getelementptr inbounds [6 x i32], ptr @a, i64 0, i64 %idxprom
  %10 = load i32, ptr %arrayidx, align 4, !tbaa !2
  store i32 %10, ptr @b, align 4, !tbaa !2
  %dec = add nsw i32 %7, -1
  store i32 %dec, ptr @d, align 4, !tbaa !2
  %cmp2 = icmp sgt i32 %7, 0
  br i1 %cmp2, label %for.cond4thread-pre-split, label %for.end11

for.end11:                                        ; preds = %for.end9, %for.end.for.end11_crit_edge
  %11 = phi i32 [ %.pre10, %for.end.for.end11_crit_edge ], [ %10, %for.end9 ]
  %call = call signext i32 (ptr, ...) @printf(ptr @.str, i32 signext %11)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %i)
  ret i32 0

for.body6.for.body6_crit_edge.1:                  ; preds = %for.body6.for.body6_crit_edge
  %.pre.1 = load i32, ptr %i, align 4, !tbaa !2
  store i32 %.pre.1, ptr %2, align 4, !tbaa !2
  %12 = load i32, ptr @e, align 4, !tbaa !2
  %inc8.1 = add nsw i32 %12, 1
  store i32 %inc8.1, ptr @e, align 4, !tbaa !2
  %cmp5.1 = icmp slt i32 %12, 0
  br i1 %cmp5.1, label %for.body6.for.body6_crit_edge.2, label %for.end9.loopexit

for.body6.for.body6_crit_edge.2:                  ; preds = %for.body6.for.body6_crit_edge.1
  %.pre.2 = load i32, ptr %i, align 4, !tbaa !2
  store i32 %.pre.2, ptr %2, align 4, !tbaa !2
  %13 = load i32, ptr @e, align 4, !tbaa !2
  %inc8.2 = add nsw i32 %13, 1
  store i32 %inc8.2, ptr @e, align 4, !tbaa !2
  %cmp5.2 = icmp slt i32 %13, 0
  br i1 %cmp5.2, label %for.body6.for.body6_crit_edge.3, label %for.end9.loopexit

for.body6.for.body6_crit_edge.3:                  ; preds = %for.body6.for.body6_crit_edge.2
  %.pre.3 = load i32, ptr %i, align 4, !tbaa !2
  store i32 %.pre.3, ptr %2, align 4, !tbaa !2
  %14 = load i32, ptr @e, align 4, !tbaa !2
  %inc8.3 = add nsw i32 %14, 1
  store i32 %inc8.3, ptr @e, align 4, !tbaa !2
  %cmp5.3 = icmp slt i32 %14, 0
  br i1 %cmp5.3, label %for.body6.for.body6_crit_edge, label %for.end9.loopexit
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

; Function Attrs: nounwind
declare dso_local signext i32 @printf(ptr nocapture readonly, ...)

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
