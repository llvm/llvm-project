; RUN: llc < %s -mtriple=mips -mcpu=mips32r2 -O2 \
; RUN:     -disable-mips-df-forward-search=false \
; RUN:     -disable-mips-df-succbb-search=false \
; RUN:     -relocation-model=static | FileCheck %s

; This test was generated with bugpoint from
; MultiSource/Applications/JM/lencod/me_fullsearch.c

%struct.SubImageContainer = type { ptr, [2 x ptr] }
%struct.storable_picture = type { i32, i32, i32, i32, i32, i32,
  [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]],
  i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
  i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr,
  ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
  ptr, ptr,
  i32, i32, i32, i32, i32, i32, i32 }

@img_height = external global i16, align 2
@width_pad = external global i32, align 4
@height_pad = external global i32, align 4
@mvbits = external global ptr, align 4
@ref_pic1_sub = external global %struct.SubImageContainer, align 4
@ref_pic2_sub = external global %struct.SubImageContainer, align 4
@wbp_weight = external global ptr, align 4
@weight1 = external global i16, align 2
@weight2 = external global i16, align 2
@offsetBi = external global i16, align 2
@computeBiPred2 = external global [3 x ptr], align 4
@computeBiPred = external global ptr, align 4
@bipred2_access_method = external global i32, align 4
@start_me_refinement_hp = external global i32, align 4

declare i32 @foobar(ptr, i32 signext , i32 signext , i32 signext ,
                    i32 signext , i32 signext , i32 signext , i32 signext ) #1

define void @SubPelBlockSearchBiPred(ptr %orig_pic, i16 signext %ref,
    i32 signext %pic_pix_x, i32 signext %pic_pix_y, i16 signext %pred_mv_y,
    ptr nocapture %mv_x, ptr nocapture %mv_y, ptr nocapture readonly %s_mv_x,
    i32 signext %search_pos2, i32 signext %min_mcost) #0 {
; CHECK-LABEL: SubPelBlockSearchBiPred:
entry:
  %add40 = shl i32 %pic_pix_x, 2
  %shl = add i32 %add40, 80
  %add41 = shl i32 %pic_pix_y, 2
  %0 = load i32, ptr @start_me_refinement_hp, align 4, !tbaa !1
  %cond47 = select i1 undef, i32 1, i32 %search_pos2
  %1 = load i16, ptr %s_mv_x, align 2, !tbaa !5
  %conv48 = sext i16 %1 to i32
  %add49 = add nsw i32 %conv48, %shl
  %idxprom52 = sext i16 %ref to i32
  %2 = load i32, ptr null, align 4, !tbaa !1
  store i32 undef, ptr @ref_pic1_sub, align 4, !tbaa !7
  %3 = load i32, ptr undef, align 4, !tbaa !10
  store i32 %3, ptr @ref_pic2_sub, align 4, !tbaa !7
  store i16 0, ptr @img_height, align 2, !tbaa !5
  %size_x_pad = getelementptr inbounds %struct.storable_picture, ptr null, i32 0, i32 22
  %4 = load i32, ptr %size_x_pad, align 4, !tbaa !12
  store i32 %4, ptr @width_pad, align 4, !tbaa !1
  %5 = load i32, ptr undef, align 4, !tbaa !13
  store i32 %5, ptr @height_pad, align 4, !tbaa !1
  %6 = load ptr, ptr @wbp_weight, align 4, !tbaa !14
  %arrayidx75 = getelementptr inbounds ptr, ptr %6, i32 undef
  %7 = load ptr, ptr %arrayidx75, align 4, !tbaa !14
  %arrayidx76 = getelementptr inbounds ptr, ptr %7, i32 %idxprom52
  %8 = load ptr, ptr %arrayidx76, align 4, !tbaa !14
  %cond87.in671 = load ptr, ptr %8, align 4
  %cond87672 = load i32, ptr %cond87.in671, align 4
  %conv88673 = trunc i32 %cond87672 to i16
  store i16 %conv88673, ptr @weight1, align 2, !tbaa !5
  %cond105 = load i32, ptr undef, align 4
  %conv106 = trunc i32 %cond105 to i16
  store i16 %conv106, ptr @weight2, align 2, !tbaa !5
  store i16 0, ptr @offsetBi, align 2, !tbaa !5
  %storemerge655 = load i32, ptr getelementptr inbounds ([3 x ptr], ptr @computeBiPred2, i32 0, i32 1), align 4
  store i32 %storemerge655, ptr @computeBiPred, align 4, !tbaa !14
  %9 = load i16, ptr %mv_x, align 2, !tbaa !5
  %cmp270 = icmp sgt i32 undef, 1
  %or.cond = and i1 %cmp270, false
  br i1 %or.cond, label %land.lhs.true277, label %if.else289

land.lhs.true277:                                 ; preds = %entry
  %10 = load i16, ptr %mv_y, align 2, !tbaa !5
  %conv278 = sext i16 %10 to i32
  %add279 = add nsw i32 %conv278, 0
  %cmp280 = icmp sgt i32 %add279, 1
  %or.cond660 = and i1 %cmp280, undef
  br i1 %or.cond660, label %if.end290, label %if.else289

if.else289:                                       ; preds = %land.lhs.true277, %entry
  br label %if.end290

if.end290:                                        ; preds = %if.else289, %land.lhs.true277
  %storemerge = phi i32 [ 1, %if.else289 ], [ 0, %land.lhs.true277 ]
  store i32 %storemerge, ptr @bipred2_access_method, align 4, !tbaa !1
  %cmp315698 = icmp slt i32 %0, %cond47
  br i1 %cmp315698, label %for.body.lr.ph, label %if.end358

for.body.lr.ph:                                   ; preds = %if.end290
  %conv328 = sext i16 %pred_mv_y to i32
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %11 = phi i16 [ %9, %for.body.lr.ph ], [ %.pre, %for.inc ]
  %min_mcost.addr.0701 = phi i32 [ %min_mcost, %for.body.lr.ph ], [ undef, %for.inc ]
  %pos.0700 = phi i32 [ %0, %for.body.lr.ph ], [ undef, %for.inc ]
  %best_pos.0699 = phi i32 [ 0, %for.body.lr.ph ], [ %best_pos.1, %for.inc ]
  %conv317 = sext i16 %11 to i32
  %add320 = add nsw i32 0, %conv317
  %12 = load i16, ptr %mv_y, align 2, !tbaa !5
  %conv321 = sext i16 %12 to i32
  %add324 = add nsw i32 0, %conv321
  %13 = load ptr, ptr @mvbits, align 4, !tbaa !14
  %14 = load i32, ptr undef, align 4, !tbaa !1
  %sub329 = sub nsw i32 %add324, %conv328
  %arrayidx330 = getelementptr inbounds i32, ptr %13, i32 %sub329
  %15 = load i32, ptr %arrayidx330, align 4, !tbaa !1
  %add331 = add nsw i32 %15, %14
  %mul = mul nsw i32 %add331, %2
  %shr332 = ashr i32 %mul, 16
  %cmp333 = icmp sgt i32 %min_mcost.addr.0701, %shr332
  br i1 %cmp333, label %if.end336, label %for.inc

if.end336:                                        ; preds = %for.body
  ; CHECK:      jalr  $25
  ; CHECK-NOT:  move  $ra, {{.*}}
  ; CHECK:      j     $BB{{.*}}
  %add337 = add nsw i32 %add320, %shl
  %add338 = add nsw i32 %add324, 0
  %call340 = tail call i32 undef(ptr %orig_pic, i32 signext undef, i32 signext
                                 undef, i32 signext 0, i32 signext %add49,
                                 i32 signext undef, i32 signext %add337,
                                 i32 signext %add338) #1
  %cmp342 = icmp slt i32 0, %min_mcost.addr.0701
  %pos.0.best_pos.0 = select i1 %cmp342, i32 %pos.0700, i32 %best_pos.0699
  br label %for.inc

for.inc:                                          ; preds = %if.end336, %for.body
  %best_pos.1 = phi i32 [ %best_pos.0699, %for.body ], [ %pos.0.best_pos.0, %if.end336 ]
  %.pre = load i16, ptr %mv_x, align 2, !tbaa !5
  br label %for.body

if.end358:                                        ; preds = %if.end290
  %.min_mcost.addr.0 = select i1 false, i32 2147483647, i32 %min_mcost
  br i1 undef, label %for.body415.lr.ph, label %if.end461

for.body415.lr.ph:                                ; preds = %if.end358
  %16 = load i16, ptr %mv_y, align 2, !tbaa !5
  %conv420 = sext i16 %16 to i32
  %add423 = add nsw i32 0, %conv420
  %cmp433 = icmp sgt i32 %.min_mcost.addr.0, 0
  br i1 %cmp433, label %if.end436, label %if.end461

if.end436:                                        ; preds = %for.body415.lr.ph
  %add438 = add nsw i32 %add423, 0
  %call440 = tail call i32 @foobar(ptr %orig_pic, i32 signext undef, i32 signext undef,
                                 i32 signext 0, i32 signext %add49, i32 signext undef,
                                 i32 signext undef, i32 signext %add438) #1
  br label %if.end461

if.end461:                                        ; preds = %if.end436, %for.body415.lr.ph, %if.end358
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="mips32r2" "target-features"="+mips32r2,+nooddspreg,+fpxx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 236218) (llvm/trunk 236237)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"short", !3, i64 0}
!7 = !{!8, !9, i64 0}
!8 = !{!"", !9, i64 0, !3, i64 4}
!9 = !{!"any pointer", !3, i64 0}
!10 = !{!11, !9, i64 6440}
!11 = !{!"storable_picture", !3, i64 0, !2, i64 4, !2, i64 8, !2, i64 12, !2, i64 16, !2, i64 20, !3, i64 24, !3, i64 1608, !3, i64 3192, !3, i64 4776, !2, i64 6360, !2, i64 6364, !2, i64 6368, !2, i64 6372, !2, i64 6376, !2, i64 6380, !2, i64 6384, !2, i64 6388, !2, i64 6392, !2, i64 6396, !2, i64 6400, !2, i64 6404, !2, i64 6408, !2, i64 6412, !2, i64 6416, !2, i64 6420, !2, i64 6424, !2, i64 6428, !2, i64 6432, !9, i64 6436, !9, i64 6440, !9, i64 6444, !9, i64 6448, !9, i64 6452, !9, i64 6456, !9, i64 6460, !9, i64 6464, !9, i64 6468, !9, i64 6472, !9, i64 6476, !9, i64 6480, !9, i64 6484, !9, i64 6488, !9, i64 6492, !2, i64 6496, !2, i64 6500, !2, i64 6504, !2, i64 6508, !2, i64 6512, !2, i64 6516, !2, i64 6520}
!12 = !{!11, !2, i64 6408}
!13 = !{!11, !2, i64 6412}
!14 = !{!9, !9, i64 0}
