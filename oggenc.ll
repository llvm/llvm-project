; ModuleID = './oggenc.c'
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-none--eabi"

; Function Attrs: nounwind
define void @bark_noise_hybridmp(i32 %n, i32* nocapture readonly %b, float* nocapture readonly %f, float* nocapture %noise, float %offset, i32 %fixed) #0 {
entry:
  %mul = shl i32 %n, 2
  %0 = alloca i8, i32 %mul, align 4
  %1 = bitcast i8* %0 to float*
  %2 = alloca i8, i32 %mul, align 4
  %3 = bitcast i8* %2 to float*
  %4 = alloca i8, i32 %mul, align 4
  %5 = bitcast i8* %4 to float*
  %6 = alloca i8, i32 %mul, align 4
  %7 = bitcast i8* %6 to float*
  %8 = alloca i8, i32 %mul, align 4
  %9 = bitcast i8* %8 to float*
  %10 = load float, float* %f, align 4, !tbaa !3
  %add = fadd float %10, %offset
  %cmp = fcmp olt float %add, 1.000000e+00
  %y.0 = select i1 %cmp, float 1.000000e+00, float %add
  %mul5 = fmul float %y.0, %y.0
  %conv7 = fmul float %mul5, 5.000000e-01
  %add8 = fadd float %conv7, 0.000000e+00
  %mul10 = fmul float %y.0, %conv7
  %add11 = fadd float %mul10, 0.000000e+00
  store float %add8, float* %1, align 4, !tbaa !3
  store float %add8, float* %3, align 4, !tbaa !3
  store float 0.000000e+00, float* %5, align 4, !tbaa !3
  store float %add11, float* %7, align 4, !tbaa !3
  store float 0.000000e+00, float* %9, align 4, !tbaa !3
  %cmp17599 = icmp sgt i32 %n, 1
  br i1 %cmp17599, label %for.body.lr.ph, label %for.cond43.preheader

for.body.lr.ph:                                   ; preds = %entry
  %11 = add i32 %n, -1
  br label %for.body

for.cond43.preheader.loopexit:                    ; preds = %for.body
  br label %for.cond43.preheader

for.cond43.preheader:                             ; preds = %for.cond43.preheader.loopexit, %entry
  %12 = load i32, i32* %b, align 4, !tbaa !7
  %shr589 = ashr i32 %12, 16
  %cmp45590 = icmp sgt i32 %shr589, -1
  br i1 %cmp45590, label %for.cond90.preheader, label %if.end48.preheader

if.end48.preheader:                               ; preds = %for.cond43.preheader
  br label %if.end48

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %x.0606 = phi float [ 1.000000e+00, %for.body.lr.ph ], [ %add42, %for.body ]
  %i.0605 = phi i32 [ 1, %for.body.lr.ph ], [ %inc, %for.body ]
  %tXY.0604 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add36, %for.body ]
  %tY.0603 = phi float [ %add11, %for.body.lr.ph ], [ %add33, %for.body ]
  %tXX.0602 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add31, %for.body ]
  %tX.0601 = phi float [ %add8, %for.body.lr.ph ], [ %add28, %for.body ]
  %tN.0600 = phi float [ %add8, %for.body.lr.ph ], [ %add26, %for.body ]
  %arrayidx19 = getelementptr inbounds float, float* %f, i32 %i.0605
  %13 = load float, float* %arrayidx19, align 4, !tbaa !3
  %add20 = fadd float %13, %offset
  %cmp21 = fcmp olt float %add20, 1.000000e+00
  %y.1 = select i1 %cmp21, float 1.000000e+00, float %add20
  %mul25 = fmul float %y.1, %y.1
  %add26 = fadd float %tN.0600, %mul25
  %mul27 = fmul float %x.0606, %mul25
  %add28 = fadd float %tX.0601, %mul27
  %mul30 = fmul float %x.0606, %mul27
  %add31 = fadd float %tXX.0602, %mul30
  %mul32 = fmul float %y.1, %mul25
  %add33 = fadd float %tY.0603, %mul32
  %mul35 = fmul float %y.1, %mul27
  %add36 = fadd float %tXY.0604, %mul35
  %arrayidx37 = getelementptr inbounds float, float* %1, i32 %i.0605
  store float %add26, float* %arrayidx37, align 4, !tbaa !3
  %arrayidx38 = getelementptr inbounds float, float* %3, i32 %i.0605
  store float %add28, float* %arrayidx38, align 4, !tbaa !3
  %arrayidx39 = getelementptr inbounds float, float* %5, i32 %i.0605
  store float %add31, float* %arrayidx39, align 4, !tbaa !3
  %arrayidx40 = getelementptr inbounds float, float* %7, i32 %i.0605
  store float %add33, float* %arrayidx40, align 4, !tbaa !3
  %arrayidx41 = getelementptr inbounds float, float* %9, i32 %i.0605
  store float %add36, float* %arrayidx41, align 4, !tbaa !3
  %inc = add nsw i32 %i.0605, 1
  %add42 = fadd float %x.0606, 1.000000e+00
  %exitcond610 = icmp eq i32 %i.0605, %11
  br i1 %exitcond610, label %for.cond43.preheader.loopexit, label %for.body

for.cond90.preheader.loopexit:                    ; preds = %if.end48
  %.lcssa = phi i32 [ %26, %if.end48 ]
  %add88.lcssa = phi float [ %add88, %if.end48 ]
  %inc87.lcssa = phi i32 [ %inc87, %if.end48 ]
  %sub77.lcssa = phi float [ %sub77, %if.end48 ]
  %sub74.lcssa = phi float [ %sub74, %if.end48 ]
  %sub71.lcssa = phi float [ %sub71, %if.end48 ]
  br label %for.cond90.preheader

for.cond90.preheader:                             ; preds = %for.cond90.preheader.loopexit, %for.cond43.preheader
  %14 = phi i32 [ %12, %for.cond43.preheader ], [ %.lcssa, %for.cond90.preheader.loopexit ]
  %x.1.lcssa = phi float [ 0.000000e+00, %for.cond43.preheader ], [ %add88.lcssa, %for.cond90.preheader.loopexit ]
  %D.0.lcssa = phi float [ undef, %for.cond43.preheader ], [ %sub77.lcssa, %for.cond90.preheader.loopexit ]
  %B.0.lcssa = phi float [ undef, %for.cond43.preheader ], [ %sub74.lcssa, %for.cond90.preheader.loopexit ]
  %A.0.lcssa = phi float [ undef, %for.cond43.preheader ], [ %sub71.lcssa, %for.cond90.preheader.loopexit ]
  %i.1.lcssa = phi i32 [ 0, %for.cond43.preheader ], [ %inc87.lcssa, %for.cond90.preheader.loopexit ]
  %and94578 = and i32 %14, 65535
  %cmp95579 = icmp slt i32 %and94578, %n
  br i1 %cmp95579, label %if.end98.preheader, label %for.cond136.preheader

if.end98.preheader:                               ; preds = %for.cond90.preheader
  br label %if.end98

if.end48:                                         ; preds = %if.end48.preheader, %if.end48
  %shr593 = phi i32 [ %shr, %if.end48 ], [ %shr589, %if.end48.preheader ]
  %15 = phi i32 [ %26, %if.end48 ], [ %12, %if.end48.preheader ]
  %x.1592 = phi float [ %add88, %if.end48 ], [ 0.000000e+00, %if.end48.preheader ]
  %i.1591 = phi i32 [ %inc87, %if.end48 ], [ 0, %if.end48.preheader ]
  %and = and i32 %15, 65535
  %arrayidx50 = getelementptr inbounds float, float* %1, i32 %and
  %16 = load float, float* %arrayidx50, align 4, !tbaa !3
  %sub = sub nsw i32 0, %shr593
  %arrayidx51 = getelementptr inbounds float, float* %1, i32 %sub
  %17 = load float, float* %arrayidx51, align 4, !tbaa !3
  %add52 = fadd float %16, %17
  %arrayidx53 = getelementptr inbounds float, float* %3, i32 %and
  %18 = load float, float* %arrayidx53, align 4, !tbaa !3
  %arrayidx55 = getelementptr inbounds float, float* %3, i32 %sub
  %19 = load float, float* %arrayidx55, align 4, !tbaa !3
  %sub56 = fsub float %18, %19
  %arrayidx57 = getelementptr inbounds float, float* %5, i32 %and
  %20 = load float, float* %arrayidx57, align 4, !tbaa !3
  %arrayidx59 = getelementptr inbounds float, float* %5, i32 %sub
  %21 = load float, float* %arrayidx59, align 4, !tbaa !3
  %add60 = fadd float %20, %21
  %arrayidx61 = getelementptr inbounds float, float* %7, i32 %and
  %22 = load float, float* %arrayidx61, align 4, !tbaa !3
  %arrayidx63 = getelementptr inbounds float, float* %7, i32 %sub
  %23 = load float, float* %arrayidx63, align 4, !tbaa !3
  %add64 = fadd float %22, %23
  %arrayidx65 = getelementptr inbounds float, float* %9, i32 %and
  %24 = load float, float* %arrayidx65, align 4, !tbaa !3
  %arrayidx67 = getelementptr inbounds float, float* %9, i32 %sub
  %25 = load float, float* %arrayidx67, align 4, !tbaa !3
  %sub68 = fsub float %24, %25
  %mul69 = fmul float %add60, %add64
  %mul70 = fmul float %sub56, %sub68
  %sub71 = fsub float %mul69, %mul70
  %mul72 = fmul float %add52, %sub68
  %mul73 = fmul float %sub56, %add64
  %sub74 = fsub float %mul72, %mul73
  %mul75 = fmul float %add52, %add60
  %mul76 = fmul float %sub56, %sub56
  %sub77 = fsub float %mul75, %mul76
  %mul78 = fmul float %x.1592, %sub74
  %add79 = fadd float %sub71, %mul78
  %div = fdiv float %add79, %sub77
  %cmp80 = fcmp olt float %div, 0.000000e+00
  %R.0 = select i1 %cmp80, float 0.000000e+00, float %div
  %sub84 = fsub float %R.0, %offset
  %arrayidx85 = getelementptr inbounds float, float* %noise, i32 %i.1591
  store float %sub84, float* %arrayidx85, align 4, !tbaa !3
  %inc87 = add nsw i32 %i.1591, 1
  %add88 = fadd float %x.1592, 1.000000e+00
  %arrayidx44 = getelementptr inbounds i32, i32* %b, i32 %inc87
  %26 = load i32, i32* %arrayidx44, align 4, !tbaa !7
  %shr = ashr i32 %26, 16
  %cmp45 = icmp sgt i32 %shr, -1
  br i1 %cmp45, label %for.cond90.preheader.loopexit, label %if.end48

for.cond136.preheader.loopexit:                   ; preds = %if.end98
  %add134.lcssa = phi float [ %add134, %if.end98 ]
  %inc133.lcssa = phi i32 [ %inc133, %if.end98 ]
  %sub122.lcssa = phi float [ %sub122, %if.end98 ]
  %sub119.lcssa = phi float [ %sub119, %if.end98 ]
  %sub116.lcssa = phi float [ %sub116, %if.end98 ]
  br label %for.cond136.preheader

for.cond136.preheader:                            ; preds = %for.cond136.preheader.loopexit, %for.cond90.preheader
  %x.2.lcssa = phi float [ %x.1.lcssa, %for.cond90.preheader ], [ %add134.lcssa, %for.cond136.preheader.loopexit ]
  %D.1.lcssa = phi float [ %D.0.lcssa, %for.cond90.preheader ], [ %sub122.lcssa, %for.cond136.preheader.loopexit ]
  %B.1.lcssa = phi float [ %B.0.lcssa, %for.cond90.preheader ], [ %sub119.lcssa, %for.cond136.preheader.loopexit ]
  %A.1.lcssa = phi float [ %A.0.lcssa, %for.cond90.preheader ], [ %sub116.lcssa, %for.cond136.preheader.loopexit ]
  %i.2.lcssa = phi i32 [ %i.1.lcssa, %for.cond90.preheader ], [ %inc133.lcssa, %for.cond136.preheader.loopexit ]
  %cmp137573 = icmp slt i32 %i.2.lcssa, %n
  br i1 %cmp137573, label %for.body139.lr.ph, label %for.end152

for.body139.lr.ph:                                ; preds = %for.cond136.preheader
  %27 = add i32 %n, -1
  br label %for.body139

if.end98:                                         ; preds = %if.end98.preheader, %if.end98
  %and94583 = phi i32 [ %and94, %if.end98 ], [ %and94578, %if.end98.preheader ]
  %shr92582.in = phi i32 [ %38, %if.end98 ], [ %14, %if.end98.preheader ]
  %x.2581 = phi float [ %add134, %if.end98 ], [ %x.1.lcssa, %if.end98.preheader ]
  %i.2580 = phi i32 [ %inc133, %if.end98 ], [ %i.1.lcssa, %if.end98.preheader ]
  %shr92582 = ashr i32 %shr92582.in, 16
  %arrayidx99 = getelementptr inbounds float, float* %1, i32 %and94583
  %28 = load float, float* %arrayidx99, align 4, !tbaa !3
  %arrayidx100 = getelementptr inbounds float, float* %1, i32 %shr92582
  %29 = load float, float* %arrayidx100, align 4, !tbaa !3
  %sub101 = fsub float %28, %29
  %arrayidx102 = getelementptr inbounds float, float* %3, i32 %and94583
  %30 = load float, float* %arrayidx102, align 4, !tbaa !3
  %arrayidx103 = getelementptr inbounds float, float* %3, i32 %shr92582
  %31 = load float, float* %arrayidx103, align 4, !tbaa !3
  %sub104 = fsub float %30, %31
  %arrayidx105 = getelementptr inbounds float, float* %5, i32 %and94583
  %32 = load float, float* %arrayidx105, align 4, !tbaa !3
  %arrayidx106 = getelementptr inbounds float, float* %5, i32 %shr92582
  %33 = load float, float* %arrayidx106, align 4, !tbaa !3
  %sub107 = fsub float %32, %33
  %arrayidx108 = getelementptr inbounds float, float* %7, i32 %and94583
  %34 = load float, float* %arrayidx108, align 4, !tbaa !3
  %arrayidx109 = getelementptr inbounds float, float* %7, i32 %shr92582
  %35 = load float, float* %arrayidx109, align 4, !tbaa !3
  %sub110 = fsub float %34, %35
  %arrayidx111 = getelementptr inbounds float, float* %9, i32 %and94583
  %36 = load float, float* %arrayidx111, align 4, !tbaa !3
  %arrayidx112 = getelementptr inbounds float, float* %9, i32 %shr92582
  %37 = load float, float* %arrayidx112, align 4, !tbaa !3
  %sub113 = fsub float %36, %37
  %mul114 = fmul float %sub107, %sub110
  %mul115 = fmul float %sub104, %sub113
  %sub116 = fsub float %mul114, %mul115
  %mul117 = fmul float %sub101, %sub113
  %mul118 = fmul float %sub104, %sub110
  %sub119 = fsub float %mul117, %mul118
  %mul120 = fmul float %sub101, %sub107
  %mul121 = fmul float %sub104, %sub104
  %sub122 = fsub float %mul120, %mul121
  %mul123 = fmul float %x.2581, %sub119
  %add124 = fadd float %sub116, %mul123
  %div125 = fdiv float %add124, %sub122
  %cmp126 = fcmp olt float %div125, 0.000000e+00
  %R.1 = select i1 %cmp126, float 0.000000e+00, float %div125
  %sub130 = fsub float %R.1, %offset
  %arrayidx131 = getelementptr inbounds float, float* %noise, i32 %i.2580
  store float %sub130, float* %arrayidx131, align 4, !tbaa !3
  %inc133 = add nsw i32 %i.2580, 1
  %add134 = fadd float %x.2581, 1.000000e+00
  %arrayidx91 = getelementptr inbounds i32, i32* %b, i32 %inc133
  %38 = load i32, i32* %arrayidx91, align 4, !tbaa !7
  %and94 = and i32 %38, 65535
  %cmp95 = icmp slt i32 %and94, %n
  br i1 %cmp95, label %if.end98, label %for.cond136.preheader.loopexit

for.body139:                                      ; preds = %for.body139, %for.body139.lr.ph
  %x.3575 = phi float [ %x.2.lcssa, %for.body139.lr.ph ], [ %add151, %for.body139 ]
  %i.3574 = phi i32 [ %i.2.lcssa, %for.body139.lr.ph ], [ %inc150, %for.body139 ]
  %mul140 = fmul float %B.1.lcssa, %x.3575
  %add141 = fadd float %A.1.lcssa, %mul140
  %div142 = fdiv float %add141, %D.1.lcssa
  %cmp143 = fcmp olt float %div142, 0.000000e+00
  %R.2 = select i1 %cmp143, float 0.000000e+00, float %div142
  %sub147 = fsub float %R.2, %offset
  %arrayidx148 = getelementptr inbounds float, float* %noise, i32 %i.3574
  store float %sub147, float* %arrayidx148, align 4, !tbaa !3
  %inc150 = add nsw i32 %i.3574, 1
  %add151 = fadd float %x.3575, 1.000000e+00
  %exitcond609 = icmp eq i32 %i.3574, %27
  br i1 %exitcond609, label %for.end152.loopexit, label %for.body139

for.end152.loopexit:                              ; preds = %for.body139
  br label %for.end152

for.end152:                                       ; preds = %for.end152.loopexit, %for.cond136.preheader
  %cmp153 = icmp slt i32 %fixed, 1
  br i1 %cmp153, label %for.end274, label %for.cond157.preheader

for.cond157.preheader:                            ; preds = %for.end152
  %div158 = sdiv i32 %fixed, 2
  %sub160561 = sub nsw i32 %div158, %fixed
  %cmp161562 = icmp sgt i32 %sub160561, -1
  br i1 %cmp161562, label %for.cond209.preheader, label %if.end164.lr.ph

if.end164.lr.ph:                                  ; preds = %for.cond157.preheader
  %39 = add i32 %fixed, -1
  %40 = sub i32 %39, %div158
  br label %if.end164

for.cond157.for.cond209.preheader_crit_edge:      ; preds = %for.inc205
  %add207.lcssa = phi float [ %add207, %for.inc205 ]
  %sub193.lcssa = phi float [ %sub193, %for.inc205 ]
  %sub190.lcssa = phi float [ %sub190, %for.inc205 ]
  %sub187.lcssa = phi float [ %sub187, %for.inc205 ]
  %41 = sub i32 %fixed, %div158
  br label %for.cond209.preheader

for.cond209.preheader:                            ; preds = %for.cond157.for.cond209.preheader_crit_edge, %for.cond157.preheader
  %x.4.lcssa = phi float [ %add207.lcssa, %for.cond157.for.cond209.preheader_crit_edge ], [ 0.000000e+00, %for.cond157.preheader ]
  %D.2.lcssa = phi float [ %sub193.lcssa, %for.cond157.for.cond209.preheader_crit_edge ], [ %D.1.lcssa, %for.cond157.preheader ]
  %B.2.lcssa = phi float [ %sub190.lcssa, %for.cond157.for.cond209.preheader_crit_edge ], [ %B.1.lcssa, %for.cond157.preheader ]
  %A.2.lcssa = phi float [ %sub187.lcssa, %for.cond157.for.cond209.preheader_crit_edge ], [ %A.1.lcssa, %for.cond157.preheader ]
  %i.4.lcssa = phi i32 [ %41, %for.cond157.for.cond209.preheader_crit_edge ], [ 0, %for.cond157.preheader ]
  %add211550 = add nsw i32 %i.4.lcssa, %div158
  %cmp213552 = icmp slt i32 %add211550, %n
  br i1 %cmp213552, label %if.end216.lr.ph, label %for.cond256.preheader

if.end216.lr.ph:                                  ; preds = %for.cond209.preheader
  %42 = add i32 %n, -1
  %43 = sub i32 %42, %div158
  br label %if.end216

if.end164:                                        ; preds = %if.end164.lr.ph, %for.inc205
  %sub160566 = phi i32 [ %sub160561, %if.end164.lr.ph ], [ %sub160, %for.inc205 ]
  %add159565 = phi i32 [ %div158, %if.end164.lr.ph ], [ %add159, %for.inc205 ]
  %x.4564 = phi float [ 0.000000e+00, %if.end164.lr.ph ], [ %add207, %for.inc205 ]
  %i.4563 = phi i32 [ 0, %if.end164.lr.ph ], [ %inc206, %for.inc205 ]
  %arrayidx165 = getelementptr inbounds float, float* %1, i32 %add159565
  %44 = load float, float* %arrayidx165, align 4, !tbaa !3
  %sub166 = sub nsw i32 0, %sub160566
  %arrayidx167 = getelementptr inbounds float, float* %1, i32 %sub166
  %45 = load float, float* %arrayidx167, align 4, !tbaa !3
  %add168 = fadd float %44, %45
  %arrayidx169 = getelementptr inbounds float, float* %3, i32 %add159565
  %46 = load float, float* %arrayidx169, align 4, !tbaa !3
  %arrayidx171 = getelementptr inbounds float, float* %3, i32 %sub166
  %47 = load float, float* %arrayidx171, align 4, !tbaa !3
  %sub172 = fsub float %46, %47
  %arrayidx173 = getelementptr inbounds float, float* %5, i32 %add159565
  %48 = load float, float* %arrayidx173, align 4, !tbaa !3
  %arrayidx175 = getelementptr inbounds float, float* %5, i32 %sub166
  %49 = load float, float* %arrayidx175, align 4, !tbaa !3
  %add176 = fadd float %48, %49
  %arrayidx177 = getelementptr inbounds float, float* %7, i32 %add159565
  %50 = load float, float* %arrayidx177, align 4, !tbaa !3
  %arrayidx179 = getelementptr inbounds float, float* %7, i32 %sub166
  %51 = load float, float* %arrayidx179, align 4, !tbaa !3
  %add180 = fadd float %50, %51
  %arrayidx181 = getelementptr inbounds float, float* %9, i32 %add159565
  %52 = load float, float* %arrayidx181, align 4, !tbaa !3
  %arrayidx183 = getelementptr inbounds float, float* %9, i32 %sub166
  %53 = load float, float* %arrayidx183, align 4, !tbaa !3
  %sub184 = fsub float %52, %53
  %mul185 = fmul float %add176, %add180
  %mul186 = fmul float %sub172, %sub184
  %sub187 = fsub float %mul185, %mul186
  %mul188 = fmul float %add168, %sub184
  %mul189 = fmul float %sub172, %add180
  %sub190 = fsub float %mul188, %mul189
  %mul191 = fmul float %add168, %add176
  %mul192 = fmul float %sub172, %sub172
  %sub193 = fsub float %mul191, %mul192
  %mul194 = fmul float %x.4564, %sub190
  %add195 = fadd float %sub187, %mul194
  %div196 = fdiv float %add195, %sub193
  %sub197 = fsub float %div196, %offset
  %arrayidx198 = getelementptr inbounds float, float* %noise, i32 %i.4563
  %54 = load float, float* %arrayidx198, align 4, !tbaa !3
  %cmp199 = fcmp olt float %sub197, %54
  br i1 %cmp199, label %if.then201, label %for.inc205

if.then201:                                       ; preds = %if.end164
  store float %sub197, float* %arrayidx198, align 4, !tbaa !3
  br label %for.inc205

for.inc205:                                       ; preds = %if.end164, %if.then201
  %inc206 = add nsw i32 %i.4563, 1
  %add207 = fadd float %x.4564, 1.000000e+00
  %add159 = add nsw i32 %div158, %inc206
  %sub160 = sub nsw i32 %add159, %fixed
  %exitcond608 = icmp eq i32 %i.4563, %40
  br i1 %exitcond608, label %for.cond157.for.cond209.preheader_crit_edge, label %if.end164

for.cond209.for.cond256.preheader_crit_edge:      ; preds = %for.inc252
  %add254.lcssa = phi float [ %add254, %for.inc252 ]
  %sub240.lcssa = phi float [ %sub240, %for.inc252 ]
  %sub237.lcssa = phi float [ %sub237, %for.inc252 ]
  %sub234.lcssa = phi float [ %sub234, %for.inc252 ]
  %55 = sub i32 %n, %div158
  br label %for.cond256.preheader

for.cond256.preheader:                            ; preds = %for.cond209.for.cond256.preheader_crit_edge, %for.cond209.preheader
  %x.5.lcssa = phi float [ %add254.lcssa, %for.cond209.for.cond256.preheader_crit_edge ], [ %x.4.lcssa, %for.cond209.preheader ]
  %D.3.lcssa = phi float [ %sub240.lcssa, %for.cond209.for.cond256.preheader_crit_edge ], [ %D.2.lcssa, %for.cond209.preheader ]
  %B.3.lcssa = phi float [ %sub237.lcssa, %for.cond209.for.cond256.preheader_crit_edge ], [ %B.2.lcssa, %for.cond209.preheader ]
  %A.3.lcssa = phi float [ %sub234.lcssa, %for.cond209.for.cond256.preheader_crit_edge ], [ %A.2.lcssa, %for.cond209.preheader ]
  %i.5.lcssa = phi i32 [ %55, %for.cond209.for.cond256.preheader_crit_edge ], [ %i.4.lcssa, %for.cond209.preheader ]
  %cmp257547 = icmp slt i32 %i.5.lcssa, %n
  br i1 %cmp257547, label %for.body259.lr.ph, label %for.end274

for.body259.lr.ph:                                ; preds = %for.cond256.preheader
  %56 = add i32 %n, -1
  br label %for.body259

if.end216:                                        ; preds = %for.inc252, %if.end216.lr.ph
  %add211550.pn = phi i32 [ %add211550, %if.end216.lr.ph ], [ %add211, %for.inc252 ]
  %x.5554 = phi float [ %x.4.lcssa, %if.end216.lr.ph ], [ %add254, %for.inc252 ]
  %i.5553 = phi i32 [ %i.4.lcssa, %if.end216.lr.ph ], [ %inc253, %for.inc252 ]
  %sub212556 = sub nsw i32 %add211550.pn, %fixed
  %arrayidx217 = getelementptr inbounds float, float* %1, i32 %add211550.pn
  %57 = load float, float* %arrayidx217, align 4, !tbaa !3
  %arrayidx218 = getelementptr inbounds float, float* %1, i32 %sub212556
  %58 = load float, float* %arrayidx218, align 4, !tbaa !3
  %sub219 = fsub float %57, %58
  %arrayidx220 = getelementptr inbounds float, float* %3, i32 %add211550.pn
  %59 = load float, float* %arrayidx220, align 4, !tbaa !3
  %arrayidx221 = getelementptr inbounds float, float* %3, i32 %sub212556
  %60 = load float, float* %arrayidx221, align 4, !tbaa !3
  %sub222 = fsub float %59, %60
  %arrayidx223 = getelementptr inbounds float, float* %5, i32 %add211550.pn
  %61 = load float, float* %arrayidx223, align 4, !tbaa !3
  %arrayidx224 = getelementptr inbounds float, float* %5, i32 %sub212556
  %62 = load float, float* %arrayidx224, align 4, !tbaa !3
  %sub225 = fsub float %61, %62
  %arrayidx226 = getelementptr inbounds float, float* %7, i32 %add211550.pn
  %63 = load float, float* %arrayidx226, align 4, !tbaa !3
  %arrayidx227 = getelementptr inbounds float, float* %7, i32 %sub212556
  %64 = load float, float* %arrayidx227, align 4, !tbaa !3
  %sub228 = fsub float %63, %64
  %arrayidx229 = getelementptr inbounds float, float* %9, i32 %add211550.pn
  %65 = load float, float* %arrayidx229, align 4, !tbaa !3
  %arrayidx230 = getelementptr inbounds float, float* %9, i32 %sub212556
  %66 = load float, float* %arrayidx230, align 4, !tbaa !3
  %sub231 = fsub float %65, %66
  %mul232 = fmul float %sub225, %sub228
  %mul233 = fmul float %sub222, %sub231
  %sub234 = fsub float %mul232, %mul233
  %mul235 = fmul float %sub219, %sub231
  %mul236 = fmul float %sub222, %sub228
  %sub237 = fsub float %mul235, %mul236
  %mul238 = fmul float %sub219, %sub225
  %mul239 = fmul float %sub222, %sub222
  %sub240 = fsub float %mul238, %mul239
  %mul241 = fmul float %x.5554, %sub237
  %add242 = fadd float %sub234, %mul241
  %div243 = fdiv float %add242, %sub240
  %sub244 = fsub float %div243, %offset
  %arrayidx245 = getelementptr inbounds float, float* %noise, i32 %i.5553
  %67 = load float, float* %arrayidx245, align 4, !tbaa !3
  %cmp246 = fcmp olt float %sub244, %67
  br i1 %cmp246, label %if.then248, label %for.inc252

if.then248:                                       ; preds = %if.end216
  store float %sub244, float* %arrayidx245, align 4, !tbaa !3
  br label %for.inc252

for.inc252:                                       ; preds = %if.end216, %if.then248
  %inc253 = add nsw i32 %i.5553, 1
  %add254 = fadd float %x.5554, 1.000000e+00
  %add211 = add nsw i32 %inc253, %div158
  %exitcond607 = icmp eq i32 %i.5553, %43
  br i1 %exitcond607, label %for.cond209.for.cond256.preheader_crit_edge, label %if.end216

for.body259:                                      ; preds = %for.inc271, %for.body259.lr.ph
  %x.6549 = phi float [ %x.5.lcssa, %for.body259.lr.ph ], [ %add273, %for.inc271 ]
  %i.6548 = phi i32 [ %i.5.lcssa, %for.body259.lr.ph ], [ %inc272, %for.inc271 ]
  %mul260 = fmul float %B.3.lcssa, %x.6549
  %add261 = fadd float %A.3.lcssa, %mul260
  %div262 = fdiv float %add261, %D.3.lcssa
  %sub263 = fsub float %div262, %offset
  %arrayidx264 = getelementptr inbounds float, float* %noise, i32 %i.6548
  %68 = load float, float* %arrayidx264, align 4, !tbaa !3
  %cmp265 = fcmp olt float %sub263, %68
  br i1 %cmp265, label %if.then267, label %for.inc271

if.then267:                                       ; preds = %for.body259
  store float %sub263, float* %arrayidx264, align 4, !tbaa !3
  br label %for.inc271

for.inc271:                                       ; preds = %for.body259, %if.then267
  %inc272 = add nsw i32 %i.6548, 1
  %add273 = fadd float %x.6549, 1.000000e+00
  %exitcond = icmp eq i32 %i.6548, %56
  br i1 %exitcond, label %for.end274.loopexit, label %for.body259

for.end274.loopexit:                              ; preds = %for.inc271
  br label %for.end274

for.end274:                                       ; preds = %for.end274.loopexit, %for.cond256.preheader, %for.end152
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 =  !{i32 1,  !"wchar_size", i32 4}
!1 =  !{i32 1,  !"min_enum_size", i32 4}
!2 =  !{ !"clang version 3.6.0 "}
!3 =  !{ !4,  !4, i64 0}
!4 =  !{ !"float",  !5, i64 0}
!5 =  !{ !"omnipotent char",  !6, i64 0}
!6 =  !{ !"Simple C/C++ TBAA"}
!7 =  !{ !8,  !8, i64 0}
!8 =  !{ !"long",  !5, i64 0}
