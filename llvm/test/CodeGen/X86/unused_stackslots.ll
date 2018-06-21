; PR26374: Check no stack slots are allocated for vregs which have no real reference.
; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ImageParameters = type { i32, i32, [0 x [16 x i16]] }
%struct.InputParameters = type { i32, i32 }

@c = common global %struct.ImageParameters* null, align 8
@a = common global i16** null, align 8
@d = common global [6 x i32] zeroinitializer, align 16
@b = common global %struct.InputParameters* null, align 8
@e = common global [4 x i32] zeroinitializer, align 16

; It is not easy to check there is no unused holes in stack allocated for spills,
; so simply check the size of stack allocated cannot exceed 350.
; (408 is used before the fix for PR26374. 344 is used after the fix).
;
; CHECK-LABEL: @fn
; CHECK: subq {{\$3[0-4][0-9]}}, %rsp

; Function Attrs: nounwind uwtable
define i32 @fn() #0 {
entry:
  %n = alloca [8 x [8 x i32]], align 16
  %tmp = bitcast [8 x [8 x i32]]* %n to i8*
  call void @llvm.lifetime.start.p0i8(i64 256, i8* %tmp) #3
  %tmp1 = bitcast [8 x [8 x i32]]* %n to i8*
  %arraydecay.1 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 1, i64 0
  %tmp2 = bitcast i32* %arraydecay.1 to i8*
  %arraydecay.2 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 2, i64 0
  %tmp3 = bitcast i32* %arraydecay.2 to i8*
  %arraydecay.3 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 3, i64 0
  %tmp4 = bitcast i32* %arraydecay.3 to i8*
  %arraydecay.4 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 4, i64 0
  %tmp5 = bitcast i32* %arraydecay.4 to i8*
  %arraydecay.5 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 5, i64 0
  %tmp6 = bitcast i32* %arraydecay.5 to i8*
  %arraydecay.6 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 6, i64 0
  %tmp7 = bitcast i32* %arraydecay.6 to i8*
  %arraydecay.7 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 7, i64 0
  %tmp8 = bitcast i32* %arraydecay.7 to i8*
  br label %for.body

for.body:                                         ; preds = %for.inc73, %entry
  %q.0131 = phi i32 [ 0, %entry ], [ %inc74, %for.inc73 ]
  %m.0130 = phi i32 [ 0, %entry ], [ %m.4, %for.inc73 ]
  %div = sdiv i32 %q.0131, 2
  %shl = shl i32 %div, 3
  %rem = srem i32 %q.0131, 2
  %shl1 = shl nsw i32 %rem, 3
  %tmp9 = sext i32 %shl1 to i64
  %tmp10 = sext i32 %shl to i64
  %tmp11 = or i32 %shl1, 4
  %tmp12 = sext i32 %tmp11 to i64
  %tmp13 = or i32 %shl, 4
  %tmp14 = sext i32 %tmp13 to i64
  br label %for.body4

for.body4:                                        ; preds = %for.inc48, %for.body
  %indvars.iv148 = phi i64 [ %tmp10, %for.body ], [ %indvars.iv.next149, %for.inc48 ]
  %m.1126 = phi i32 [ %m.0130, %for.body ], [ %m.3.lcssa, %for.inc48 ]
  %tmp15 = load %struct.ImageParameters*, %struct.ImageParameters** @c, align 8
  %opix_y = getelementptr inbounds %struct.ImageParameters, %struct.ImageParameters* %tmp15, i64 0, i32 1
  %tmp16 = load i32, i32* %opix_y, align 4
  %tmp17 = trunc i64 %indvars.iv148 to i32
  %add5 = add nsw i32 %tmp16, %tmp17
  %tmp18 = sub nuw nsw i64 %indvars.iv148, %tmp10
  %tmp19 = sext i32 %add5 to i64
  %tmp20 = add nsw i64 %tmp19, 1
  %tmp21 = or i64 %indvars.iv148, 1
  %tmp22 = or i64 %tmp18, 1
  %tmp23 = add nsw i64 %tmp19, 2
  %tmp24 = or i64 %indvars.iv148, 2
  %tmp25 = or i64 %tmp18, 2
  %tmp26 = add nsw i64 %tmp19, 3
  %tmp27 = or i64 %indvars.iv148, 3
  %tmp28 = or i64 %tmp18, 3
  br label %for.body9

for.body9:                                        ; preds = %for.inc45.for.body9_crit_edge, %for.body4
  %tmp29 = phi %struct.ImageParameters* [ %tmp15, %for.body4 ], [ %.pre, %for.inc45.for.body9_crit_edge ]
  %indvars.iv145 = phi i64 [ %tmp9, %for.body4 ], [ %indvars.iv.next146, %for.inc45.for.body9_crit_edge ]
  %m.2124 = phi i32 [ %m.1126, %for.body4 ], [ %m.3, %for.inc45.for.body9_crit_edge ]
  %opix_x = getelementptr inbounds %struct.ImageParameters, %struct.ImageParameters* %tmp29, i64 0, i32 0
  %tmp30 = load i32, i32* %opix_x, align 4
  %tmp31 = trunc i64 %indvars.iv145 to i32
  %add10 = add nsw i32 %tmp30, %tmp31
  tail call void @LumaPrediction4x4(i32 %tmp31, i32 %tmp17, i32 0, i32 0, i32 0, i16 signext 0, i16 signext 0) #3
  %tmp32 = load i16**, i16*** @a, align 8
  %tmp33 = load %struct.ImageParameters*, %struct.ImageParameters** @c, align 8
  %tmp34 = sub nuw nsw i64 %indvars.iv145, %tmp9
  %tmp35 = sext i32 %add10 to i64
  br label %for.cond14.preheader

for.cond14.preheader:                             ; preds = %for.body9
  %arrayidx = getelementptr inbounds i16*, i16** %tmp32, i64 %tmp19
  %tmp36 = load i16*, i16** %arrayidx, align 8
  %arrayidx20 = getelementptr inbounds i16, i16* %tmp36, i64 %tmp35
  %arrayidx26 = getelementptr inbounds %struct.ImageParameters, %struct.ImageParameters* %tmp33, i64 0, i32 2, i64 %indvars.iv148, i64 %indvars.iv145
  %arrayidx35 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 %tmp18, i64 %tmp34
  %tmp37 = bitcast i16* %arrayidx20 to <4 x i16>*
  %tmp38 = load <4 x i16>, <4 x i16>* %tmp37, align 2
  %tmp39 = zext <4 x i16> %tmp38 to <4 x i32>
  %tmp40 = bitcast i16* %arrayidx26 to <4 x i16>*
  %tmp41 = load <4 x i16>, <4 x i16>* %tmp40, align 2
  %tmp42 = zext <4 x i16> %tmp41 to <4 x i32>
  %tmp43 = sub nsw <4 x i32> %tmp39, %tmp42
  %tmp44 = bitcast i32* %arrayidx35 to <4 x i32>*
  store <4 x i32> %tmp43, <4 x i32>* %tmp44, align 16
  store <4 x i32> %tmp43, <4 x i32>* bitcast ([6 x i32]* @d to <4 x i32>*), align 16
  %arrayidx.1 = getelementptr inbounds i16*, i16** %tmp32, i64 %tmp20
  %tmp45 = load i16*, i16** %arrayidx.1, align 8
  %arrayidx20.1 = getelementptr inbounds i16, i16* %tmp45, i64 %tmp35
  %arrayidx26.1 = getelementptr inbounds %struct.ImageParameters, %struct.ImageParameters* %tmp33, i64 0, i32 2, i64 %tmp21, i64 %indvars.iv145
  %arrayidx35.1 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 %tmp22, i64 %tmp34
  %tmp46 = bitcast i16* %arrayidx20.1 to <4 x i16>*
  %tmp47 = load <4 x i16>, <4 x i16>* %tmp46, align 2
  %tmp48 = zext <4 x i16> %tmp47 to <4 x i32>
  %tmp49 = bitcast i16* %arrayidx26.1 to <4 x i16>*
  %tmp50 = load <4 x i16>, <4 x i16>* %tmp49, align 2
  %tmp51 = zext <4 x i16> %tmp50 to <4 x i32>
  %tmp52 = sub nsw <4 x i32> %tmp48, %tmp51
  %tmp53 = bitcast i32* %arrayidx35.1 to <4 x i32>*
  store <4 x i32> %tmp52, <4 x i32>* %tmp53, align 16
  store <4 x i32> %tmp52, <4 x i32>* bitcast (i32* getelementptr inbounds ([6 x i32], [6 x i32]* @d, i64 0, i64 4) to <4 x i32>*), align 16
  %arrayidx.2 = getelementptr inbounds i16*, i16** %tmp32, i64 %tmp23
  %tmp54 = load i16*, i16** %arrayidx.2, align 8
  %arrayidx20.2 = getelementptr inbounds i16, i16* %tmp54, i64 %tmp35
  %arrayidx26.2 = getelementptr inbounds %struct.ImageParameters, %struct.ImageParameters* %tmp33, i64 0, i32 2, i64 %tmp24, i64 %indvars.iv145
  %arrayidx35.2 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 %tmp25, i64 %tmp34
  %tmp55 = bitcast i16* %arrayidx20.2 to <4 x i16>*
  %tmp56 = load <4 x i16>, <4 x i16>* %tmp55, align 2
  %tmp57 = zext <4 x i16> %tmp56 to <4 x i32>
  %tmp58 = bitcast i16* %arrayidx26.2 to <4 x i16>*
  %tmp59 = load <4 x i16>, <4 x i16>* %tmp58, align 2
  %tmp60 = zext <4 x i16> %tmp59 to <4 x i32>
  %tmp61 = sub nsw <4 x i32> %tmp57, %tmp60
  %tmp62 = bitcast i32* %arrayidx35.2 to <4 x i32>*
  store <4 x i32> %tmp61, <4 x i32>* %tmp62, align 16
  store <4 x i32> %tmp61, <4 x i32>* bitcast (i32* getelementptr ([6 x i32], [6 x i32]* @d, i64 1, i64 2) to <4 x i32>*), align 16
  %arrayidx.3 = getelementptr inbounds i16*, i16** %tmp32, i64 %tmp26
  %tmp63 = load i16*, i16** %arrayidx.3, align 8
  %arrayidx20.3 = getelementptr inbounds i16, i16* %tmp63, i64 %tmp35
  %arrayidx26.3 = getelementptr inbounds %struct.ImageParameters, %struct.ImageParameters* %tmp33, i64 0, i32 2, i64 %tmp27, i64 %indvars.iv145
  %arrayidx35.3 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %n, i64 0, i64 %tmp28, i64 %tmp34
  %tmp64 = bitcast i16* %arrayidx20.3 to <4 x i16>*
  %tmp65 = load <4 x i16>, <4 x i16>* %tmp64, align 2
  %tmp66 = zext <4 x i16> %tmp65 to <4 x i32>
  %tmp67 = bitcast i16* %arrayidx26.3 to <4 x i16>*
  %tmp68 = load <4 x i16>, <4 x i16>* %tmp67, align 2
  %tmp69 = zext <4 x i16> %tmp68 to <4 x i32>
  %tmp70 = sub nsw <4 x i32> %tmp66, %tmp69
  %tmp71 = bitcast i32* %arrayidx35.3 to <4 x i32>*
  store <4 x i32> %tmp70, <4 x i32>* %tmp71, align 16
  store <4 x i32> %tmp70, <4 x i32>* bitcast (i32* getelementptr ([6 x i32], [6 x i32]* @d, i64 2, i64 0) to <4 x i32>*), align 16
  %tmp72 = load %struct.InputParameters*, %struct.InputParameters** @b, align 8
  %rdopt = getelementptr inbounds %struct.InputParameters, %struct.InputParameters* %tmp72, i64 0, i32 0
  %tmp73 = load i32, i32* %rdopt, align 4
  %cmp42 = icmp eq i32 %tmp73, 0
  br i1 %cmp42, label %land.lhs.true, label %if.then

land.lhs.true:                                    ; preds = %for.cond14.preheader
  %Transform8x8Mode = getelementptr inbounds %struct.InputParameters, %struct.InputParameters* %tmp72, i64 0, i32 1
  %tmp74 = load i32, i32* %Transform8x8Mode, align 4
  %tobool = icmp eq i32 %tmp74, 0
  br i1 %tobool, label %if.then, label %for.inc45

if.then:                                          ; preds = %land.lhs.true, %for.cond14.preheader
  %call = tail call i32 @distortion4x4(i32* nonnull getelementptr inbounds ([6 x i32], [6 x i32]* @d, i64 0, i64 0)) #3
  %add44 = add nsw i32 %call, %m.2124
  br label %for.inc45

for.inc45:                                        ; preds = %if.then, %land.lhs.true
  %m.3 = phi i32 [ %m.2124, %land.lhs.true ], [ %add44, %if.then ]
  %cmp8 = icmp slt i64 %indvars.iv145, %tmp12
  br i1 %cmp8, label %for.inc45.for.body9_crit_edge, label %for.inc48

for.inc45.for.body9_crit_edge:                    ; preds = %for.inc45
  %indvars.iv.next146 = add nsw i64 %indvars.iv145, 4
  %.pre = load %struct.ImageParameters*, %struct.ImageParameters** @c, align 8
  br label %for.body9

for.inc48:                                        ; preds = %for.inc45
  %m.3.lcssa = phi i32 [ %m.3, %for.inc45 ]
  %indvars.iv.next149 = add nsw i64 %indvars.iv148, 4
  %cmp3 = icmp slt i64 %indvars.iv148, %tmp14
  br i1 %cmp3, label %for.body4, label %for.end50

for.end50:                                        ; preds = %for.inc48
  %m.3.lcssa.lcssa = phi i32 [ %m.3.lcssa, %for.inc48 ]
  %tmp75 = load %struct.InputParameters*, %struct.InputParameters** @b, align 8
  %rdopt51 = getelementptr inbounds %struct.InputParameters, %struct.InputParameters* %tmp75, i64 0, i32 0
  %tmp76 = load i32, i32* %rdopt51, align 4
  %cmp52 = icmp eq i32 %tmp76, 0
  br i1 %cmp52, label %land.lhs.true54, label %for.inc73

land.lhs.true54:                                  ; preds = %for.end50
  %Transform8x8Mode55 = getelementptr inbounds %struct.InputParameters, %struct.InputParameters* %tmp75, i64 0, i32 1
  %tmp77 = load i32, i32* %Transform8x8Mode55, align 4
  %tobool56 = icmp eq i32 %tmp77, 0
  br i1 %tobool56, label %for.inc73, label %for.body61.preheader

for.body61.preheader:                             ; preds = %land.lhs.true54
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull bitcast (i32* getelementptr ([4 x i32], [4 x i32]* @e, i64 4, i64 0) to i8*), i8* %tmp1, i64 32, i32 16, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull bitcast (i32* getelementptr ([4 x i32], [4 x i32]* @e, i64 6, i64 0) to i8*), i8* %tmp2, i64 32, i32 16, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull bitcast (i32* getelementptr ([4 x i32], [4 x i32]* @e, i64 8, i64 0) to i8*), i8* %tmp3, i64 32, i32 16, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull bitcast (i32* getelementptr ([4 x i32], [4 x i32]* @e, i64 10, i64 0) to i8*), i8* %tmp4, i64 32, i32 16, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull bitcast (i32* getelementptr ([4 x i32], [4 x i32]* @e, i64 12, i64 0) to i8*), i8* %tmp5, i64 32, i32 16, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull bitcast (i32* getelementptr ([4 x i32], [4 x i32]* @e, i64 14, i64 0) to i8*), i8* %tmp6, i64 32, i32 16, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull bitcast (i32* getelementptr ([4 x i32], [4 x i32]* @e, i64 16, i64 0) to i8*), i8* %tmp7, i64 32, i32 16, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull bitcast (i32* getelementptr ([4 x i32], [4 x i32]* @e, i64 18, i64 0) to i8*), i8* %tmp8, i64 32, i32 16, i1 false)
  %call70 = tail call i32 @distortion4x4(i32* nonnull getelementptr inbounds ([4 x i32], [4 x i32]* @e, i64 0, i64 0)) #3
  %add71 = add nsw i32 %call70, %m.3.lcssa.lcssa
  br label %for.inc73

for.inc73:                                        ; preds = %for.body61.preheader, %land.lhs.true54, %for.end50
  %m.4 = phi i32 [ %add71, %for.body61.preheader ], [ %m.3.lcssa.lcssa, %land.lhs.true54 ], [ %m.3.lcssa.lcssa, %for.end50 ]
  %inc74 = add nuw nsw i32 %q.0131, 1
  %exitcond156 = icmp eq i32 %inc74, 4
  br i1 %exitcond156, label %for.end75, label %for.body

for.end75:                                        ; preds = %for.inc73
  %m.4.lcssa = phi i32 [ %m.4, %for.inc73 ]
  call void @llvm.lifetime.end.p0i8(i64 256, i8* %tmp) #3
  ret i32 %m.4.lcssa
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare void @LumaPrediction4x4(i32, i32, i32, i32, i32, i16 signext, i16 signext) #2

declare i32 @distortion4x4(i32*) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

