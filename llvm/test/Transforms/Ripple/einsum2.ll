; RUN: opt -passes='module(function(mem2reg,mergereturn),ripple,function(dce))' -S %s | FileCheck %s --implicit-check-not="warning:"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable
define dso_local void @_Z22nkctv_nkvw_nctw_dram_0PKhS0_Phfififimmmmmm(ptr noundef readonly captures(none) %aptr, ptr noundef readonly captures(none) %bptr, ptr noundef writeonly captures(none) %cptr, float noundef %a_scale, i32 noundef %a_offset, float noundef %b_scale, i32 noundef %b_offset, float noundef %c_scale, i32 noundef %c_offset, i64 noundef %N, i64 noundef %K, i64 noundef %C, i64 noundef %T, i64 noundef %V, i64 noundef %W) local_unnamed_addr #0 {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 8, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %0 = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %1 = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 1)
  %2 = tail call i64 @llvm.ripple.block.getsize.i64(ptr %BS, i64 0)
  %3 = tail call i64 @llvm.ripple.block.getsize.i64(ptr %BS, i64 1)
  %cmp389.not = icmp eq i64 %N, 0
  br i1 %cmp389.not, label %for.cond.cleanup, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp2387.not = icmp eq i64 %C, 0
  %cmp6.not352 = icmp ugt i64 %3, %T
  %invariant.gep343 = getelementptr i8, ptr %bptr, i64 %0
  %invariant.gep345 = getelementptr i8, ptr %cptr, i64 %0
  %cmp10.not347 = icmp ugt i64 %2, %W
  %cmp13339.not = icmp eq i64 %K, 0
  %cmp17336.not = icmp eq i64 %V, 0
  %conv46 = sitofp i32 %c_offset to float
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.lr.ph, %for.cond.cleanup3
  %n.0390 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %inc200, %for.cond.cleanup3 ]
  br i1 %cmp2387.not, label %for.cond.cleanup3, label %for.cond5.preheader.lr.ph

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %mul = mul i64 %n.0390, %K
  %mul51 = mul i64 %n.0390, %C
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void

for.cond5.preheader:                              ; preds = %for.cond5.preheader.lr.ph, %if.end195
  %c.0388 = phi i64 [ 0, %for.cond5.preheader.lr.ph ], [ %inc197, %if.end195 ]
  br i1 %cmp6.not352, label %for.end63, label %for.cond8.preheader.lr.ph

for.cond8.preheader.lr.ph:                        ; preds = %for.cond5.preheader
  %add52 = add i64 %c.0388, %mul51
  %mul53 = mul i64 %add52, %T
  %invariant.op356 = add i64 %1, %mul53
  br label %for.cond8.preheader

for.cond.cleanup3:                                ; preds = %if.end195, %for.cond1.preheader
  %inc200 = add nuw i64 %n.0390, 1
  %exitcond399.not = icmp eq i64 %inc200, %N
  br i1 %exitcond399.not, label %for.cond.cleanup, label %for.cond1.preheader, !llvm.loop !5

for.cond5.loopexit:                               ; preds = %for.cond.cleanup14, %for.cond8.preheader
  %add = add i64 %add354, %3
  %cmp6.not = icmp ugt i64 %add, %T
  br i1 %cmp6.not, label %for.end63, label %for.cond8.preheader, !llvm.loop !7

for.cond8.preheader:                              ; preds = %for.cond8.preheader.lr.ph, %for.cond5.loopexit
  %add354 = phi i64 [ %3, %for.cond8.preheader.lr.ph ], [ %add, %for.cond5.loopexit ]
  %t.0353 = phi i64 [ 0, %for.cond8.preheader.lr.ph ], [ %add354, %for.cond5.loopexit ]
  br i1 %cmp10.not347, label %for.cond5.loopexit, label %for.cond12.preheader.lr.ph

for.cond12.preheader.lr.ph:                       ; preds = %for.cond8.preheader
  %add20 = add i64 %t.0353, %1
  %add54.reass.reass = add i64 %t.0353, %invariant.op356
  %mul55 = mul i64 %add54.reass.reass, %W
  %invariant.gep350 = getelementptr i8, ptr %invariant.gep345, i64 %mul55
  br label %for.cond12.preheader

for.cond12.preheader:                             ; preds = %for.cond12.preheader.lr.ph, %for.cond.cleanup14
  %add9349 = phi i64 [ %2, %for.cond12.preheader.lr.ph ], [ %add9, %for.cond.cleanup14 ]
  %w.0348 = phi i64 [ 0, %for.cond12.preheader.lr.ph ], [ %add9349, %for.cond.cleanup14 ]
  br i1 %cmp13339.not, label %for.cond.cleanup14, label %for.cond16.preheader.lr.ph

for.cond16.preheader.lr.ph:                       ; preds = %for.cond12.preheader
  %gep344 = getelementptr i8, ptr %invariant.gep343, i64 %w.0348
  br label %for.cond16.preheader

for.cond16.preheader:                             ; preds = %for.cond16.preheader.lr.ph, %for.cond.cleanup18
  %k.0341 = phi i64 [ 0, %for.cond16.preheader.lr.ph ], [ %inc44, %for.cond.cleanup18 ]
  %acc.0340 = phi float [ 0.000000e+00, %for.cond16.preheader.lr.ph ], [ %acc.1.lcssa, %for.cond.cleanup18 ]
  br i1 %cmp17336.not, label %for.cond.cleanup18, label %for.body19.lr.ph

for.body19.lr.ph:                                 ; preds = %for.cond16.preheader
  %add21 = add i64 %k.0341, %mul
  %mul22 = mul i64 %add21, %C
  %add23 = add i64 %mul22, %c.0388
  %mul24 = mul i64 %add23, %T
  %add25 = add i64 %add20, %mul24
  %mul26 = mul i64 %add25, %V
  %invariant.gep = getelementptr i8, ptr %aptr, i64 %mul26
  %mul33 = mul i64 %add21, %V
  br label %for.body19

for.cond.cleanup14:                               ; preds = %for.cond.cleanup18, %for.cond12.preheader
  %acc.0.lcssa = phi float [ 0.000000e+00, %for.cond12.preheader ], [ %acc.1.lcssa, %for.cond.cleanup18 ]
  %div = fdiv float %acc.0.lcssa, %c_scale
  %add47 = fadd float %div, %conv46
  %cmp.i332 = fcmp olt float %add47, 0.000000e+00
  %cmp1.i333 = fcmp ogt float %add47, 2.550000e+02
  %.x.i334 = select i1 %cmp1.i333, float 2.550000e+02, float %add47
  %retval.0.i335 = select i1 %cmp.i332, float 0.000000e+00, float %.x.i334
  %conv48 = fptoui float %retval.0.i335 to i8
  %gep351 = getelementptr i8, ptr %invariant.gep350, i64 %w.0348
  store i8 %conv48, ptr %gep351, align 1, !tbaa !8
  %add9 = add i64 %add9349, %2
  %cmp10.not = icmp ugt i64 %add9, %W
  br i1 %cmp10.not, label %for.cond5.loopexit, label %for.cond12.preheader, !llvm.loop !11

for.cond.cleanup18:                               ; preds = %for.body19, %for.cond16.preheader
  %acc.1.lcssa = phi float [ %acc.0340, %for.cond16.preheader ], [ %6, %for.body19 ]
  %inc44 = add nuw i64 %k.0341, 1
  %exitcond393.not = icmp eq i64 %inc44, %K
  br i1 %exitcond393.not, label %for.cond.cleanup14, label %for.cond16.preheader, !llvm.loop !12

for.body19:                                       ; preds = %for.body19.lr.ph, %for.body19
  %v.0338 = phi i64 [ 0, %for.body19.lr.ph ], [ %inc, %for.body19 ]
  %acc.1337 = phi float [ %acc.0340, %for.body19.lr.ph ], [ %6, %for.body19 ]
  %gep = getelementptr i8, ptr %invariant.gep, i64 %v.0338
  %4 = load i8, ptr %gep, align 1, !tbaa !8
  %conv = zext i8 %4 to i32
  %sub = sub nsw i32 %conv, %a_offset
  %conv28 = sitofp i32 %sub to float
  %mul29 = fmul float %a_scale, %conv28
  %add34 = add i64 %v.0338, %mul33
  %mul35 = mul i64 %add34, %W
  %arrayidx37 = getelementptr i8, ptr %gep344, i64 %mul35
  %5 = load i8, ptr %arrayidx37, align 1, !tbaa !8
  %conv38 = zext i8 %5 to i32
  %sub39 = sub nsw i32 %conv38, %b_offset
  %conv40 = sitofp i32 %sub39 to float
  %mul41 = fmul float %b_scale, %conv40
  %6 = tail call float @llvm.fmuladd.f32(float %mul29, float %mul41, float %acc.1337)
  %inc = add nuw i64 %v.0338, 1
  %exitcond.not = icmp eq i64 %inc, %V
  br i1 %exitcond.not, label %for.cond.cleanup18, label %for.body19, !llvm.loop !13

for.end63:                                        ; preds = %for.cond5.loopexit, %for.cond5.preheader
  %t.0.lcssa = phi i64 [ 0, %for.cond5.preheader ], [ %add354, %for.cond5.loopexit ]
  %add64 = add i64 %t.0.lcssa, %1
  %cmp65 = icmp ult i64 %add64, %T
  br i1 %cmp65, label %for.cond67.preheader, label %if.end195

for.cond67.preheader:                             ; preds = %for.end63
  br i1 %cmp10.not347, label %for.end132, label %for.cond73.preheader.lr.ph

for.cond73.preheader.lr.ph:                       ; preds = %for.cond67.preheader
  %add124 = add i64 %c.0388, %mul51
  %mul125 = mul i64 %add124, %T
  %add126 = add i64 %add64, %mul125
  %mul127 = mul i64 %add126, %W
  %invariant.gep375 = getelementptr i8, ptr %invariant.gep345, i64 %mul127
  br label %for.cond73.preheader

for.cond73.preheader:                             ; preds = %for.cond73.preheader.lr.ph, %for.cond.cleanup75
  %add68373 = phi i64 [ %2, %for.cond73.preheader.lr.ph ], [ %add68, %for.cond.cleanup75 ]
  %w66.0372 = phi i64 [ 0, %for.cond73.preheader.lr.ph ], [ %add68373, %for.cond.cleanup75 ]
  br i1 %cmp13339.not, label %for.cond.cleanup75, label %for.cond78.preheader.lr.ph

for.cond78.preheader.lr.ph:                       ; preds = %for.cond73.preheader
  %gep368 = getelementptr i8, ptr %invariant.gep343, i64 %w66.0372
  br label %for.cond78.preheader

for.cond78.preheader:                             ; preds = %for.cond78.preheader.lr.ph, %for.cond.cleanup80
  %k72.0365 = phi i64 [ 0, %for.cond78.preheader.lr.ph ], [ %inc113, %for.cond.cleanup80 ]
  %acc71.0364 = phi float [ 0.000000e+00, %for.cond78.preheader.lr.ph ], [ %acc71.1.lcssa, %for.cond.cleanup80 ]
  br i1 %cmp17336.not, label %for.cond.cleanup80, label %for.body81.lr.ph

for.body81.lr.ph:                                 ; preds = %for.cond78.preheader
  %add84 = add i64 %k72.0365, %mul
  %mul85 = mul i64 %add84, %C
  %add86 = add i64 %mul85, %c.0388
  %mul87 = mul i64 %add86, %T
  %add88 = add i64 %mul87, %add64
  %mul89 = mul i64 %add88, %V
  %invariant.gep361 = getelementptr i8, ptr %aptr, i64 %mul89
  %mul99 = mul i64 %add84, %V
  br label %for.body81

for.cond.cleanup75:                               ; preds = %for.cond.cleanup80, %for.cond73.preheader
  %acc71.0.lcssa = phi float [ 0.000000e+00, %for.cond73.preheader ], [ %acc71.1.lcssa, %for.cond.cleanup80 ]
  %div116 = fdiv float %acc71.0.lcssa, %c_scale
  %add118 = fadd float %div116, %conv46
  %cmp.i328 = fcmp olt float %add118, 0.000000e+00
  %cmp1.i329 = fcmp ogt float %add118, 2.550000e+02
  %.x.i330 = select i1 %cmp1.i329, float 2.550000e+02, float %add118
  %retval.0.i331 = select i1 %cmp.i328, float 0.000000e+00, float %.x.i330
  %conv120 = fptoui float %retval.0.i331 to i8
  %gep376 = getelementptr i8, ptr %invariant.gep375, i64 %w66.0372
  store i8 %conv120, ptr %gep376, align 1, !tbaa !8
  %add68 = add i64 %add68373, %2
  %cmp69.not = icmp ugt i64 %add68, %W
  br i1 %cmp69.not, label %for.end132, label %for.cond73.preheader, !llvm.loop !14

for.cond.cleanup80:                               ; preds = %for.body81, %for.cond78.preheader
  %acc71.1.lcssa = phi float [ %acc71.0364, %for.cond78.preheader ], [ %9, %for.body81 ]
  %inc113 = add nuw i64 %k72.0365, 1
  %exitcond395.not = icmp eq i64 %inc113, %K
  br i1 %exitcond395.not, label %for.cond.cleanup75, label %for.cond78.preheader, !llvm.loop !15

for.body81:                                       ; preds = %for.body81.lr.ph, %for.body81
  %v77.0359 = phi i64 [ 0, %for.body81.lr.ph ], [ %inc110, %for.body81 ]
  %acc71.1358 = phi float [ %acc71.0364, %for.body81.lr.ph ], [ %9, %for.body81 ]
  %gep362 = getelementptr i8, ptr %invariant.gep361, i64 %v77.0359
  %7 = load i8, ptr %gep362, align 1, !tbaa !8
  %conv92 = zext i8 %7 to i32
  %sub93 = sub nsw i32 %conv92, %a_offset
  %conv94 = sitofp i32 %sub93 to float
  %mul95 = fmul float %a_scale, %conv94
  %add100 = add i64 %v77.0359, %mul99
  %mul101 = mul i64 %add100, %W
  %arrayidx103 = getelementptr i8, ptr %gep368, i64 %mul101
  %8 = load i8, ptr %arrayidx103, align 1, !tbaa !8
  %conv104 = zext i8 %8 to i32
  %sub105 = sub nsw i32 %conv104, %b_offset
  %conv106 = sitofp i32 %sub105 to float
  %mul107 = fmul float %b_scale, %conv106
  %9 = tail call float @llvm.fmuladd.f32(float %mul95, float %mul107, float %acc71.1358)
  %inc110 = add nuw i64 %v77.0359, 1
  %exitcond394.not = icmp eq i64 %inc110, %V
  br i1 %exitcond394.not, label %for.cond.cleanup80, label %for.body81, !llvm.loop !16

for.end132:                                       ; preds = %for.cond.cleanup75, %for.cond67.preheader
  %w66.0.lcssa = phi i64 [ 0, %for.cond67.preheader ], [ %add68373, %for.cond.cleanup75 ]
  %add133 = add i64 %w66.0.lcssa, %0
  %cmp134 = icmp ult i64 %add133, %W
  br i1 %cmp134, label %for.cond138.preheader, label %if.end195

for.cond138.preheader:                            ; preds = %for.end132
  br i1 %cmp13339.not, label %for.cond.cleanup140, label %for.cond143.preheader.lr.ph

for.cond143.preheader.lr.ph:                      ; preds = %for.cond138.preheader
  %10 = getelementptr i8, ptr %bptr, i64 %add133
  br label %for.cond143.preheader

for.cond143.preheader:                            ; preds = %for.cond143.preheader.lr.ph, %for.cond.cleanup145
  %k137.0385 = phi i64 [ 0, %for.cond143.preheader.lr.ph ], [ %inc178, %for.cond.cleanup145 ]
  %acc136.0384 = phi float [ 0.000000e+00, %for.cond143.preheader.lr.ph ], [ %acc136.1.lcssa, %for.cond.cleanup145 ]
  br i1 %cmp17336.not, label %for.cond.cleanup145, label %for.body146.lr.ph

for.body146.lr.ph:                                ; preds = %for.cond143.preheader
  %add149 = add i64 %k137.0385, %mul
  %mul150 = mul i64 %add149, %C
  %add151 = add i64 %mul150, %c.0388
  %mul152 = mul i64 %add151, %T
  %add153 = add i64 %mul152, %add64
  %mul154 = mul i64 %add153, %V
  %invariant.gep381 = getelementptr i8, ptr %aptr, i64 %mul154
  %mul164 = mul i64 %add149, %V
  br label %for.body146

for.cond.cleanup140:                              ; preds = %for.cond.cleanup145, %for.cond138.preheader
  %acc136.0.lcssa = phi float [ 0.000000e+00, %for.cond138.preheader ], [ %acc136.1.lcssa, %for.cond.cleanup145 ]
  %div181 = fdiv float %acc136.0.lcssa, %c_scale
  %add183 = fadd float %div181, %conv46
  %cmp.i = fcmp olt float %add183, 0.000000e+00
  %cmp1.i = fcmp ogt float %add183, 2.550000e+02
  %.x.i = select i1 %cmp1.i, float 2.550000e+02, float %add183
  %retval.0.i = select i1 %cmp.i, float 0.000000e+00, float %.x.i
  %conv185 = fptoui float %retval.0.i to i8
  %add189 = add i64 %c.0388, %mul51
  %mul190 = mul i64 %add189, %T
  %add191 = add i64 %add64, %mul190
  %mul192 = mul i64 %add191, %W
  %11 = getelementptr i8, ptr %cptr, i64 %add133
  %arrayidx194 = getelementptr i8, ptr %11, i64 %mul192
  store i8 %conv185, ptr %arrayidx194, align 1, !tbaa !8
  br label %if.end195

for.cond.cleanup145:                              ; preds = %for.body146, %for.cond143.preheader
  %acc136.1.lcssa = phi float [ %acc136.0384, %for.cond143.preheader ], [ %14, %for.body146 ]
  %inc178 = add nuw i64 %k137.0385, 1
  %exitcond397.not = icmp eq i64 %inc178, %K
  br i1 %exitcond397.not, label %for.cond.cleanup140, label %for.cond143.preheader, !llvm.loop !17

for.body146:                                      ; preds = %for.body146.lr.ph, %for.body146
  %v142.0379 = phi i64 [ 0, %for.body146.lr.ph ], [ %inc175, %for.body146 ]
  %acc136.1378 = phi float [ %acc136.0384, %for.body146.lr.ph ], [ %14, %for.body146 ]
  %gep382 = getelementptr i8, ptr %invariant.gep381, i64 %v142.0379
  %12 = load i8, ptr %gep382, align 1, !tbaa !8
  %conv157 = zext i8 %12 to i32
  %sub158 = sub nsw i32 %conv157, %a_offset
  %conv159 = sitofp i32 %sub158 to float
  %mul160 = fmul float %a_scale, %conv159
  %add165 = add i64 %v142.0379, %mul164
  %mul166 = mul i64 %add165, %W
  %arrayidx168 = getelementptr i8, ptr %10, i64 %mul166
  %13 = load i8, ptr %arrayidx168, align 1, !tbaa !8
  %conv169 = zext i8 %13 to i32
  %sub170 = sub nsw i32 %conv169, %b_offset
  %conv171 = sitofp i32 %sub170 to float
  %mul172 = fmul float %b_scale, %conv171
  %14 = tail call float @llvm.fmuladd.f32(float %mul160, float %mul172, float %acc136.1378)
  %inc175 = add nuw i64 %v142.0379, 1
  %exitcond396.not = icmp eq i64 %inc175, %V
  br i1 %exitcond396.not, label %for.cond.cleanup145, label %for.body146, !llvm.loop !18

if.end195:                                        ; preds = %for.end132, %for.cond.cleanup140, %for.end63
  %inc197 = add nuw i64 %c.0388, 1
  %exitcond398.not = icmp eq i64 %inc197, %C
  br i1 %exitcond398.not, label %for.cond.cleanup3, label %for.cond5.preheader, !llvm.loop !19
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.getsize.i64(ptr, i64 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !{!11, !6}
!12 = distinct !{!12, !6}
!13 = distinct !{!13, !6}
!14 = distinct !{!14, !6}
!15 = distinct !{!15, !6}
!16 = distinct !{!16, !6}
!17 = distinct !{!17, !6}
!18 = distinct !{!18, !6}
!19 = distinct !{!19, !6}
