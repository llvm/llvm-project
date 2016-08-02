; RUN: opt < %s -loop-vectorize -pass-remarks-analysis='loop-vectorize' -mtriple=x86_64-unknown-linux -S 2>&1 | FileCheck %s

; Verify analysis remarks are generated when interleaving is not beneficial.
; CHECK: remark: vectorization-remarks-profitable.c:5:17: the cost-model indicates that vectorization is not beneficial
; CHECK: remark: vectorization-remarks-profitable.c:5:17: the cost-model indicates that interleaving is not beneficial and is explicitly disabled or interleave count is set to 1
; CHECK: remark: vectorization-remarks-profitable.c:12:17: the cost-model indicates that vectorization is not beneficial
; CHECK: remark: vectorization-remarks-profitable.c:12:17: the cost-model indicates that interleaving is not beneficial

; First loop.
;  #pragma clang loop interleave(disable) unroll(disable)
;  for(int i = 0; i < n; i++) {
;    out[i] = *in[i];
;  }

; Second loop.
;  #pragma clang loop unroll(disable)
;  for(int i = 0; i < n; i++) {
;    out[i] = *in[i];
;  }

; ModuleID = 'vectorization-remarks-profitable.ll'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind uwtable
define void @do_not_interleave(float** noalias nocapture readonly %in, float* noalias nocapture %out, i32 %size) #0 !dbg !4 {
entry:
  %cmp.4 = icmp eq i32 %size, 0, !dbg !10
  br i1 %cmp.4, label %for.end, label %for.body.preheader, !dbg !11

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !12

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float*, float** %in, i64 %indvars.iv, !dbg !12
  %0 = bitcast float** %arrayidx to i32**, !dbg !12
  %1 = load i32*, i32** %0, align 8, !dbg !12
  %2 = load i32, i32* %1, align 4, !dbg !13
  %arrayidx2 = getelementptr inbounds float, float* %out, i64 %indvars.iv, !dbg !14
  %3 = bitcast float* %arrayidx2 to i32*, !dbg !15
  store i32 %2, i32* %3, align 4, !dbg !15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !11
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !11
  %exitcond = icmp eq i32 %lftr.wideiv, %size, !dbg !11
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !11, !llvm.loop !16

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end, !dbg !19

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !19
}

; Function Attrs: nounwind uwtable
define void @interleave_not_profitable(float** noalias nocapture readonly %in, float* noalias nocapture %out, i32 %size) #0 !dbg !6 {
entry:
  %cmp.4 = icmp eq i32 %size, 0, !dbg !20
  br i1 %cmp.4, label %for.end, label %for.body, !dbg !21

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float*, float** %in, i64 %indvars.iv, !dbg !22
  %0 = bitcast float** %arrayidx to i32**, !dbg !22
  %1 = load i32*, i32** %0, align 8, !dbg !22
  %2 = load i32, i32* %1, align 4, !dbg !23
  %arrayidx2 = getelementptr inbounds float, float* %out, i64 %indvars.iv, !dbg !24
  %3 = bitcast float* %arrayidx2 to i32*, !dbg !25
  store i32 %2, i32* %3, align 4, !dbg !25
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !21
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !21
  %exitcond = icmp eq i32 %lftr.wideiv, %size, !dbg !21
  br i1 %exitcond, label %for.end, label %for.body, !dbg !21, !llvm.loop !26

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !27
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 250016)", isOptimized: false, runtimeVersion: 0, emissionKind: 2, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "vectorization-remarks-profitable.c", directory: "")
!2 = !{}
!3 = !{!4, !6}
!4 = distinct !DISubprogram(name: "do_not_interleave", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !2)
!6 = distinct !DISubprogram(name: "interleave_not_profitable", scope: !1, file: !1, line: 8, type: !5, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.8.0 (trunk 250016)"}
!10 = !DILocation(line: 4, column: 23, scope: !4)
!11 = !DILocation(line: 4, column: 3, scope: !4)
!12 = !DILocation(line: 5, column: 17, scope: !4)
!13 = !DILocation(line: 5, column: 16, scope: !4)
!14 = !DILocation(line: 5, column: 7, scope: !4)
!15 = !DILocation(line: 5, column: 14, scope: !4)
!16 = distinct !{!16, !17, !18}
!17 = !{!"llvm.loop.interleave.count", i32 1}
!18 = !{!"llvm.loop.unroll.disable"}
!19 = !DILocation(line: 6, column: 1, scope: !4)
!20 = !DILocation(line: 11, column: 23, scope: !6)
!21 = !DILocation(line: 11, column: 3, scope: !6)
!22 = !DILocation(line: 12, column: 17, scope: !6)
!23 = !DILocation(line: 12, column: 16, scope: !6)
!24 = !DILocation(line: 12, column: 7, scope: !6)
!25 = !DILocation(line: 12, column: 14, scope: !6)
!26 = distinct !{!26, !18}
!27 = !DILocation(line: 13, column: 1, scope: !6)

