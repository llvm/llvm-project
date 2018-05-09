; RUN: opt < %s -loop-vectorize -mtriple=x86_64-unknown-linux -S -pass-remarks='loop-vectorize' -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s

; CHECK: remark: no_fpmath.c:6:11: loop not vectorized: cannot prove it is safe to reorder floating-point operations
; CHECK: remark: no_fpmath.c:6:14: loop not vectorized
; CHECK: remark: no_fpmath.c:17:14: vectorized loop (vectorization width: 2, interleaved count: 2)

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind readonly ssp uwtable
define double @cond_sum(i32* nocapture readonly %v, i32 %n) #0 !dbg !4 {
entry:
  %cmp.7 = icmp sgt i32 %n, 0, !dbg !3
  br i1 %cmp.7, label %for.body.preheader, label %for.cond.cleanup, !dbg !8

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !9

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add.lcssa = phi double [ %add, %for.body ]
  br label %for.cond.cleanup, !dbg !10

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %a.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret double %a.0.lcssa, !dbg !10

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %a.08 = phi double [ %add, %for.body ], [ 0.000000e+00, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %v, i64 %indvars.iv, !dbg !9
  %0 = load i32, i32* %arrayidx, align 4, !dbg !9, !tbaa !11
  %cmp1 = icmp eq i32 %0, 0, !dbg !15
  %cond = select i1 %cmp1, double 3.400000e+00, double 1.150000e+00, !dbg !9
  %add = fadd double %a.08, %cond, !dbg !16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !8
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !8
  %exitcond = icmp eq i32 %lftr.wideiv, %n, !dbg !8
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body, !dbg !8, !llvm.loop !17
}

; Function Attrs: nounwind readonly ssp uwtable
define double @cond_sum_loop_hint(i32* nocapture readonly %v, i32 %n) #0 !dbg !20 {
entry:
  %cmp.7 = icmp sgt i32 %n, 0, !dbg !19
  br i1 %cmp.7, label %for.body.preheader, label %for.cond.cleanup, !dbg !21

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !22

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add.lcssa = phi double [ %add, %for.body ]
  br label %for.cond.cleanup, !dbg !23

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %a.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret double %a.0.lcssa, !dbg !23

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %a.08 = phi double [ %add, %for.body ], [ 0.000000e+00, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %v, i64 %indvars.iv, !dbg !22
  %0 = load i32, i32* %arrayidx, align 4, !dbg !22, !tbaa !11
  %cmp1 = icmp eq i32 %0, 0, !dbg !24
  %cond = select i1 %cmp1, double 3.400000e+00, double 1.150000e+00, !dbg !22
  %add = fadd double %a.08, %cond, !dbg !25
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !21
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !21
  %exitcond = icmp eq i32 %lftr.wideiv, %n, !dbg !21
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body, !dbg !21, !llvm.loop !26
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!28}
!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"PIC Level", i32 2}
!2 = !{!"clang version 3.7.0"}
!3 = !DILocation(line: 5, column: 20, scope: !4)
!4 = distinct !DISubprogram(name: "cond_sum", scope: !5, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !28, retainedNodes: !7)
!5 = !DIFile(filename: "no_fpmath.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 5, column: 3, scope: !4)
!9 = !DILocation(line: 6, column: 14, scope: !4)
!10 = !DILocation(line: 9, column: 3, scope: !4)
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !DILocation(line: 6, column: 19, scope: !4)
!16 = !DILocation(line: 6, column: 11, scope: !4)
!17 = distinct !{!17, !18}
!18 = !{!"llvm.loop.unroll.disable"}
!19 = !DILocation(line: 16, column: 20, scope: !20)
!20 = distinct !DISubprogram(name: "cond_sum_loop_hint", scope: !5, file: !5, line: 12, type: !6, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: true, unit: !28, retainedNodes: !7)
!21 = !DILocation(line: 16, column: 3, scope: !20)
!22 = !DILocation(line: 17, column: 14, scope: !20)
!23 = !DILocation(line: 20, column: 3, scope: !20)
!24 = !DILocation(line: 17, column: 19, scope: !20)
!25 = !DILocation(line: 17, column: 11, scope: !20)
!26 = distinct !{!26, !27, !18}
!27 = !{!"llvm.loop.vectorize.enable", i1 true}
!28 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !5,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
