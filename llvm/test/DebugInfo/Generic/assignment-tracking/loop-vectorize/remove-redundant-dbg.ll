; RUN: opt %s -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=2 -S -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators %s -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=2 -S -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Check that loop-vectorize removes redundant debug intrinsics after it makes
;; a change. This has a significant positive impact on peak memory and compiler
;; run time.

;; Check there is only one dbg.assign.
; CHECK: call void @llvm.dbg.assign

;; Check that the loop was actually modified.
; CHECK: extractelement


define void @test1(ptr noalias nocapture %a, ptr noalias nocapture readonly %b) {
entry:
  call void @llvm.dbg.assign(metadata ptr %a, metadata !11, metadata !DIExpression(), metadata !16, metadata ptr undef, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.assign(metadata ptr %a, metadata !11, metadata !DIExpression(), metadata !16, metadata ptr undef, metadata !DIExpression()), !dbg !28
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp1 = fcmp ogt float %0, 1.000000e+02
  tail call void @llvm.assume(i1 %cmp1)
  %add = fadd float %0, 1.000000e+00
  %arrayidx5 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  store float %add, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare void @llvm.assume(i1) #0
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0)"}
!7 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "Counter", scope: !7, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = distinct !DIAssignID()
!19 = distinct !DILexicalBlock(scope: !7, file: !1, line: 4, column: 3)
!28 = !DILocation(line: 6, column: 1, scope: !7)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
