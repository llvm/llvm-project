; RUN: opt < %s -passes=unpredictable-profile-loader -unpredictable-hints-file=%S/Inputs/inline.misp.prof -unpredictable-hints-frequency-profile=%S/Inputs/inline.freq.prof -unpredictable-hints-min-ratio=0.1 -S | FileCheck %s
; RUN: opt < %s -passes=unpredictable-profile-loader -unpredictable-hints-file=%S/Inputs/inline.misp.prof -unpredictable-hints-frequency-profile=%S/Inputs/inline.freq.prof -unpredictable-hints-min-ratio=0.5 -S | FileCheck --check-prefixes=MIN %s

; Test that we can apply branch mispredict profile data when the branch of
; interest in `callee` has been inlined into `caller`.

; // Original C source:
; static int callee(double *A, double *B) {
;   int count = 0;
;   for(int i=0; i<1000000; ++i)
;     if(A[i] > 100)
;       count += B[i] * 3;
;
;   return count;
; }
;
; int caller(double *X, double *Y) {
;   return callee(X, Y);
; }

; CHECK-LABEL: @caller
define dso_local i32 @caller(ptr nocapture noundef readonly %X, ptr nocapture noundef readonly %Y) local_unnamed_addr !dbg !7 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %X, metadata !14, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata ptr %Y, metadata !15, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata ptr %X, metadata !17, metadata !DIExpression()), !dbg !24
  tail call void @llvm.dbg.value(metadata ptr %Y, metadata !20, metadata !DIExpression()), !dbg !24
  tail call void @llvm.dbg.value(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !24
  tail call void @llvm.dbg.value(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !26
  br label %for.body.i, !dbg !27

for.body.i:                                       ; preds = %for.inc.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.inc.i ]
  %count.09.i = phi i32 [ 0, %entry ], [ %count.1.i, %for.inc.i ]
  tail call void @llvm.dbg.value(metadata i64 %indvars.iv.i, metadata !22, metadata !DIExpression()), !dbg !26
  tail call void @llvm.dbg.value(metadata i32 %count.09.i, metadata !21, metadata !DIExpression()), !dbg !24
  %arrayidx.i = getelementptr inbounds double, ptr %X, i64 %indvars.iv.i, !dbg !28
  %0 = load double, ptr %arrayidx.i, align 8, !dbg !28
  %cmp1.i = fcmp reassoc nsz arcp contract afn ogt double %0, 1.000000e+02, !dbg !35
; CHECK: br i1 %cmp1.i, label %if.then.i, label %for.inc.i
; CHECK-SAME: !unpredictable
; MIN: br i1 %cmp1.i, label %if.then.i, label %for.inc.i
; MIN-NOT: !unpredictable
  br i1 %cmp1.i, label %if.then.i, label %for.inc.i, !dbg !36

if.then.i:                                        ; preds = %for.body.i
  %arrayidx3.i = getelementptr inbounds double, ptr %Y, i64 %indvars.iv.i, !dbg !37
  %1 = load double, ptr %arrayidx3.i, align 8, !dbg !37
  %mul.i = fmul reassoc nsz arcp contract afn double %1, 3.000000e+00, !dbg !38
  %conv.i = sitofp i32 %count.09.i to double, !dbg !39
  %add.i = fadd reassoc nsz arcp contract afn double %mul.i, %conv.i, !dbg !39
  %conv4.i = fptosi double %add.i to i32, !dbg !39
  tail call void @llvm.dbg.value(metadata i32 %conv4.i, metadata !21, metadata !DIExpression()), !dbg !24
  br label %for.inc.i, !dbg !40

for.inc.i:                                        ; preds = %if.then.i, %for.body.i
  %count.1.i = phi i32 [ %conv4.i, %if.then.i ], [ %count.09.i, %for.body.i ]
  tail call void @llvm.dbg.value(metadata i32 %count.1.i, metadata !21, metadata !DIExpression()), !dbg !24
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1, !dbg !41
  tail call void @llvm.dbg.value(metadata i64 %indvars.iv.next.i, metadata !22, metadata !DIExpression()), !dbg !26
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 1000000, !dbg !42
  br i1 %exitcond.not.i, label %callee.exit, label %for.body.i, !dbg !27

callee.exit:                                      ; preds = %for.inc.i
  ret i32 %count.1.i, !dbg !47
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1)
!1 = !DIFile(filename: "inlined.c", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!7 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "X", arg: 1, scope: !7, file: !1, line: 10, type: !11)
!15 = !DILocalVariable(name: "Y", arg: 2, scope: !7, file: !1, line: 10, type: !11)
!16 = !DILocation(line: 0, scope: !7)
!17 = !DILocalVariable(name: "A", arg: 1, scope: !18, file: !1, line: 1, type: !11)
!18 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!19 = !{!17, !20, !21, !22}
!20 = !DILocalVariable(name: "B", arg: 2, scope: !18, file: !1, line: 1, type: !11)
!21 = !DILocalVariable(name: "count", scope: !18, file: !1, line: 2, type: !10)
!22 = !DILocalVariable(name: "i", scope: !23, file: !1, line: 3, type: !10)
!23 = distinct !DILexicalBlock(scope: !18, file: !1, line: 3, column: 3)
!24 = !DILocation(line: 0, scope: !18, inlinedAt: !25)
!25 = distinct !DILocation(line: 11, column: 10, scope: !7)
!26 = !DILocation(line: 0, scope: !23, inlinedAt: !25)
!27 = !DILocation(line: 3, column: 3, scope: !23, inlinedAt: !25)
!28 = !DILocation(line: 4, column: 8, scope: !29, inlinedAt: !25)
!29 = distinct !DILexicalBlock(scope: !30, file: !1, line: 4, column: 8)
!30 = distinct !DILexicalBlock(scope: !23, file: !1, line: 3, column: 3)
!35 = !DILocation(line: 4, column: 13, scope: !29, inlinedAt: !25)
!36 = !DILocation(line: 4, column: 8, scope: !30, inlinedAt: !25)
!37 = !DILocation(line: 5, column: 16, scope: !29, inlinedAt: !25)
!38 = !DILocation(line: 5, column: 21, scope: !29, inlinedAt: !25)
!39 = !DILocation(line: 5, column: 13, scope: !29, inlinedAt: !25)
!40 = !DILocation(line: 5, column: 7, scope: !29, inlinedAt: !25)
!41 = !DILocation(line: 3, column: 27, scope: !30, inlinedAt: !25)
!42 = !DILocation(line: 3, column: 17, scope: !30, inlinedAt: !25)
!44 = !DILocation(line: 5, column: 23, scope: !23, inlinedAt: !25)
!47 = !DILocation(line: 11, column: 3, scope: !7)
