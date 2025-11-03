; RUN: opt < %s -passes=loop-vectorize,transform-warning -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize'  2>&1 | FileCheck %s
; RUN: opt < %s -passes=loop-vectorize,transform-warning -o /dev/null -pass-remarks-output=%t.yaml
; RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s

; C/C++ code for tests
; void test(int *A, int Length) {
; #pragma clang loop vectorize(enable) interleave(enable)
;   for (int i = 0; i < Length; i++) {
;     A[i] = i;
;     if (A[i] > Length)
;       break;
;   }
; }
; File, line, and column should match those specified in the metadata
; CHECK: remark: source.cpp:5:9: loop not vectorized: Cannot vectorize uncountable loop
; CHECK: remark: source.cpp:5:9: loop not vectorized

; void test_disabled(int *A, int Length) {
; #pragma clang loop vectorize(disable) interleave(disable)
;   for (int i = 0; i < Length; i++)
;     A[i] = i;
; }
; CHECK: remark: source.cpp:12:8: loop not vectorized: vectorization and interleaving are explicitly disabled, or the loop has already been vectorized

; void test_array_bounds(int *A, int *B, int Length) {
; #pragma clang loop vectorize(enable)
;   for (int i = 0; i < Length; i++)
;     A[i] = A[B[i]];
; }
; CHECK: remark: source.cpp:19:5: loop not vectorized: cannot identify array bounds
; CHECK: remark: source.cpp:18:8: loop not vectorized (Force=true)
; CHECK: warning: source.cpp:18:8: loop not vectorized: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering

; int foo();
; void test_multiple_failures(int *A) {
;   int k = 0;
; #pragma clang loop vectorize(enable) interleave(enable)
;   for (int i = 0; i < 1000; i+=A[i]) {
;     if (A[i])
;       k = foo();
;   }
;   return k;
; }
; CHECK: remark: source.cpp:29:7: loop not vectorized: Control flow cannot be substituted for a select
; CHECK: remark: source.cpp:27:3: loop not vectorized

; YAML:       --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            UnsupportedUncountableLoop
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 5, Column: 9 }
; YAML-NEXT: Function:        _Z4testPii
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          Cannot vectorize uncountable loop
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            MissedDetails
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 5, Column: 9 }
; YAML-NEXT: Function:        _Z4testPii
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            AllDisabled
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 12, Column: 8 }
; YAML-NEXT: Function:        _Z13test_disabledPii
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: vectorization and interleaving are explicitly disabled, or the loop has already been vectorized
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            ''
; YAML-NEXT: Name:            CantIdentifyArrayBounds
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 19, Column: 5 }
; YAML-NEXT: Function:        _Z17test_array_boundsPiS_i
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          cannot identify array bounds
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            MissedDetails
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 18, Column: 8 }
; YAML-NEXT: Function:        _Z17test_array_boundsPiS_i
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT:   - String:          ' (Force='
; YAML-NEXT:   - Force:           'true'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT: ...
; YAML-NEXT: --- !Failure
; YAML-NEXT: Pass:            transform-warning
; YAML-NEXT: Name:            FailedRequestedVectorization
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 18, Column: 8 }
; YAML-NEXT: Function:        _Z17test_array_boundsPiS_i
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering'
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            NoCFGForSelect
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 29, Column: 7 }
; YAML-NEXT: Function:        test_multiple_failures
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          Control flow cannot be substituted for a select
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            NonReductionValueUsedOutsideLoop
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 27, Column: 3 }
; YAML-NEXT: Function:        test_multiple_failures
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          value that could not be identified as reduction is used outside the loop
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            NonReductionValueUsedOutsideLoop
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 27, Column: 3 }
; YAML-NEXT: Function:        test_multiple_failures
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          value that could not be identified as reduction is used outside the loop
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            CantVectorizeLibcall
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 29, Column: 11 }
; YAML-NEXT: Function:        test_multiple_failures
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          call instruction cannot be vectorized
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            NoInductionVariable
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 27, Column: 3 }
; YAML-NEXT: Function:        test_multiple_failures
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          loop induction variable could not be identified
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            UnsupportedUncountableLoop
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 27, Column: 3 }
; YAML-NEXT: Function:        test_multiple_failures
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          Cannot vectorize uncountable loop
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            CantComputeNumberOfIterations
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 27, Column: 3 }
; YAML-NEXT: Function:        test_multiple_failures
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          could not determine number of loop iterations
; YAML-NEXT: ...
; YAML:      --- !Missed
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            MissedDetails
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 27, Column: 3 }
; YAML-NEXT: Function:        test_multiple_failures
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT: ...

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z4testPii(ptr nocapture %A, i32 %Length) #0 !dbg !4 {
entry:
  %cmp10 = icmp sgt i32 %Length, 0, !dbg !12
  br i1 %cmp10, label %for.body, label %for.end, !dbg !12, !llvm.loop !14

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !16
  %0 = trunc i64 %indvars.iv to i32, !dbg !16
  %ld = load i32, ptr %arrayidx, align 4
  store i32 %0, ptr %arrayidx, align 4, !dbg !16, !tbaa !18
  %cmp3 = icmp sle i32 %ld, %Length, !dbg !22
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !12
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, %Length, !dbg !12
  %or.cond = and i1 %cmp3, %cmp, !dbg !22
  br i1 %or.cond, label %for.body, label %for.end, !dbg !22

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !24
}

; CHECK: _Z4testPii
; CHECK-NOT: x i32>
; CHECK: ret

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z13test_disabledPii(ptr nocapture %A, i32 %Length) #0 !dbg !7 {
entry:
  %cmp4 = icmp sgt i32 %Length, 0, !dbg !25
  br i1 %cmp4, label %for.body, label %for.end, !dbg !25, !llvm.loop !27

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !30
  %0 = trunc i64 %indvars.iv to i32, !dbg !30
  store i32 %0, ptr %arrayidx, align 4, !dbg !30, !tbaa !18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !25
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !25
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !25
  br i1 %exitcond, label %for.end, label %for.body, !dbg !25, !llvm.loop !27

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !31
}

; CHECK: _Z13test_disabledPii
; CHECK-NOT: x i32>
; CHECK: ret

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z17test_array_boundsPiS_i(ptr nocapture %A, ptr nocapture readonly %B, i32 %Length) #0 !dbg !8 {
entry:
  %cmp9 = icmp sgt i32 %Length, 0, !dbg !32
  br i1 %cmp9, label %for.body.preheader, label %for.end, !dbg !32, !llvm.loop !34

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !32

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv, !dbg !35
  %0 = load i32, ptr %arrayidx, align 4, !dbg !35, !tbaa !18
  %idxprom1 = sext i32 %0 to i64, !dbg !35
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %idxprom1, !dbg !35
  %1 = load i32, ptr %arrayidx2, align 4, !dbg !35, !tbaa !18
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !35
  store i32 %1, ptr %arrayidx4, align 4, !dbg !35, !tbaa !18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !32
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !32
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !32
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !32, !llvm.loop !34

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !36
}

; CHECK: _Z17test_array_boundsPiS_i
; CHECK-NOT: x i32>
; CHECK: ret

; Function Attrs: nounwind uwtable
define i32 @test_multiple_failures(ptr nocapture readonly %A) #0 !dbg !46 {
entry:
  br label %for.body, !dbg !38

for.body:                                         ; preds = %entry, %for.inc
  %i.09 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %k.09 = phi i32 [ 0, %entry ], [ %k.1, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.09, !dbg !40
  %0 = load i32, ptr %arrayidx, align 4, !dbg !40
  %tobool = icmp eq i32 %0, 0, !dbg !40
  br i1 %tobool, label %for.inc, label %if.then, !dbg !40

if.then:                                          ; preds = %for.body
  %call = tail call i32 (...) @foo(), !dbg !41
  %.pre = load i32, ptr %arrayidx, align 4
  br label %for.inc, !dbg !42

for.inc:                                          ; preds = %for.body, %if.then
  %1 = phi i32 [ %.pre, %if.then ], [ 0, %for.body ], !dbg !43
  %k.1 = phi i32 [ %call, %if.then ], [ %k.09, %for.body ]
  %add = add nsw i32 %1, %i.09, !dbg !44
  %cmp = icmp slt i32 %add, 1000, !dbg !45
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !38

for.cond.cleanup:                                 ; preds = %for.inc
  ret i32 %k.1, !dbg !39
}

declare i32 @foo(...)

; CHECK: test_multiple_failure
; CHECK-NOT: x i32>
; CHECK: ret

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0", isOptimized: true, runtimeVersion: 6, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "source.cpp", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "test", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "source.cpp", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "test_disabled", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 10, file: !1, scope: !5, type: !6, retainedNodes: !2)
!8 = distinct !DISubprogram(name: "test_array_bounds", line: 16, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 16, file: !1, scope: !5, type: !6, retainedNodes: !2)
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.5.0"}
!12 = !DILocation(line: 3, column: 8, scope: !13)
!13 = distinct !DILexicalBlock(line: 3, column: 3, file: !1, scope: !4)
!14 = !{!14, !15, !15}
!15 = !{!"llvm.loop.vectorize.enable", i1 true}
!16 = !DILocation(line: 4, column: 5, scope: !17)
!17 = distinct !DILexicalBlock(line: 3, column: 36, file: !1, scope: !13)
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 5, column: 9, scope: !23)
!23 = distinct !DILexicalBlock(line: 5, column: 9, file: !1, scope: !17)
!24 = !DILocation(line: 8, column: 1, scope: !4)
!25 = !DILocation(line: 12, column: 8, scope: !26)
!26 = distinct !DILexicalBlock(line: 12, column: 3, file: !1, scope: !7)
!27 = !{!27, !28, !29}
!28 = !{!"llvm.loop.interleave.count", i32 1}
!29 = !{!"llvm.loop.vectorize.width", i32 1}
!30 = !DILocation(line: 13, column: 5, scope: !26)
!31 = !DILocation(line: 14, column: 1, scope: !7)
!32 = !DILocation(line: 18, column: 8, scope: !33)
!33 = distinct !DILexicalBlock(line: 18, column: 3, file: !1, scope: !8)
!34 = !{!34, !15}
!35 = !DILocation(line: 19, column: 5, scope: !33)
!36 = !DILocation(line: 20, column: 1, scope: !8)
!37 = distinct !DILexicalBlock(line: 24, column: 3, file: !1, scope: !46)
!38 = !DILocation(line: 27, column: 3, scope: !37)
!39 = !DILocation(line: 31, column: 3, scope: !37)
!40 = !DILocation(line: 28, column: 9, scope: !37)
!41 = !DILocation(line: 29, column: 11, scope: !37)
!42 = !DILocation(line: 29, column: 7, scope: !37)
!43 = !DILocation(line: 27, column: 32, scope: !37)
!44 = !DILocation(line: 27, column: 30, scope: !37)
!45 = !DILocation(line: 27, column: 21, scope: !37)
!46 = distinct !DISubprogram(name: "test_multiple_failures", line: 26, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 26, file: !1, scope: !5, type: !6, retainedNodes: !2)
