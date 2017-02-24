; RUN: opt < %s -loop-vectorize -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s
; RUN: opt < %s -loop-vectorize -o /dev/null -pass-remarks-output=%t.yaml
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

; void test_disabled(int *A, int Length) {
; #pragma clang loop vectorize(disable) interleave(disable)
;   for (int i = 0; i < Length; i++)
;     A[i] = i;
; }

; void test_array_bounds(int *A, int *B, int Length) {
; #pragma clang loop vectorize(enable)
;   for (int i = 0; i < Length; i++)
;     A[i] = A[B[i]];
; }

; File, line, and column should match those specified in the metadata
; CHECK: remark: source.cpp:4:5: loop not vectorized: could not determine number of loop iterations
; CHECK: remark: source.cpp:4:5: loop not vectorized
; CHECK: remark: source.cpp:13:5: loop not vectorized: vectorization and interleaving are explicitly disabled, or vectorize width and interleave count are both set to 1
; CHECK: remark: source.cpp:19:5: loop not vectorized: cannot identify array bounds
; CHECK: remark: source.cpp:19:5: loop not vectorized
; CHECK: warning: source.cpp:19:5: loop not vectorized: failed explicitly specified loop vectorization

; CHECK: _Z4testPii
; CHECK-NOT: x i32>
; CHECK: ret

; CHECK: _Z13test_disabledPii
; CHECK-NOT: x i32>
; CHECK: ret

; CHECK: _Z17test_array_boundsPiS_i
; CHECK-NOT: x i32>
; CHECK: ret

; YAML:       --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            CantComputeNumberOfIterations
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 4, Column: 5 }
; YAML-NEXT: Function:        _Z4testPii
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          could not determine number of loop iterations
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            MissedDetails
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 4, Column: 5 }
; YAML-NEXT: Function:        _Z4testPii
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            AllDisabled
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 13, Column: 5 }
; YAML-NEXT: Function:        _Z13test_disabledPii
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: vectorization and interleaving are explicitly disabled, or vectorize width and interleave count are both set to 1'
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
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 19, Column: 5 }
; YAML-NEXT: Function:        _Z17test_array_boundsPiS_i
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT:   - String:          ' (Force='
; YAML-NEXT:   - Force:           'true'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT: ...
; YAML-NEXT: --- !Failure
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            FailedRequestedVectorization
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 19, Column: 5 }
; YAML-NEXT: Function:        _Z17test_array_boundsPiS_i
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          failed explicitly specified loop vectorization
; YAML-NEXT: ...

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z4testPii(i32* nocapture %A, i32 %Length) #0 !dbg !4 {
entry:
  %cmp10 = icmp sgt i32 %Length, 0, !dbg !12
  br i1 %cmp10, label %for.body, label %for.end, !dbg !12, !llvm.loop !14

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv, !dbg !16
  %0 = trunc i64 %indvars.iv to i32, !dbg !16
  %ld = load i32, i32* %arrayidx, align 4
  store i32 %0, i32* %arrayidx, align 4, !dbg !16, !tbaa !18
  %cmp3 = icmp sle i32 %ld, %Length, !dbg !22
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !12
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, %Length, !dbg !12
  %or.cond = and i1 %cmp3, %cmp, !dbg !22
  br i1 %or.cond, label %for.body, label %for.end, !dbg !22

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !24
}

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z13test_disabledPii(i32* nocapture %A, i32 %Length) #0 !dbg !7 {
entry:
  %cmp4 = icmp sgt i32 %Length, 0, !dbg !25
  br i1 %cmp4, label %for.body, label %for.end, !dbg !25, !llvm.loop !27

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv, !dbg !30
  %0 = trunc i64 %indvars.iv to i32, !dbg !30
  store i32 %0, i32* %arrayidx, align 4, !dbg !30, !tbaa !18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !25
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !25
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !25
  br i1 %exitcond, label %for.end, label %for.body, !dbg !25, !llvm.loop !27

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !31
}

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z17test_array_boundsPiS_i(i32* nocapture %A, i32* nocapture readonly %B, i32 %Length) #0 !dbg !8 {
entry:
  %cmp9 = icmp sgt i32 %Length, 0, !dbg !32
  br i1 %cmp9, label %for.body.preheader, label %for.end, !dbg !32, !llvm.loop !34

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !35

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv, !dbg !35
  %0 = load i32, i32* %arrayidx, align 4, !dbg !35, !tbaa !18
  %idxprom1 = sext i32 %0 to i64, !dbg !35
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %idxprom1, !dbg !35
  %1 = load i32, i32* %arrayidx2, align 4, !dbg !35, !tbaa !18
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv, !dbg !35
  store i32 %1, i32* %arrayidx4, align 4, !dbg !35, !tbaa !18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !32
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !32
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !32
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !32, !llvm.loop !34

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !36
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0", isOptimized: true, runtimeVersion: 6, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "source.cpp", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "test", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "source.cpp", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "test_disabled", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 10, file: !1, scope: !5, type: !6, variables: !2)
!8 = distinct !DISubprogram(name: "test_array_bounds", line: 16, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 16, file: !1, scope: !5, type: !6, variables: !2)
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
