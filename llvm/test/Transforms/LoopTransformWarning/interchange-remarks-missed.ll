; RUN: opt < %s -passes=transform-warning -disable-output -pass-remarks-missed=transform-warning -pass-remarks-analysis=transform-warning 2>&1 | FileCheck %s
; RUN: opt < %s -passes=transform-warning -disable-output -pass-remarks-output=%t.yaml
; RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s

; C/C++ code for tests
;
; float a[200][200];
; void f() {
;   #pragma clang loop interchange(enable)
;   for (int i = 0; i < 10; i++) {
;     for (int j = 0; j < 10; j++) {
;       a[j*j][i+5] += a[j+5][i*i];
;     }
;   }
; }

; CHECK: warning: source.c:6:3: loop not interchanged: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering

; YAML:     --- !Failure
; YAML-NEXT: Pass:            transform-warning
; YAML-NEXT: Name:            FailedRequestedInterchange
; YAML-NEXT: DebugLoc:        { File: source.c, Line: 6, Column: 3 }
; YAML-NEXT: Function:        test_interchange_enable
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not interchanged: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering'
; YAML-NEXT: ...

@a = dso_local local_unnamed_addr global [200 x [200 x float]] zeroinitializer, align 4

define dso_local void @test_interchange_enable() !dbg !18 {
entry:
  br label %for.cond1.preheader, !dbg !30

for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv27 = phi i64 [ 0, %entry ], [ %indvars.iv.next28, %for.cond.cleanup3 ]
  %0 = mul nuw nsw i64 %indvars.iv27, %indvars.iv27
  br label %for.body4, !dbg !32

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void, !dbg !33

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next28 = add nuw nsw i64 %indvars.iv27, 1, !dbg !34
  %exitcond31.not = icmp eq i64 %indvars.iv.next28, 10, !dbg !35
  br i1 %exitcond31.not, label %for.cond.cleanup, label %for.cond1.preheader, !dbg !30, !llvm.loop !36

for.body4:                                        ; preds = %for.cond1.preheader, %for.body4
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %arrayidx6 = getelementptr inbounds nuw [200 x [200 x float]], ptr @a, i64 0, i64 %indvars.iv, i64 %0, !dbg !41
  %1 = load float, ptr %arrayidx6, align 4, !dbg !41, !tbaa !44
  %2 = mul nuw nsw i64 %indvars.iv, %indvars.iv, !dbg !48
  %arrayidx11 = getelementptr inbounds nuw [200 x [200 x float]], ptr @a, i64 0, i64 %2, i64 %indvars.iv27, !dbg !49
  %3 = load float, ptr %arrayidx11, align 4, !dbg !50, !tbaa !44
  %add = fadd float %1, %3, !dbg !50
  store float %add, ptr %arrayidx11, align 4, !dbg !50, !tbaa !44
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !51
  %exitcond.not = icmp eq i64 %indvars.iv.next, 10, !dbg !52
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4, !dbg !32, !llvm.loop !53
}

!llvm.module.flags = !{!9, !10}

!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3)
!3 = !DIFile(filename: "source.c", directory: ".")
!9 = !{i32 7, !"Dwarf Version", i32 5}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!18 = distinct !DISubprogram(name: "test_interchange_enable", scope: !3, file: !3, line: 4, type: !19, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !{!22, !25}
!22 = !DILocalVariable(name: "i", scope: !23, file: !3, line: 6, type: !24)
!23 = distinct !DILexicalBlock(scope: !18, file: !3, line: 6, column: 3)
!24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!25 = !DILocalVariable(name: "j", scope: !26, file: !3, line: 7, type: !24)
!26 = distinct !DILexicalBlock(scope: !27, file: !3, line: 7, column: 5)
!27 = distinct !DILexicalBlock(scope: !28, file: !3, line: 6, column: 32)
!28 = distinct !DILexicalBlock(scope: !23, file: !3, line: 6, column: 3)
!29 = !DILocation(line: 0, scope: !23)
!30 = !DILocation(line: 6, column: 3, scope: !23)
!31 = !DILocation(line: 0, scope: !26)
!32 = !DILocation(line: 7, column: 5, scope: !26)
!33 = !DILocation(line: 11, column: 1, scope: !18)
!34 = !DILocation(line: 6, column: 28, scope: !28)
!35 = !DILocation(line: 6, column: 21, scope: !28)
!36 = distinct !{!36, !30, !37, !38, !39, !40}
!37 = !DILocation(line: 10, column: 3, scope: !23)
!38 = !{!"llvm.loop.mustprogress"}
!39 = !{!"llvm.loop.unroll.disable"}
!40 = !{!"llvm.loop.interchange.enable", i1 true}
!41 = !DILocation(line: 8, column: 20, scope: !42)
!42 = distinct !DILexicalBlock(scope: !43, file: !3, line: 7, column: 34)
!43 = distinct !DILexicalBlock(scope: !26, file: !3, line: 7, column: 5)
!44 = !{!45, !45, i64 0}
!45 = !{!"float", !46, i64 0}
!46 = !{!"omnipotent char", !47, i64 0}
!47 = !{!"Simple C/C++ TBAA"}
!48 = !DILocation(line: 8, column: 10, scope: !42)
!49 = !DILocation(line: 8, column: 7, scope: !42)
!50 = !DILocation(line: 8, column: 17, scope: !42)
!51 = !DILocation(line: 7, column: 30, scope: !43)
!52 = !DILocation(line: 7, column: 23, scope: !43)
!53 = distinct !{!53, !32, !54, !38, !39}
!54 = !DILocation(line: 9, column: 5, scope: !26)
