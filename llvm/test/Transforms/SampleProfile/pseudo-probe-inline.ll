; RUN: opt < %s -passes=pseudo-probe,sample-profile  -sample-profile-file=%S/Inputs/pseudo-probe-inline.prof -S -pass-remarks=sample-profile -sample-profile-prioritized-inline=0 -pass-remarks-output=%t.opt.yaml 2>&1 | FileCheck %s
; RUN: FileCheck %s -check-prefixes=YAML,YAML-NO-ANNOTATE < %t.opt.yaml

; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/pseudo-probe-inline.prof -o %t2
; RUN: opt < %s -passes=pseudo-probe,sample-profile  -sample-profile-file=%t2 -S -pass-remarks=sample-profile -sample-profile-prioritized-inline=0 -pass-remarks-output=%t2.opt.yaml 2>&1 | FileCheck %s
; RUN: FileCheck %s -check-prefixes=YAML,YAML-NO-ANNOTATE < %t2.opt.yaml

; RUN: opt < %s -passes=pseudo-probe,sample-profile -annotate-sample-profile-inline-phase=true -sample-profile-file=%S/Inputs/pseudo-probe-inline.prof -S -pass-remarks=sample-profile -sample-profile-prioritized-inline=0 -pass-remarks-output=%t3.opt.yaml 2>&1 | FileCheck %s
; RUN: FileCheck %s -check-prefixes=YAML,YAML-ANNOTATE < %t3.opt.yaml

; RUN: opt < %s -passes=pseudo-probe,sample-profile -annotate-sample-profile-inline-phase=true -sample-profile-file=%t2 -S -pass-remarks=sample-profile -sample-profile-prioritized-inline=0 -pass-remarks-output=%t4.opt.yaml 2>&1 | FileCheck %s
; RUN: FileCheck %s -check-prefixes=YAML,YAML-ANNOTATE < %t4.opt.yaml

@factor = dso_local global i32 3, align 4

define dso_local i32 @foo(i32 %x) #0 !dbg !12 {
entry:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID1:]], i64 1, i32 0, i64 -1)
  %add = add nsw i32 %x, 100000, !dbg !19
;; Check zen is fully inlined so there's no call to zen anymore.
;; Check code from the inlining of zen is properly annotated here.
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2:]], i64 1, i32 0, i64 -1)
; CHECK: br i1 %cmp.i, label %while.cond.i, label %while.cond2.i, !dbg ![[#]], !prof ![[PD1:[0-9]+]]
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 2, i32 0, i64 -1)
; CHECK: br i1 %cmp1.i, label %while.body.i, label %zen.exit, !dbg ![[#]], !prof ![[PD2:[0-9]+]]
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 3, i32 0, i64 -1)
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 4, i32 0, i64 -1)
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 5, i32 0, i64 -1)
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 6, i32 0, i64 -1)
; CHECK-NOT: call i32 @zen
  %call = call i32 @zen(i32 %add), !dbg !20
  ret i32 %call, !dbg !21
}

; CHECK: define dso_local i32 @zen
define dso_local i32 @zen(i32 %x) #0 !dbg !22 {
entry:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 1, i32 0, i64 -1)
  %cmp = icmp sgt i32 %x, 0, !dbg !26
  br i1 %cmp, label %while.cond, label %while.cond2, !dbg !28

while.cond:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 2, i32 0, i64 -1)
  %x.addr.0 = phi i32 [ %x, %entry ], [ %sub, %while.body ]
  %cmp1 = icmp sgt i32 %x.addr.0, 0, !dbg !29
  br i1 %cmp1, label %while.body, label %if.end, !dbg !31

while.body:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 3, i32 0, i64 -1)
  %0 = load volatile i32, ptr @factor, align 4, !dbg !32
  %sub = sub nsw i32 %x.addr.0, %0, !dbg !39
  br label %while.cond, !dbg !31

while.cond2:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 4, i32 0, i64 -1)
  %x.addr.1 = phi i32 [ %x, %entry ], [ %add, %while.body4 ]
  %cmp3 = icmp slt i32 %x.addr.1, 0, !dbg !42
  br i1 %cmp3, label %while.body4, label %if.end, !dbg !44

while.body4:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 5, i32 0, i64 -1)
  %1 = load volatile i32, ptr @factor, align 4, !dbg !45
  %add = add nsw i32 %x.addr.1, %1, !dbg !48
  br label %while.cond2, !dbg !44

if.end:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID2]], i64 6, i32 0, i64 -1)
  %x.addr.2 = phi i32 [ %x.addr.0, %while.cond ], [ %x.addr.1, %while.cond2 ]
  ret i32 %x.addr.2, !dbg !51
}

; CHECK: !llvm.pseudo_probe_desc = !{![[#DESC0:]], ![[#DESC1:]]}
; CHECK: ![[#DESC0]] = !{i64 [[#GUID1]], i64 [[#HASH1:]], !"foo"}
; CHECK: ![[#DESC1]] = !{i64 [[#GUID2]], i64 [[#HASH2:]], !"zen"}
; CHECK: ![[PD1]] = !{!"branch_weights", i32 5, i32 0}
; CHECK: ![[PD2]] = !{!"branch_weights", i32 382915, i32 5}

; Checking to see if YAML file is generated and contains remarks
;YAML: --- !Passed
;YAML-NO-ANNOTATE-NEXT:  Pass:            sample-profile-inline
;YAML-ANNOTATE-NEXT: Pass:    main-sample-profile-inline
;YAML-NEXT:  Name:            Inlined
;YAML-NEXT:  DebugLoc:        { File: test.cpp, Line: 10, Column: 11 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          ''''
;YAML-NEXT:    - Callee:          zen
;YAML-NEXT:      DebugLoc:        { File: test.cpp, Line: 38, Column: 0 }
;YAML-NEXT:    - String:          ''' inlined into '''
;YAML-NEXT:    - Caller:          foo
;YAML-NEXT:      DebugLoc:        { File: test.cpp, Line: 9, Column: 0 }
;YAML-NEXT:    - String:          ''''
;YAML-NEXT:    - String:          ' to match profiling context'
;YAML-NEXT:    - String:          ' with '
;YAML-NEXT:    - String:          '(cost='
;YAML-NEXT:    - Cost:            '15'
;YAML-NEXT:    - String:          ', threshold='
;YAML-NEXT:    - Threshold:       '3000'
;YAML-NEXT:    - String:          ')'
;YAML-NEXT:    - String:          ' at callsite '
;YAML-NEXT:    - String:          foo
;YAML-NEXT:    - String:          ':'
;YAML-NEXT:    - Line:            '1'
;YAML-NEXT:    - String:          ':'
;YAML-NEXT:    - Column:          '11'
;YAML-NEXT:    - String:          ';'
;YAML-NEXT:  ...
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: test.cpp, Line: 10, Column: 22 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '23'
;YAML-NEXT:    - String:          ' samples from profile (ProbeId='
;YAML-NEXT:    - ProbeId:         '1'
;YAML-NEXT:    - String:          ', Factor='
;YAML-NEXT:    - Factor:          '1.000000e+00'
;YAML-NEXT:    - String:          ', OriginalSamples='
;YAML-NEXT:    - OriginalSamples: '23'
;YAML-NEXT:    - String:          ')'
;YAML-NEXT:  ...
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: test.cpp, Line: 39, Column: 9 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '23'
;YAML-NEXT:    - String:          ' samples from profile (ProbeId='
;YAML-NEXT:    - ProbeId:         '1'
;YAML-NEXT:    - String:          ', Factor='
;YAML-NEXT:    - Factor:          '1.000000e+00'
;YAML-NEXT:    - String:          ', OriginalSamples='
;YAML-NEXT:    - OriginalSamples: '23'
;YAML-NEXT:    - String:          ')'
;YAML-NEXT:  ...
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: test.cpp, Line: 41, Column: 14 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '382920'
;YAML-NEXT:    - String:          ' samples from profile (ProbeId='
;YAML-NEXT:    - ProbeId:         '2'
;YAML-NEXT:    - String:          ', Factor='
;YAML-NEXT:    - Factor:          '1.000000e+00'
;YAML-NEXT:    - String:          ', OriginalSamples='
;YAML-NEXT:    - OriginalSamples: '382920'
;YAML-NEXT:    - String:          ')'
;YAML-NEXT:  ...

attributes #0 = {"use-sample-profile"}

!llvm.module.flags = !{!8, !9}

!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3)
!3 = !DIFile(filename: "test.cpp", directory: "test")
!4 = !{}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!12 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 9, type: !13, scopeLine: 9, unit: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{!7, !7}
!18 = !DILocation(line: 0, scope: !12)
!19 = !DILocation(line: 10, column: 22, scope: !12)
!20 = !DILocation(line: 10, column: 11, scope: !12)
!21 = !DILocation(line: 12, column: 3, scope: !12)
!22 = distinct !DISubprogram(name: "zen", scope: !3, file: !3, line: 37, type: !13, scopeLine: 38, unit: !2)
!25 = !DILocation(line: 0, scope: !22)
!26 = !DILocation(line: 39, column: 9, scope: !27)
!27 = distinct !DILexicalBlock(scope: !22, file: !3, line: 39, column: 7)
!28 = !DILocation(line: 39, column: 7, scope: !22)
!29 = !DILocation(line: 41, column: 14, scope: !30)
!30 = distinct !DILexicalBlock(scope: !27, file: !3, line: 39, column: 14)
!31 = !DILocation(line: 41, column: 5, scope: !30)
!32 = !DILocation(line: 42, column: 16, scope: !33)
!33 = distinct !DILexicalBlock(scope: !30, file: !3, line: 41, column: 19)
!38 = !DILocation(line: 42, column: 12, scope: !33)
!39 = !DILocation(line: 42, column: 9, scope: !33)
!42 = !DILocation(line: 48, column: 14, scope: !43)
!43 = distinct !DILexicalBlock(scope: !27, file: !3, line: 46, column: 8)
!44 = !DILocation(line: 48, column: 5, scope: !43)
!45 = !DILocation(line: 49, column: 16, scope: !46)
!46 = distinct !DILexicalBlock(scope: !43, file: !3, line: 48, column: 19)
!47 = !DILocation(line: 49, column: 12, scope: !46)
!48 = !DILocation(line: 49, column: 9, scope: !46)
!51 = !DILocation(line: 53, column: 3, scope: !22)
