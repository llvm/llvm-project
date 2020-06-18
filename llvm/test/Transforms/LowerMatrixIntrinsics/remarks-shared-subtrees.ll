; REQUIRES: aarch64-registered-target

; This test needs to be target specific due to the cost estimate in the output.

; RUN: opt -lower-matrix-intrinsics -pass-remarks-output=%t -pass-remarks=lower-matrix-intrinsics -mtriple=arm64-apple-iphoneos %s 2>&1 -disable-output | FileCheck --check-prefix=STDERR %s
; RUN: FileCheck --input-file=%t --check-prefix=YAML %s

; YAML-LABEL: --- !Passed
; YAML-NEXT:  Pass:            lower-matrix-intrinsics
; YAML-NEXT:  Name:            matrix-lowered
; YAML-NEXT:  DebugLoc:        { File: test.cpp, Line: 35, Column: 71 }
; YAML-NEXT:  Function:        test_2leafs
; YAML-NEXT:  Args:
; YAML-NEXT:    - String:          'Lowered with '
; YAML-NEXT:    - NumStores:       '4'
; YAML-NEXT:    - String:          ' stores, '
; YAML-NEXT:    - NumLoads:        '0'
; YAML-NEXT:    - String:          ' loads, '
; YAML-NEXT:    - NumComputeOps:   '0'
; YAML-NEXT:    - String:          ' compute ops'
; YAML-NEXT:    - String:          ',
; YAML-NEXT:  additionally '
; YAML-NEXT:    - NumStores:       '0'
; YAML-NEXT:    - String:          ' stores, '
; YAML-NEXT:    - NumLoads:        '4'
; YAML-NEXT:    - String:          ' loads, '
; YAML-NEXT:    - NumFPOps:        '16'
; YAML-NEXT:    - String:          ' compute ops'
; YAML-NEXT:    - String:          ' are shared with other expressions'
; YAML-NEXT:    - String:           |
; YAML:           column.major.store.4x2.double(
; YAML-NEXT:         shared with remark at line 35 column 45 (transpose.2x4.double(column.major.load.2x4.double(addr %arg1,
; YAML-NEXT:           scalar)),
; YAML-NEXT:         addr %arg3,
; YAML-NEXT:         10)

; YAML-LABEL: --- !Passed
; YAML-NEXT:  Pass:            lower-matrix-intrinsics
; YAML-NEXT:  Name:            matrix-lowered
; YAML-NEXT:  DebugLoc:        { File: test.cpp, Line: 35, Column: 45 }
; YAML-NEXT:  Function:        test_2leafs
; YAML-NEXT:  Args:
; YAML-NEXT:    - String:          'Lowered with '
; YAML-NEXT:    - NumStores:       '30'
; YAML-NEXT:    - String:          ' stores, '
; YAML-NEXT:    - NumLoads:        '45'
; YAML-NEXT:    - String:          ' loads, '
; YAML-NEXT:    - NumComputeOps:   '120'
; YAML-NEXT:    - String:          ' compute ops'
; YAML-NEXT:    - String:          ',
; YAML-NEXT:  additionally '
; YAML-NEXT:    - NumStores:       '0'
; YAML-NEXT:    - String:          ' stores, '
; YAML-NEXT:    - NumLoads:        '4'
; YAML-NEXT:    - String:          ' loads, '
; YAML-NEXT:    - NumFPOps:        '16'
; YAML-NEXT:    - String:          ' compute ops'
; YAML-NEXT:    - String:          ' are shared with other expressions'
; YAML-NEXT:    - String:           |
; YAML:            column.major.store.4x15.double(
; YAML-NEXT:         fsub(
; YAML-NEXT:          column.major.load.4x15.double(addr %arg2, 20),
; YAML-NEXT:          multiply.4x2.2x15.double(
; YAML-NEXT:           shared with remark at line 35 column 71 (transpose.2x4.double(column.major.load.2x4.double(addr %arg1,
; YAML-NEXT:             scalar)),
; YAML-NEXT:           column.major.load.2x15.double(addr %arg3, scalar))),
; YAML-NEXT:         addr %arg2,
; YAML-NEXT:         10)


; STDERR-LABEL: remark: test.cpp:35:71: Lowered with 4 stores, 0 loads, 0 compute ops,
; STDERR-NEXT:  additionally 0 stores, 4 loads, 16 compute ops are shared with other expressions
; STDERR-NEXT:  column.major.store.4x2.double(
; STDERR-NEXT:   shared with remark at line 35 column 45 (transpose.2x4.double(column.major.load.2x4.double(addr %arg1,
; STDERR-NEXT:     scalar)),
; STDERR-NEXT:   addr %arg3,
; STDERR-NEXT:   10)

; STDERR-LABEL: remark: test.cpp:35:45: Lowered with 30 stores, 45 loads, 120 compute ops,
; STDERR-NEXT:  additionally 0 stores, 4 loads, 16 compute ops are shared with other expressions
; STDERR-NEXT:  column.major.store.4x15.double(
; STDERR-NEXT:   fsub(
; STDERR-NEXT:    column.major.load.4x15.double(addr %arg2, 20),
; STDERR-NEXT:    multiply.4x2.2x15.double(
; STDERR-NEXT:     shared with remark at line 35 column 71 (transpose.2x4.double(column.major.load.2x4.double(addr %arg1,
; STDERR-NEXT:       scalar)),
; STDERR-NEXT:     column.major.load.2x15.double(addr %arg3, scalar))),
; STDERR-NEXT:   addr %arg2,
; STDERR-NEXT:   10)
define void @test_2leafs(double* %arg1, double* %arg2, double* %arg3, i64 %stride) !dbg !8 {
bb:
  %shared.load = tail call <8 x double> @llvm.matrix.column.major.load.v8f64.p0f64(double* %arg1, i64 %stride, i1 false, i32 2, i32 4), !dbg !10, !noalias !10
  %shared.load.2 = tail call <30 x double> @llvm.matrix.column.major.load.v30f64.p0f64(double* %arg3, i64 %stride, i1 false, i32 2, i32 15), !dbg !10, !noalias !10
  %tmp17 = tail call <8 x double> @llvm.matrix.transpose.v8f64(<8 x double> %shared.load, i32 2, i32 4), !dbg !10
  tail call void @llvm.matrix.column.major.store.v8f64.p0f64(<8 x double> %tmp17, double* %arg3, i64 10, i1 false, i32 4, i32 2), !dbg !10
  %tmp18 = tail call <60 x double> @llvm.matrix.column.major.load.v60f64.p0f64(double* %arg2, i64 20, i1 false, i32 4, i32 15), !dbg !11
  %tmp48 = tail call <60 x double> @llvm.matrix.multiply.v60f64.v8f64.v30f64(<8 x double> %tmp17, <30 x double> %shared.load.2, i32 4, i32 2, i32 15), !dbg !11
  %tmp49 = fsub <60 x double> %tmp18, %tmp48, !dbg !11
  tail call void @llvm.matrix.column.major.store.v60f64.p0f64(<60 x double> %tmp49, double* %arg2, i64 10, i1 false, i32 4, i32 15), !dbg !11
  ret void
}

declare <8 x double> @llvm.matrix.transpose.v8f64(<8 x double>, i32 immarg, i32 immarg)
declare <8 x double> @llvm.matrix.column.major.load.v8f64.p0f64(double*, i64, i1 immarg, i32 immarg, i32 immarg)
declare <30 x double> @llvm.matrix.column.major.load.v30f64.p0f64(double*, i64, i1 immarg, i32 immarg, i32 immarg)
declare <60 x double> @llvm.matrix.column.major.load.v60f64.p0f64(double*, i64, i1 immarg, i32 immarg, i32 immarg)
declare void @llvm.matrix.column.major.store.v60f64.p0f64(<60 x double>, double* writeonly, i64, i1 immarg, i32 immarg, i32 immarg)
declare void @llvm.matrix.column.major.store.v8f64.p0f64(<8 x double>, double* writeonly, i64, i1 immarg, i32 immarg, i32 immarg)
declare <60 x double> @llvm.matrix.multiply.v60f64.v8f64.v30f64(<8 x double>, <30 x double>, i32 immarg, i32 immarg, i32 immarg)

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.dbg.cu = !{!4}
!llvm.ident = !{!7}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 13, i32 0]}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 7, !"PIC Level", i32 2}
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !5, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !6, nameTableKind: GNU)
!5 = !DIFile(filename: "test.cpp", directory: "")
!6 = !{}
!7 = !{!"clang"}
!8 = distinct !DISubprogram(name: "test", scope: !5, file: !5, line: 26, type: !9, scopeLine: 27, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!9 = !DISubroutineType(types: !6)
!10 = distinct !DILocation(line: 35, column: 71, scope: !8)
!11 = distinct !DILocation(line: 35, column: 45, scope: !8)
!12 = !DILocation(line: 800, column: 17, scope: !13, inlinedAt: !15)
!13 = distinct !DISubprogram(name: "foo", scope: !14, file: !14, line: 789, type: !9, scopeLine: 790, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!14 = !DIFile(filename: "bar.h", directory: "bar")
!15 = distinct !DILocation(line: 1280, column: 5, scope: !16, inlinedAt: !18)
!16 = distinct !DISubprogram(name: "zar", scope: !17, file: !17, line: 1275, type: !9, scopeLine: 1278, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!17 = !DIFile(filename: "file1.h", directory: "dir1")
!18 = distinct !DILocation(line: 1278, column: 1, scope: !19, inlinedAt: !20)
!19 = distinct !DISubprogram(name: "yo", scope: !17, file: !17, line: 1275, type: !9, scopeLine: 1278, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!20 = distinct !DILocation(line: 2514, column: 26, scope: !21, inlinedAt: !22)
!21 = distinct !DISubprogram(name: "zzzz", scope: !14, file: !14, line: 2505, type: !9, scopeLine: 2506, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!22 = distinct !DILocation(line: 1263, column: 5, scope: !23, inlinedAt: !24)
!23 = distinct !DISubprogram(name: "ppppp", scope: !17, file: !17, line: 1258, type: !9, scopeLine: 1261, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!24 = distinct !DILocation(line: 1261, column: 1, scope: !25, inlinedAt: !26)
!25 = distinct !DISubprogram(name: "qqqq", scope: !17, file: !17, line: 1258, type: !9, scopeLine: 1261, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!26 = distinct !DILocation(line: 168, column: 7, scope: !27, inlinedAt: !29)
!27 = distinct !DISubprogram(name: "lll", scope: !28, file: !28, line: 166, type: !9, scopeLine: 169, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!28 = !DIFile(filename: "file2.h", directory: "dir2")
!29 = distinct !DILocation(line: 169, column: 1, scope: !30, inlinedAt: !31)
!30 = distinct !DISubprogram(name: "Expr1", scope: !28, file: !28, line: 166, type: !9, scopeLine: 169, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!31 = distinct !DILocation(line: 368, column: 12, scope: !32, inlinedAt: !33)
!32 = distinct !DISubprogram(name: "yyyyy", scope: !14, file: !14, line: 364, type: !9, scopeLine: 365, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!33 = distinct !DILocation(line: 1297, column: 34, scope: !34, inlinedAt: !35)
!34 = distinct !DISubprogram(name: "eeeee", scope: !14, file: !14, line: 1290, type: !9, scopeLine: 1291, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!35 = distinct !DILocation(line: 2306, column: 5, scope: !36, inlinedAt: !11)
!36 = distinct !DISubprogram(name: "aaaaa", scope: !37, file: !37, line: 2304, type: !9, scopeLine: 2305, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!37 = !DIFile(filename: "foo.c", directory: "/")
!38 = distinct !DISubprogram(name: "test2", scope: !5, file: !5, line: 90, type: !9, scopeLine: 27, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!39 = distinct !DILocation(line: 44, column: 44, scope: !38)
!40 = distinct !DILocation(line: 55, column: 55, scope: !38)
!41 = distinct !DILocation(line: 66, column: 66, scope: !38)
!42 = distinct !DISubprogram(name: "test2", scope: !5, file: !5, line: 90, type: !9, scopeLine: 27, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!43 = distinct !DILocation(line: 77, column: 77, scope: !42)
!44 = distinct !DILocation(line: 88, column: 88, scope: !42)
!45 = distinct !DISubprogram(name: "test2", scope: !5, file: !5, line: 90, type: !9, scopeLine: 27, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!46 = distinct !DILocation(line: 99, column: 99, scope: !45)
!47 = distinct !DILocation(line: 111, column: 111, scope: !45)
