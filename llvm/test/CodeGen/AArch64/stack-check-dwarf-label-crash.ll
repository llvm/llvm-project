; RUN: llc < %s -mtriple=arm64-apple-darwin -verify-machineinstrs | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

%struct.foo = type { %struct.pluto }
%struct.pluto = type { i8*, i8*, %struct.wombat, %struct.wombat, %struct.widget, %struct.hoge, i8, i8, i8, %struct.barney.1 }
%struct.wombat = type { %struct.barney**, %struct.barney**, %struct.barney**, [32 x %struct.barney*] }
%struct.barney = type <{ i32 (...)**, i8, i8, i8, i8, [4 x i8] }>
%struct.widget = type { %struct.barney**, %struct.barney**, %struct.barney**, [8 x %struct.barney*] }
%struct.hoge = type { %struct.pluto.0**, %struct.pluto.0**, %struct.pluto.0**, [4 x %struct.pluto.0*] }
%struct.pluto.0 = type <{ %struct.spam, [4 x i8], i64, %struct.barney*, i8, [7 x i8] }>
%struct.spam = type <{ i32 (...)**, i8, i8, i8, i8 }>
%struct.barney.1 = type { %struct.ham }
%struct.ham = type { [4096 x i8], %struct.bar* }
%struct.bar = type { %struct.bar*, i64 }

; Check we don't crash on this.
; CHECK-LABEL: barney
define void @barney() local_unnamed_addr #0 !dbg !3 {
bb:
  %tmp = alloca %struct.foo, align 8
  br i1 undef, label %bb2, label %bb1, !dbg !7

bb1:                                              ; preds = %bb
  br i1 undef, label %bb2, label %bb4, !dbg !8

bb2:                                              ; preds = %bb1, %bb
  br i1 undef, label %bb34, label %bb3, !dbg !9

bb3:                                              ; preds = %bb2
  br label %bb34, !dbg !10

bb4:                                              ; preds = %bb1
  br i1 undef, label %bb18, label %bb5, !dbg !11

bb5:                                              ; preds = %bb4
  br i1 undef, label %bb6, label %bb7, !dbg !12

bb6:                                              ; preds = %bb5
  br i1 undef, label %bb18, label %bb8, !dbg !16

bb7:                                              ; preds = %bb5
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6
  br i1 undef, label %bb10, label %bb9, !dbg !17

bb9:                                              ; preds = %bb8
  br label %bb10, !dbg !21

bb10:                                             ; preds = %bb9, %bb8
  br i1 undef, label %bb11, label %bb12, !dbg !22

bb11:                                             ; preds = %bb10
  br label %bb15, !dbg !22

bb12:                                             ; preds = %bb10
  br i1 undef, label %bb14, label %bb13, !dbg !27

bb13:                                             ; preds = %bb12
  br label %bb15, !dbg !27

bb14:                                             ; preds = %bb12
  unreachable, !dbg !28

bb15:                                             ; preds = %bb13, %bb11
  br i1 undef, label %bb17, label %bb16, !dbg !29

bb16:                                             ; preds = %bb15
  br label %bb17, !dbg !30

bb17:                                             ; preds = %bb16, %bb15
  br label %bb18

bb18:                                             ; preds = %bb17, %bb6, %bb4
  br i1 undef, label %bb20, label %bb19, !dbg !31

bb19:                                             ; preds = %bb18
  br label %bb20, !dbg !32

bb20:                                             ; preds = %bb19, %bb18
  br i1 undef, label %bb25, label %bb21, !dbg !33

bb21:                                             ; preds = %bb20
  br label %bb22, !dbg !33

bb22:                                             ; preds = %bb24, %bb21
  br i1 undef, label %bb24, label %bb23, !dbg !50

bb23:                                             ; preds = %bb22
  br label %bb24, !dbg !51

bb24:                                             ; preds = %bb23, %bb22
  br i1 undef, label %bb25, label %bb22, !dbg !33, !llvm.loop !52

bb25:                                             ; preds = %bb24, %bb20
  br i1 undef, label %bb27, label %bb26, !dbg !54

bb26:                                             ; preds = %bb25
  br label %bb27, !dbg !59

bb27:                                             ; preds = %bb26, %bb25
  br i1 undef, label %bb29, label %bb28, !dbg !60

bb28:                                             ; preds = %bb27
  br label %bb29, !dbg !65

bb29:                                             ; preds = %bb28, %bb27
  br i1 undef, label %bb31, label %bb30, !dbg !66

bb30:                                             ; preds = %bb29
  br label %bb31, !dbg !71

bb31:                                             ; preds = %bb30, %bb29
  br i1 undef, label %bb33, label %bb32, !dbg !72

bb32:                                             ; preds = %bb31
  br label %bb33, !dbg !75

bb33:                                             ; preds = %bb32, %bb31
  br label %bb34

bb34:                                             ; preds = %bb33, %bb3, %bb2
  ret void
}

attributes #0 = { "probe-stack"="__chkstk_darwin" }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: GNU)
!2 = !DIFile(filename: "foo", directory: "/")
!3 = distinct !DISubprogram(name: "foo", scope: !4, file: !4, line: 326, type: !5, scopeLine: 327, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!4 = !DIFile(filename: "foo", directory: "")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 328, column: 30, scope: !3)
!8 = !DILocation(line: 328, column: 49, scope: !3)
!9 = !DILocation(line: 329, column: 9, scope: !3)
!10 = !DILocation(line: 330, column: 7, scope: !3)
!11 = !DILocation(line: 340, column: 7, scope: !3)
!12 = !DILocation(line: 177, column: 7, scope: !13, inlinedAt: !15)
!13 = distinct !DISubprogram(name: "foo", scope: !14, file: !14, line: 174, type: !5, scopeLine: 175, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!14 = !DIFile(filename: "foo", directory: "")
!15 = distinct !DILocation(line: 342, column: 13, scope: !3)
!16 = !DILocation(line: 179, column: 9, scope: !13, inlinedAt: !15)
!17 = !DILocation(line: 182, column: 9, scope: !18, inlinedAt: !20)
!18 = distinct !DISubprogram(name: "print", scope: !19, file: !19, line: 180, type: !5, scopeLine: 180, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!19 = !DIFile(filename: "foo", directory: "")
!20 = distinct !DILocation(line: 346, column: 10, scope: !3)
!21 = !DILocation(line: 183, column: 7, scope: !18, inlinedAt: !20)
!22 = !DILocation(line: 34, column: 9, scope: !23, inlinedAt: !24)
!23 = distinct !DISubprogram(name: "grow", scope: !14, file: !14, line: 33, type: !5, scopeLine: 33, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!24 = distinct !DILocation(line: 91, column: 5, scope: !25, inlinedAt: !26)
!25 = distinct !DISubprogram(name: "operator+=", scope: !14, file: !14, line: 90, type: !5, scopeLine: 90, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!26 = distinct !DILocation(line: 347, column: 7, scope: !3)
!27 = !DILocation(line: 39, column: 11, scope: !23, inlinedAt: !24)
!28 = !DILocation(line: 40, column: 9, scope: !23, inlinedAt: !24)
!29 = !DILocation(line: 348, column: 9, scope: !3)
!30 = !DILocation(line: 349, column: 7, scope: !3)
!31 = !DILocation(line: 353, column: 7, scope: !3)
!32 = !DILocation(line: 354, column: 5, scope: !3)
!33 = !DILocation(line: 291, column: 5, scope: !34, inlinedAt: !35)
!34 = distinct !DISubprogram(name: "reset", scope: !4, file: !4, line: 290, type: !5, scopeLine: 290, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!35 = distinct !DILocation(line: 300, column: 29, scope: !36, inlinedAt: !37)
!36 = distinct !DISubprogram(name: "~foo", scope: !4, file: !4, line: 300, type: !5, scopeLine: 300, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!37 = distinct !DILocation(line: 300, column: 27, scope: !38, inlinedAt: !39)
!38 = distinct !DISubprogram(name: "~foo", scope: !4, file: !4, line: 300, type: !5, scopeLine: 300, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!39 = distinct !DILocation(line: 303, column: 7, scope: !40, inlinedAt: !41)
!40 = distinct !DISubprogram(name: "~foo", scope: !4, file: !4, line: 303, type: !5, scopeLine: 303, flags: DIFlagArtificial | DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!41 = distinct !DILocation(line: 303, column: 7, scope: !42, inlinedAt: !43)
!42 = distinct !DISubprogram(name: "~foo", scope: !4, file: !4, line: 303, type: !5, scopeLine: 303, flags: DIFlagArtificial | DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!43 = distinct !DILocation(line: 2156, column: 52, scope: !44, inlinedAt: !45)
!44 = distinct !DISubprogram(name: "~foo", scope: !19, file: !19, line: 2156, type: !5, scopeLine: 2156, flags: DIFlagArtificial | DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!45 = distinct !DILocation(line: 5227, column: 8, scope: !46, inlinedAt: !47)
!46 = distinct !DISubprogram(name: "~foo", scope: !19, file: !19, line: 5227, type: !5, scopeLine: 5227, flags: DIFlagArtificial | DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!47 = distinct !DILocation(line: 5227, column: 8, scope: !48, inlinedAt: !49)
!48 = distinct !DISubprogram(name: "~foo", scope: !19, file: !19, line: 5227, type: !5, scopeLine: 5227, flags: DIFlagArtificial | DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!49 = distinct !DILocation(line: 356, column: 1, scope: !3)
!50 = !DILocation(line: 294, column: 11, scope: !34, inlinedAt: !35)
!51 = !DILocation(line: 295, column: 9, scope: !34, inlinedAt: !35)
!52 = distinct !{!52, !33, !53}
!53 = !DILocation(line: 296, column: 5, scope: !34, inlinedAt: !35)
!54 = !DILocation(line: 2151, column: 9, scope: !55, inlinedAt: !56)
!55 = distinct !DISubprogram(name: "~foo", scope: !19, file: !19, line: 2150, type: !5, scopeLine: 2150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!56 = distinct !DILocation(line: 2150, column: 21, scope: !57, inlinedAt: !58)
!57 = distinct !DISubprogram(name: "~foo", scope: !19, file: !19, line: 2150, type: !5, scopeLine: 2150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!58 = distinct !DILocation(line: 2156, column: 52, scope: !44, inlinedAt: !45)
!59 = !DILocation(line: 2152, column: 7, scope: !55, inlinedAt: !56)
!60 = !DILocation(line: 2151, column: 9, scope: !61, inlinedAt: !62)
!61 = distinct !DISubprogram(name: "~foo", scope: !19, file: !19, line: 2150, type: !5, scopeLine: 2150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!62 = distinct !DILocation(line: 2150, column: 21, scope: !63, inlinedAt: !64)
!63 = distinct !DISubprogram(name: "~foo", scope: !19, file: !19, line: 2150, type: !5, scopeLine: 2150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!64 = distinct !DILocation(line: 2156, column: 52, scope: !44, inlinedAt: !45)
!65 = !DILocation(line: 2152, column: 7, scope: !61, inlinedAt: !62)
!66 = !DILocation(line: 2151, column: 9, scope: !67, inlinedAt: !68)
!67 = distinct !DISubprogram(name: "~foo", scope: !19, file: !19, line: 2150, type: !5, scopeLine: 2150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!68 = distinct !DILocation(line: 2150, column: 21, scope: !69, inlinedAt: !70)
!69 = distinct !DISubprogram(name: "~foo", scope: !19, file: !19, line: 2150, type: !5, scopeLine: 2150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!70 = distinct !DILocation(line: 2156, column: 52, scope: !44, inlinedAt: !45)
!71 = !DILocation(line: 2152, column: 7, scope: !67, inlinedAt: !68)
!72 = !DILocation(line: 2151, column: 9, scope: !67, inlinedAt: !73)
!73 = distinct !DILocation(line: 2150, column: 21, scope: !69, inlinedAt: !74)
!74 = distinct !DILocation(line: 2156, column: 52, scope: !44, inlinedAt: !45)
!75 = !DILocation(line: 2152, column: 7, scope: !67, inlinedAt: !73)
