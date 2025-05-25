; RUN: opt -S -passes=separate-const-offset-from-gep < %s | FileCheck %s

; Check that SeparateConstOffsetFromGEP's reuniteExts() propagates the debug location
; to the new sext instruction, which replaces the old add/sub instruction (`%add.sext`/`%sub.sext`).

define i64 @add_sext__dominating_add_nsw(i32 %arg0, i32 %arg1) !dbg !5 {
; CHECK-LABEL: define i64 @add_sext__dominating_add_nsw(
; CHECK:  entry:
; CHECK:    [[ADD_SEXT:%.*]] = sext i32 [[ADD_NSW:%.*]] to i64, !dbg [[DBG9:![0-9]+]]
;
entry:
  %add.nsw = add nsw i32 %arg0, %arg1, !dbg !8
  %arg0.sext = sext i32 %arg0 to i64, !dbg !9
  %arg1.sext = sext i32 %arg1 to i64, !dbg !10
  %add.sext = add i64 %arg0.sext, %arg1.sext, !dbg !11
  call void @use.i32(i32 %add.nsw), !dbg !12
  ret i64 %add.sext, !dbg !13
}

define i64 @sub_sext__dominating_sub_nsw(i32 %arg0, i32 %arg1) !dbg !14 {
; CHECK-LABEL: define i64 @sub_sext__dominating_sub_nsw(
; CHECK:  entry:
; CHECK:    [[SUB_SEXT:%.*]] = sext i32 [[SUB_NSW:%.*]] to i64, !dbg [[DBG14:![0-9]+]]

;
entry:
  %sub.nsw = sub nsw i32 %arg0, %arg1, !dbg !15
  %arg0.sext = sext i32 %arg0 to i64, !dbg !16
  %arg1.sext = sext i32 %arg1 to i64, !dbg !17
  %sub.sext = sub i64 %arg0.sext, %arg1.sext, !dbg !18
  call void @use.i32(i32 %sub.nsw), !dbg !19
  ret i64 %sub.sext, !dbg !20
}

declare void @use.i32(i32 noundef)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

; CHECK: [[DBG9]] = !DILocation(line: 4,
; CHECK: [[DBG14]] = !DILocation(line: 10,

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{i32 12}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "add_sext__dominating_add_nsw", linkageName: "add_sext__dominating_add_nsw", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = distinct !DISubprogram(name: "sub_sext__dominating_sub_nsw", linkageName: "sub_sext__dominating_sub_nsw", scope: null, file: !1, line: 7, type: !6, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocation(line: 7, column: 1, scope: !14)
!16 = !DILocation(line: 8, column: 1, scope: !14)
!17 = !DILocation(line: 9, column: 1, scope: !14)
!18 = !DILocation(line: 10, column: 1, scope: !14)
!19 = !DILocation(line: 11, column: 1, scope: !14)
!20 = !DILocation(line: 12, column: 1, scope: !14)
