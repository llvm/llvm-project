; RUN: llc %s -stop-after=codegenprepare -o - | FileCheck %s
; RUN: llc %s -stop-after=codegenprepare -o - --try-experimental-debuginfo-iterators | FileCheck %s
; REQUIRES: x86-registered-target
;
; Test that when we skip over multiple selects in CGP, that the debug-info
; attached to those selects is still fixed up.

; CHECK: #dbg_value(ptr %sunkaddr,

source_filename = "reduced.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.(anonymous namespace)::CFIInstrInserter" = type { ptr, ptr, ptr, ptr }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

define i1 @_ZN12_GLOBAL__N_116CFIInstrInserter20runOnMachineFunctionERN4llvm15MachineFunctionE(ptr %this, i1 %or.cond.i) !dbg !5 {
entry:
  %CSRLocMap.i = getelementptr %"class.(anonymous namespace)::CFIInstrInserter", ptr %this, i64 0, i32 2, !dbg !16
  %bf.load.i.i.i.i = load i32, ptr %CSRLocMap.i, align 8, !dbg !17
  br i1 %or.cond.i, label %_ZN4llvm12DenseMapBaseINS_13SmallDenseMapIjN12_GLOBAL__N_116CFIInstrInserter16CSRSavedLocationELj16ENS_12DenseMapInfoIjvEENS_6detail12DenseMapPairIjS4_EEEEjS4_S6_S9_E5clearEv.exit.i, label %if.end.i.i, !dbg !18

if.end.i.i:                                       ; preds = %entry
  store ptr null, ptr null, align 8, !dbg !19
  %bf.load.i.i.i.pre.i.i.i.i.i = load i32, ptr %CSRLocMap.i, align 8, !dbg !20
  %cond.i.i.i.i.i.i.i.i.i = select i1 false, ptr null, ptr null, !dbg !21
  tail call void @llvm.dbg.value(metadata ptr %CSRLocMap.i, metadata !14, metadata !DIExpression()), !dbg !21
  %cond.i.i.i7.i.i.i.i.i.i = select i1 false, i32 0, i32 0, !dbg !22
  br label %_ZN4llvm12DenseMapBaseINS_13SmallDenseMapIjN12_GLOBAL__N_116CFIInstrInserter16CSRSavedLocationELj16ENS_12DenseMapInfoIjvEENS_6detail12DenseMapPairIjS4_EEEEjS4_S6_S9_E5clearEv.exit.i, !dbg !23

_ZN4llvm12DenseMapBaseINS_13SmallDenseMapIjN12_GLOBAL__N_116CFIInstrInserter16CSRSavedLocationELj16ENS_12DenseMapInfoIjvEENS_6detail12DenseMapPairIjS4_EEEEjS4_S6_S9_E5clearEv.exit.i: ; preds = %if.end.i.i, %entry
  ret i1 false, !dbg !24
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!llvm.debugify = !{!3, !4}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "reduced.ll", directory: "/")
!3 = !{i32 9}
!4 = !{i32 5}
!5 = distinct !DISubprogram(name: "_ZN12_GLOBAL__N_116CFIInstrInserter20runOnMachineFunctionERN4llvm15MachineFunctionE", linkageName: "_ZN12_GLOBAL__N_116CFIInstrInserter20runOnMachineFunctionERN4llvm15MachineFunctionE", scope: null, file: !2, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !13, !14, !15}
!9 = !DILocalVariable(name: "1", scope: !5, file: !2, line: 1, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !2, line: 2, type: !12)
!12 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "3", scope: !5, file: !2, line: 5, type: !12)
!14 = !DILocalVariable(name: "4", scope: !5, file: !2, line: 6, type: !10)
!15 = !DILocalVariable(name: "5", scope: !5, file: !2, line: 7, type: !12)
!16 = !DILocation(line: 1, column: 1, scope: !5)
!17 = !DILocation(line: 2, column: 1, scope: !5)
!18 = !DILocation(line: 3, column: 1, scope: !5)
!19 = !DILocation(line: 4, column: 1, scope: !5)
!20 = !DILocation(line: 5, column: 1, scope: !5)
!21 = !DILocation(line: 6, column: 1, scope: !5)
!22 = !DILocation(line: 7, column: 1, scope: !5)
!23 = !DILocation(line: 8, column: 1, scope: !5)
!24 = !DILocation(line: 9, column: 1, scope: !5)
