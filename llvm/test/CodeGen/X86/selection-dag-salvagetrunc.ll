; RUN: llc --stop-after=finalize-isel < %s | FileCheck %s
;
; Verify that we can correctly salvage truncate expressions during SelectionDAG.
; Fixes LLVM issue #63076.

; CHECK: body
; CHECK: DBG_INSTR_REF !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_LLVM_convert, 1, DW_ATE_unsigned)

source_filename = "repro.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local noundef i32 @_Z3funv() !dbg !10 {
entry:
  %call = tail call noundef zeroext i1 @_Z3getv(), !dbg !17
  call void @llvm.dbg.value(metadata i1 %call, metadata !15, metadata !DIExpression()), !dbg !18
  %conv = zext i1 %call to i32, !dbg !19
  ret i32 %conv, !dbg !20
}

declare !dbg !21 noundef zeroext i1 @_Z3getv()

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "repro.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 18.0.0"}
!10 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 2, type: !11, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DILocalVariable(name: "b", scope: !10, file: !1, line: 4, type: !16)
!16 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!17 = !DILocation(line: 4, column: 11, scope: !10)
!18 = !DILocation(line: 0, scope: !10)
!19 = !DILocation(line: 5, column: 9, scope: !10)
!20 = !DILocation(line: 5, column: 2, scope: !10)
!21 = !DISubprogram(name: "get", linkageName: "_Z3getv", scope: !1, file: !1, line: 1, type: !22, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!22 = !DISubroutineType(types: !23)
!23 = !{!16}
