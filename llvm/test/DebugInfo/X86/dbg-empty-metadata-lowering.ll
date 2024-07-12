; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG --check-prefixes=AT-DISABLED,BOTH
; RUN: llc %s -stop-after=finalize-isel -o - --try-experimental-debuginfo-iterators \
; RUN: | FileCheck %s --implicit-check-not=DBG --check-prefixes=AT-DISABLED,BOTH
; RUN: llc --try-experimental-debuginfo-iterators %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG --check-prefixes=AT-DISABLED,BOTH

;; Check that dbg.values with empty metadata are treated as kills (i.e. become
;; DBG_VALUE $noreg, ...). dbg.declares with empty metadata location operands
;; should be ignored.

; RUN: sed 's/;Uncomment-with-sed//g' < %s \
; RUN: | llc -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG --check-prefixes=AT-ENABLED,BOTH
; RUN: sed 's/;Uncomment-with-sed//g' < %s \
; RUN: | llc --try-experimental-debuginfo-iterators -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG --check-prefixes=AT-ENABLED,BOTH
;; Check the same behaviour occurs with assignment tracking enabled.

;; First ensure assignment tracking is truly unset/set.
; AT-DISABLED: !llvm.module.flags = !{![[DWARFv:[0-9]+]], ![[DEBUGv:[0-9]+]]}
; AT-DISABLED-NOT: !{i32 7, !"debug-info-assignment-tracking", i1 true}
; AT-ENABLED: !llvm.module.flags = !{![[DWARFv:[0-9]+]], ![[DEBUGv:[0-9]+]], ![[AT:[0-9]+]]}
; AT-ENABLED-DAG: ![[AT]] = !{i32 7, !"debug-info-assignment-tracking", i1 true}
; BOTH-DAG: ![[DWARFv]] = !{i32 7, !"Dwarf Version", i32 5}
; BOTH-DAG: ![[DEBUGv]] = !{i32 2, !"Debug Info Version", i32 3}

; BOTH: stack: []
; BOTH: DBG_VALUE float 5.000000e+00
; BOTH: @ext
; BOTH: DBG_VALUE $noreg
; BOTH: @ext

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @fun() local_unnamed_addr #0 !dbg !9 {
entry:
  call void @llvm.dbg.declare(metadata !{}, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata float 5.000000e+00, metadata !13, metadata !DIExpression()), !dbg !15
  tail call void @ext(), !dbg !16
  call void @llvm.dbg.value(metadata !{}, metadata !13, metadata !DIExpression()), !dbg !15
  tail call void @ext(), !dbg !16
  ret void, !dbg !17
}

declare !dbg !18 void @ext() local_unnamed_addr
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3
;Uncomment-with-sed , !21
}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 16.0.0"}
!9 = distinct !DISubprogram(name: "fun", linkageName: "fun", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{!13}
!13 = !DILocalVariable(name: "f", scope: !9, file: !1, line: 3, type: !14)
!14 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!15 = !DILocation(line: 0, scope: !9)
!16 = !DILocation(line: 4, column: 3, scope: !9)
!17 = !DILocation(line: 5, column: 1, scope: !9)
!18 = !DISubprogram(name: "ext", linkageName: "ext", scope: !1, file: !1, line: 1, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !19)
!19 = !{}
!20 = !DILocalVariable(name: "g", scope: !9, file: !1, line: 3, type: !14)
!21 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
