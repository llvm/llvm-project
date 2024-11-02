; RUN: llc %s -start-after=codegenprepare -stop-before=finalize-isel -experimental-debug-variable-locations=false -o - | FileCheck --check-prefixes=CHECK,DAG %s
; RUN: llc %s -fast-isel=true -start-after=codegenprepare -stop-before=finalize-isel -experimental-debug-variable-locations=false -o - | FileCheck --check-prefixes=CHECK,FAST %s
; RUN: llc %s -global-isel=true -start-after=codegenprepare -stop-before=finalize-isel -experimental-debug-variable-locations=false -o - | FileCheck --check-prefixes=CHECK,GLOBAL %s

;; Run with experimental debuginfo iterators.
; RUN: llc --try-experimental-debuginfo-iterators %s -start-after=codegenprepare -stop-before=finalize-isel -experimental-debug-variable-locations=false -o - | FileCheck --check-prefixes=CHECK,DAG %s
; RUN: llc --try-experimental-debuginfo-iterators %s -fast-isel=true -start-after=codegenprepare -stop-before=finalize-isel -experimental-debug-variable-locations=false -o - | FileCheck --check-prefixes=CHECK,FAST %s
;; FIXME: Disabled until global isel support is added.
; run: llc --try-experimental-debuginfo-iterators %s -global-isel=true -start-after=codegenprepare -stop-before=finalize-isel -experimental-debug-variable-locations=false -o - | FileCheck --check-prefixes=CHECK,GLOBAL %s

;; Run with assignment tracking enabled (use sed to add the module flag).
; RUN: sed 's/;Uncomment-with-sed//g' < %s \
; RUN: | llc -global-isel=false -start-after=codegenprepare -stop-before=finalize-isel -experimental-debug-variable-locations=false -o - \
; RUN: | FileCheck --check-prefixes=CHECK,DAG %s

;; Test that a dbg.value that uses a DIArgList is correctly converted to a
;; DBG_VALUE_LIST that uses the registers corresponding to its operands.

; CHECK-DAG: [[A_VAR:![0-9]+]] = !DILocalVariable(name: "a"
; CHECK-DAG: [[B_VAR:![0-9]+]] = !DILocalVariable(name: "b"
; CHECK-DAG: [[C_VAR:![0-9]+]] = !DILocalVariable(name: "c"
; CHECK-DAG: [[D_VAR:![0-9]+]] = !DILocalVariable(name: "d"
; CHECK-LABEL: bb.{{(0|1)}}.entry
; DAG-DAG: DBG_VALUE_LIST [[A_VAR]], !DIExpression(DW_OP_LLVM_arg, 0), %0, debug-location
; DAG-DAG: DBG_VALUE_LIST [[B_VAR]], !DIExpression(DW_OP_LLVM_arg, 0), %1, debug-location
; DAG: DBG_VALUE_LIST [[C_VAR]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus), %0, %1, debug-location
; DAG: DBG_VALUE_LIST [[D_VAR]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_eq, DW_OP_LLVM_arg, 0, DW_OP_ne, DW_OP_LLVM_arg, 1, DW_OP_gt, DW_OP_LLVM_arg, 0, DW_OP_lt, DW_OP_LLVM_arg, 1, DW_OP_le), %0, %1, debug-location
; FAST-DAG: DBG_VALUE $noreg, $noreg, [[A_VAR]], !DIExpression(DW_OP_LLVM_arg, 0), debug-location
; FAST-DAG: DBG_VALUE $noreg, $noreg, [[B_VAR]], !DIExpression(DW_OP_LLVM_arg, 0), debug-location
; FAST: DBG_VALUE $noreg, $noreg, [[C_VAR]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus), debug-location
; FAST: DBG_VALUE $noreg, $noreg, [[D_VAR]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_eq, DW_OP_LLVM_arg, 0, DW_OP_ne, DW_OP_LLVM_arg, 1, DW_OP_gt, DW_OP_LLVM_arg, 0, DW_OP_lt, DW_OP_LLVM_arg, 1, DW_OP_le), debug-location
; GLOBAL-DAG: DBG_VALUE $noreg, 0, [[A_VAR]], !DIExpression(DW_OP_LLVM_arg, 0), debug-location
; GLOBAL-DAG: DBG_VALUE $noreg, 0, [[B_VAR]], !DIExpression(DW_OP_LLVM_arg, 0), debug-location
; GLOBAL: DBG_VALUE $noreg, 0, [[C_VAR]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus), debug-location
; GLOBAL: DBG_VALUE $noreg, 0, [[D_VAR]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_eq, DW_OP_LLVM_arg, 0, DW_OP_ne, DW_OP_LLVM_arg, 1, DW_OP_gt, DW_OP_LLVM_arg, 0, DW_OP_lt, DW_OP_LLVM_arg, 1, DW_OP_le), debug-location

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.16.27034"

define dso_local i32 @"?foo@@YAHHH@Z"(i32 %a, i32 %b) local_unnamed_addr !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata !DIArgList(i32 %b), metadata !14, metadata !DIExpression(DW_OP_LLVM_arg, 0)), !dbg !17
  call void @llvm.dbg.value(metadata !DIArgList(i32 %a), metadata !15, metadata !DIExpression(DW_OP_LLVM_arg, 0)), !dbg !17
  call void @llvm.dbg.value(metadata !DIArgList(i32 %a, i32 %b), metadata !16, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus)), !dbg !17
  call void @llvm.dbg.value(metadata !DIArgList(i32 %a, i32 %b), metadata !20, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_eq, DW_OP_LLVM_arg, 0, DW_OP_ne, DW_OP_LLVM_arg, 1, DW_OP_gt, DW_OP_LLVM_arg, 0, DW_OP_lt, DW_OP_LLVM_arg, 1, DW_OP_le)), !dbg !17
  %mul = mul nsw i32 %b, %a, !dbg !18
  ret i32 %mul, !dbg !18
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5,
;Uncomment-with-sed !19,
!6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "debug_value_list_selectiondag.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 11.0.0"}
!8 = distinct !DISubprogram(name: "foo", linkageName: "?foo@@YAHHH@Z", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!9 = !DIFile(filename: ".\\debug_value_list.cpp", directory: "/tmp")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !15, !16}
!14 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !9, line: 1, type: !12)
!15 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !9, line: 1, type: !12)
!16 = !DILocalVariable(name: "c", scope: !8, file: !9, line: 2, type: !12)
!17 = !DILocation(line: 0, scope: !8)
!18 = !DILocation(line: 3, scope: !8)
!19 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!20 = !DILocalVariable(name: "d", scope: !8, file: !9, line: 2, type: !12)
