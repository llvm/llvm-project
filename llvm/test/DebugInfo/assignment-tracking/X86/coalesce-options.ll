;; Test coalescing of contiguous fragments in adjacent location definitions.
;; This test contains the first function from coalesce-simple.ll. Just use it
;; to check whether coalescing happens or not with different flag settings.
;;
;; +=================+==============================+======================+
;; | Coalescing flag | Instruction-Referencing flag | Coalescing behaviour |
;; +=================+==============================+======================+
;; | default         | enabled                      | enabled              |
;; | default         | disabled                     | disabled             |
;; | enabled         | *                            | enabled              |
;; | disabled        | *                            | disabled             |
;; +-----------------+------------------------------+----------------------+

;; Coalescing default + instructino-referencing enabled = enable.
; RUN: llc %s -o - -stop-after=finalize-isel -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,ENABLE

;; Coalescing default + instructino-referencing disabled = disable.
; RUN: llc %s -o - -stop-after=finalize-isel -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --check-prefixes=CHECK,DISABLE

;; Coalescing enabled + instructino-referencing disabled = enable.
; RUN: llc %s -o - -stop-after=finalize-isel -experimental-debug-variable-locations=false \
; RUN:     -debug-ata-coalesce-frags=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,ENABLE

;; Coalescing disabled + instructino-referencing enabled = disable.
; RUN: llc %s -o - -stop-after=finalize-isel -experimental-debug-variable-locations=true \
; RUN:     -debug-ata-coalesce-frags=false \
; RUN: | FileCheck %s --check-prefixes=CHECK,DISABLE

; CHECK: MOV32mi %stack.0.a, 1, $noreg, 0, $noreg, 5
; ENABLE-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_deref)
; DISABLE-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 32)

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z3funv() local_unnamed_addr !dbg !16 {
entry:
  %a = alloca i64, !DIAssignID !37
  call void @llvm.dbg.assign(metadata i64 poison, metadata !20, metadata !DIExpression(), metadata !37, metadata ptr %a, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !20, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !25
  store i32 5, ptr %a, !DIAssignID !38
  call void @llvm.dbg.assign(metadata i32 5, metadata !20, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !38, metadata ptr %a, metadata !DIExpression()), !dbg !25
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "G", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "F", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 8, !"PIC Level", i32 2}
!12 = !{i32 7, !"PIE Level", i32 2}
!13 = !{i32 7, !"uwtable", i32 2}
!14 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!15 = !{!"clang version 17.0.0"}
!16 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !3, file: !3, line: 3, type: !17, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !{!20}
!20 = !DILocalVariable(name: "X", scope: !16, file: !3, line: 4, type: !21)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Pair", file: !3, line: 2, size: 64, flags: DIFlagTypePassByValue, elements: !22, identifier: "_ZTS4Pair")
!22 = !{}
!25 = !DILocation(line: 0, scope: !16)
!26 = !DILocation(line: 7, column: 7, scope: !27)
!27 = distinct !DILexicalBlock(scope: !16, file: !3, line: 7, column: 7)
!37 = distinct !DIAssignID()
!38 = distinct !DIAssignID()
