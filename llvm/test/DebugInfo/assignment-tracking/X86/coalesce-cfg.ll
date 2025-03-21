; RUN: llc %s -o - -stop-after=finalize-isel \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Test coalescing of contiguous fragments in adjacent location definitions.
;; Further details and check directives inline.

target triple = "x86_64-unknown-linux-gnu"

@cond = dso_local global i8 0, align 1

;; The final store and linked dbg.assign indicate the whole variable is located
;; on the stack. Coalesce the two fragment defs that are generated (0-32,
;; 32-64) at the final dbg.assign into one (0-64, which covers the whole
;; variable meaning we don't need a fragment expression). And check the two
;; DBG_VALUEs in if.then are not coalesced, since they specify different
;; locations. This is the same as the first test in coalesce-simple.ll except
;; the dbg intrinsics are split up over a simple diamond CFG to check the info
;; is propagated betweeen blocks correctly.

; CHECK-LABEL: bb.0.entry:
; CHECK-NEXT: successors:
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_deref)
; CHECK-NEXT: TEST8mi
; CHECK-NEXT: JCC_1 %bb.2
; CHECK-NEXT: JMP_1 %bb.1

; CHECK-LABEL: bb.1.if.then:
; CHECK-NEXT: successors:
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT: MOV8mi $rip, 1, $noreg, @cond, $noreg, 0 :: (store (s8) into @cond)
; CHECK-NEXT: DBG_VALUE 1, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT: JMP_1 %bb.3

; CHECK-LABEL: bb.2.if.else:
; CHECK-NEXT: successors:
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:   MOV8mi $rip, 1, $noreg, @cond, $noreg, 1 :: (store (s8) into @cond)

; CHECK-LABEL: bb.3.if.end:
; CHECK-NEXT:   MOV32mi %stack.0.a, 1, $noreg, 0, $noreg, 5 :: (store (s32) into %ir.a, align 8)
; CHECK-NEXT:   DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:   RET 0

define dso_local void @_Z3funv() local_unnamed_addr !dbg !16 {
entry:
  %a = alloca i64, !DIAssignID !37
  call void @llvm.dbg.assign(metadata i64 poison, metadata !20, metadata !DIExpression(), metadata !37, metadata ptr %a, metadata !DIExpression()), !dbg !25
  %0 = load i8, ptr @cond, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.else

if.then:
  store i1 false, ptr @cond
  call void @llvm.dbg.value(metadata i32 1, metadata !20, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !25
  br label %if.end

if.else:
  store i1 true, ptr @cond
  br label %if.end

if.end:
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
!28 = distinct !DIAssignID()
!37 = distinct !DIAssignID()
!38 = distinct !DIAssignID()
