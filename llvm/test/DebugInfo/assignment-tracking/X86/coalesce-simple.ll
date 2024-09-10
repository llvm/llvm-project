; RUN: llc %s -o - -stop-after=finalize-isel \
; RUN: | FileCheck %s --implicit-check-not=DBG_
; RUN: llc --try-experimental-debuginfo-iterators %s -o - -stop-after=finalize-isel \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Test coalescing of contiguous fragments in adjacent location definitions.
;; Further details and check directives inline.

target triple = "x86_64-unknown-linux-gnu"

;; The final store and linked dbg.assign indicate the whole variable is located
;; on the stack. Coalesce the two fragment defs that are generated (0-32,
;; 32-64) at the final dbg.assign into one (0-64, which covers the whole
;; variable meaning we don't need a fragment expression). And check the
;; first two DBG_VALUEs are not coalesced, since they specify different
;; locations.
; CHECK: _Z3fun
; CHECK-LABEL: bb.0.entry:
; CHECK-NEXT: DBG_VALUE 1, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT: MOV32mi %stack.0.a, 1, $noreg, 0, $noreg, 5
; CHECK-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_deref)
; CHECK-NEXT: RET
define dso_local void @_Z3funv() local_unnamed_addr !dbg !16 {
entry:
  %a = alloca i64, !DIAssignID !37
  call void @llvm.dbg.assign(metadata i64 poison, metadata !20, metadata !DIExpression(), metadata !37, metadata ptr %a, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !20, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !25
  store i32 5, ptr %a, !DIAssignID !38
  call void @llvm.dbg.assign(metadata i32 5, metadata !20, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !38, metadata ptr %a, metadata !DIExpression()), !dbg !25
  ret void
}

;; Similar to the test above except that the variable has been split over two
;; allocas, so coalescing should not take place (different memory location for
;; the fragments).
; CHECK: Z3funv2
; CHECK-LABEL: bb.0.entry:
; CHECK-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE %stack.1.b, $noreg, ![[#]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT: DBG_VALUE 1, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: MOV32mi %stack.0.a, 1, $noreg, 0, $noreg, 5
;; Both fragments 0-32 and 32-64 are in memory now, but located in different stack slots.
; CHECK-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: RET
define dso_local void @_Z3funv2() local_unnamed_addr !dbg !39 {
entry:
  %a = alloca i32, !DIAssignID !42
  call void @llvm.dbg.assign(metadata i64 poison, metadata !41, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !42, metadata ptr %a, metadata !DIExpression()), !dbg !44
  %b = alloca i32, !DIAssignID !45
  call void @llvm.dbg.assign(metadata i64 poison, metadata !41, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !45, metadata ptr %b, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !44
  store i32 5, ptr %a, !DIAssignID !43
  call void @llvm.dbg.assign(metadata i32 5, metadata !41, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !43, metadata ptr %a, metadata !DIExpression()), !dbg !44
  ret void
}

;; Similar to the first test above except the part that it's the slice 16-32
;; that is "partially promoted". The dbg defs after the alloca cannot be
;; coalesced (slices 0-16 and 32-64 are in memory but 16-32 isn't). The entire
;; variable is on the stack after the store.
; CHECK: _Z2funv3
; CHECK-LABEL: bb.0.entry:
; CHECK-NEXT: DBG_VALUE 2, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 16, 16)
; CHECK-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 16)
; CHECK-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT: MOV32mi %stack.0.a, 1, $noreg, 0, $noreg, 5
; CHECK-NEXT: DBG_VALUE %stack.0.a, $noreg, ![[#]], !DIExpression(DW_OP_deref)
; CHECK-NEXT: RET
define dso_local void @_Z2funv3() local_unnamed_addr !dbg !46 {
entry:
  %a = alloca i64, !DIAssignID !49
  call void @llvm.dbg.assign(metadata i64 poison, metadata !48, metadata !DIExpression(), metadata !49, metadata ptr %a, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 2, metadata !48, metadata !DIExpression(DW_OP_LLVM_fragment, 16, 16)), !dbg !51
  store i32 5, ptr %a, !DIAssignID !50
  call void @llvm.dbg.assign(metadata i32 5, metadata !48, metadata !DIExpression(DW_OP_LLVM_fragment, 16, 16), metadata !50, metadata ptr %a, metadata !DIExpression(DW_OP_plus_uconst, 2)), !dbg !51
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
!39 = distinct !DISubprogram(name: "fun2", linkageName: "_Z3funv2", scope: !3, file: !3, line: 3, type: !17, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !40)
!40 = !{!41}
!41 = !DILocalVariable(name: "X", scope: !39, file: !3, line: 10, type: !21)
!42 = distinct !DIAssignID()
!43 = distinct !DIAssignID()
!44 = !DILocation(line: 0, scope: !39)
!45 = distinct !DIAssignID()
!46 = distinct !DISubprogram(name: "fun3", linkageName: "_Z3funv3", scope: !3, file: !3, line: 3, type: !17, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !47)
!47 = !{!48}
!48 = !DILocalVariable(name: "X", scope: !46, file: !3, line: 10, type: !21)
!49 = distinct !DIAssignID()
!50 = distinct !DIAssignID()
!51 = !DILocation(line: 0, scope: !46)
