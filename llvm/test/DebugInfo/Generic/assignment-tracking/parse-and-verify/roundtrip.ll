; RUN: opt %s -verify -experimental-assignment-tracking   \
; RUN: | opt -verify -S -experimental-assignment-tracking \
; RUN: | FileCheck %s

;; Roundtrip test (text -> bitcode -> text) for DIAssignID metadata and
;; llvm.dbg.assign intrinsics.

;; DIAssignID attachment only.
; CHECK-LABEL: @fun()
; CHECK: %local = alloca i32, align 4, !DIAssignID ![[ID1:[0-9]+]]
define dso_local void @fun() !dbg !7 {
entry:
  %local = alloca i32, align 4, !DIAssignID !14
  ret void, !dbg !13
}

;; Unlinked llvm.dbg.assign.
; CHECK-DAG: @fun2()
; CHECK: llvm.dbg.assign(metadata i32 undef, metadata ![[VAR2:[0-9]+]], metadata !DIExpression(), metadata ![[ID2:[0-9]+]], metadata i32 undef, metadata !DIExpression()), !dbg ![[DBG2:[0-9]+]]
define dso_local void @fun2() !dbg !15 {
entry:
  %local = alloca i32, align 4
  call void @llvm.dbg.assign(metadata i32 undef, metadata !16, metadata !DIExpression(), metadata !18, metadata i32 undef, metadata !DIExpression()), !dbg !17
  ret void, !dbg !17
}

;; An llvm.dbg.assign linked to an alloca.
; CHECK-LABEL: @fun3()
; CHECK: %local = alloca i32, align 4, !DIAssignID ![[ID3:[0-9]+]]
; CHECK-NEXT: llvm.dbg.assign(metadata i32 undef, metadata ![[VAR3:[0-9]+]], metadata !DIExpression(), metadata ![[ID3]], metadata i32 undef, metadata !DIExpression()), !dbg ![[DBG3:[0-9]+]]
define dso_local void @fun3() !dbg !19 {
entry:
  %local = alloca i32, align 4, !DIAssignID !22
  call void @llvm.dbg.assign(metadata i32 undef, metadata !20, metadata !DIExpression(), metadata !22, metadata i32 undef, metadata !DIExpression()), !dbg !21
  ret void, !dbg !21
}

;; Check that using a DIAssignID as an operand before using it as an attachment
;; works (the order of the alloca and dbg.assign has been swapped).
; CHECK-LABEL: @fun4()
; CHECK: llvm.dbg.assign(metadata i32 undef, metadata ![[VAR4:[0-9]+]], metadata !DIExpression(), metadata ![[ID4:[0-9]+]], metadata i32 undef, metadata !DIExpression()), !dbg ![[DBG4:[0-9]+]]
; CHECK-NEXT: %local = alloca i32, align 4, !DIAssignID ![[ID4]]
define dso_local void @fun4() !dbg !23 {
entry:
  call void @llvm.dbg.assign(metadata i32 undef, metadata !24, metadata !DIExpression(), metadata !26, metadata i32 undef, metadata !DIExpression()), !dbg !25
  %local = alloca i32, align 4, !DIAssignID !26
  ret void, !dbg !25
}

;; Check that the value and address operands print correctly.
;; There are currently no plans to support DIArgLists for the address component.
; CHECK-LABEL: @fun5
; CHECK: %local = alloca i32, align 4, !DIAssignID ![[ID5:[0-9]+]]
; CHECK-NEXT: llvm.dbg.assign(metadata i32 %v, metadata ![[VAR5:[0-9]+]], metadata !DIExpression(), metadata ![[ID5]], metadata i32* %local, metadata !DIExpression()), !dbg ![[DBG5:[0-9]+]]
; CHECK-NEXT: llvm.dbg.assign(metadata !DIArgList(i32 %v, i32 1), metadata ![[VAR5]], metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_minus, DW_OP_stack_value), metadata ![[ID5]], metadata i32* %local, metadata !DIExpression()), !dbg ![[DBG5]]
define dso_local void @fun5(i32 %v) !dbg !27 {
entry:
  %local = alloca i32, align 4, !DIAssignID !30
  call void @llvm.dbg.assign(metadata i32 %v, metadata !28, metadata !DIExpression(), metadata !30, metadata i32* %local, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.assign(metadata !DIArgList(i32 %v, i32 1), metadata !28, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_minus, DW_OP_stack_value), metadata !30, metadata i32* %local, metadata !DIExpression()), !dbg !29
  ret void
}

; CHECK-DAG: ![[ID1]] = distinct !DIAssignID()
; CHECK-DAG: ![[ID2]] = distinct !DIAssignID()
; CHECK-DAG: ![[VAR2]] = !DILocalVariable(name: "local2",
; CHECK-DAG: ![[DBG2]] = !DILocation(line: 2
; CHECK-DAG: ![[ID3]] = distinct !DIAssignID()
; CHECK-DAG: ![[VAR3]] = !DILocalVariable(name: "local3",
; CHECK-DAG: ![[DBG3]] = !DILocation(line: 3,
; CHECK-DAG: ![[ID4]] = distinct !DIAssignID()
; CHECK-DAG: ![[VAR4]] = !DILocalVariable(name: "local4",
; CHECK-DAG: ![[DBG4]] = !DILocation(line: 4,
; CHECK-DAG: ![[ID5]] = distinct !DIAssignID()
; CHECK-DAG: ![[VAR5]] = !DILocalVariable(name: "local5",
; CHECK-DAG: ![[DBG5]] = !DILocation(line: 5,

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 2, type: !11)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 1, column: 1, scope: !7)
!14 = distinct !DIAssignID()
!15 = distinct !DISubprogram(name: "fun2", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DILocalVariable(name: "local2", scope: !15, file: !1, line: 2, type: !11)
!17 = !DILocation(line: 2, column: 1, scope: !15)
!18 = distinct !DIAssignID()
!19 = distinct !DISubprogram(name: "fun3", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!20 = !DILocalVariable(name: "local3", scope: !19, file: !1, line: 2, type: !11)
!21 = !DILocation(line: 3, column: 1, scope: !19)
!22 = distinct !DIAssignID()
!23 = distinct !DISubprogram(name: "fun4", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!24 = !DILocalVariable(name: "local4", scope: !23, file: !1, line: 2, type: !11)
!25 = !DILocation(line: 4, column: 1, scope: !23)
!26 = distinct !DIAssignID()
!27 = distinct !DISubprogram(name: "fun5", scope: !1, file: !1, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!28 = !DILocalVariable(name: "local5", scope: !27, file: !1, line: 2, type: !11)
!29 = !DILocation(line: 5, column: 1, scope: !27)
!30 = distinct !DIAssignID()
!31 = !DISubroutineType(types: !32)
!32 = !{null, !11}
