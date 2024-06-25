; RUN: opt -passes=adce -S < %s | FileCheck %s
; RUN: opt -passes=adce -S < %s --try-experimental-debuginfo-iterators | FileCheck %s
; Test that debug info intrinsics in dead scopes get eliminated by -adce.

; Generated with 'clang -g -S -emit-llvm | opt -passes=mem2reg -inline' at r262899
; (before -passes=adce  was augmented) and then hand-reduced.  This was the input:
;
;;void sink(void);
;;
;;void variable_in_unused_subscope(void) {
;;  { int i = 0; }
;;  sink();
;;}
;;
;;void variable_in_parent_scope(void) {
;;  int i = 0;
;;  { sink(); }
;;}
;;
;;static int empty_function_with_unused_variable(void) {
;;  { int i = 0; }
;;  return 0;
;;}
;;
;;void calls_empty_function_with_unused_variable_in_unused_subscope(void) {
;;  { empty_function_with_unused_variable(); }
;;  sink();
;;}

declare void @llvm.dbg.value(metadata, metadata, metadata)

declare void @sink()

; CHECK-LABEL: define void @variable_in_unused_subscope(
define void @variable_in_unused_subscope() !dbg !4 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @sink
; CHECK-NEXT:   ret void
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !17), !dbg !18
  call void @sink(), !dbg !19
  ret void, !dbg !20
}

; CHECK-LABEL: define void @variable_in_parent_scope(
define void @variable_in_parent_scope() !dbg !7 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   #dbg_value
; CHECK-NEXT:   call void @sink
; CHECK-NEXT:   ret void
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !21, metadata !17), !dbg !22
  call void @sink(), !dbg !23
  ret void, !dbg !25
}

; CHECK-LABEL: define void @calls_empty_function_with_unused_variable_in_unused_subscope(
define void @calls_empty_function_with_unused_variable_in_unused_subscope() !dbg !8 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @sink
; CHECK-NEXT:   ret void
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !17), !dbg !28
  call void @sink(), !dbg !31
  ret void, !dbg !32
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "/path/to/test/Transforms/ADCE")
!2 = !{}
!4 = distinct !DISubprogram(name: "variable_in_unused_subscope", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DISubprogram(name: "variable_in_parent_scope", scope: !1, file: !1, line: 8, type: !5, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = distinct !DISubprogram(name: "calls_empty_function_with_unused_variable_in_unused_subscope", scope: !1, file: !1, line: 18, type: !5, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!10 = distinct !DISubprogram(name: "empty_function_with_unused_variable", scope: !1, file: !1, line: 13, type: !11, isLocal: true, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !DILocalVariable(name: "i", scope: !16, file: !1, line: 4, type: !13)
!16 = distinct !DILexicalBlock(scope: !4, file: !1, line: 4, column: 3)
!17 = !DIExpression()
!18 = !DILocation(line: 4, column: 9, scope: !16)
!19 = !DILocation(line: 5, column: 3, scope: !4)
!20 = !DILocation(line: 6, column: 1, scope: !4)
!21 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 9, type: !13)
!22 = !DILocation(line: 9, column: 7, scope: !7)
!23 = !DILocation(line: 10, column: 5, scope: !24)
!24 = distinct !DILexicalBlock(scope: !7, file: !1, line: 10, column: 3)
!25 = !DILocation(line: 11, column: 1, scope: !7)
!26 = !DILocalVariable(name: "i", scope: !27, file: !1, line: 14, type: !13)
!27 = distinct !DILexicalBlock(scope: !10, file: !1, line: 14, column: 3)
!28 = !DILocation(line: 14, column: 9, scope: !27, inlinedAt: !29)
!29 = distinct !DILocation(line: 19, column: 5, scope: !30)
!30 = distinct !DILexicalBlock(scope: !8, file: !1, line: 19, column: 3)
!31 = !DILocation(line: 20, column: 3, scope: !8)
!32 = !DILocation(line: 21, column: 1, scope: !8)
