; RUN: llc %s -stop-before expand-isel-pseudos -o - | FileCheck %s

;--------------------------------------------------------------------
; This test case is basically generated from the following C code.
; Compiled with "--target=x86_64-apple-darwin -S -g -O3" to get debug
; info for optimized code.
;
; struct SS {
;   int a;
;   int b;
; } S = { .a = 23, .b = -17 };
;
; int test1() {
;   struct SS* foo1 = &S;
;   return (int)foo1;
; }
;
; int test2() {
;   struct SS* foo2 = &S;
;   struct SS* bar2 = &S;
;   return (int)foo2 + (int)bar2;
; }
;
; int test3() {
;   struct SS* bar3 = &S;
;   struct SS* foo3 = &S;
;   return (int)foo3 + (int)bar3;
; }
;
; int test4() {
;   struct SS* foo4 = &S;
;   struct SS* bar4 = &S;
;   foo = 0;
;   return (int)foo4 + (int)bar4;
; }
;
; int test5() {
;   struct SS* bar5 = &S;
;   struct SS* foo5 = &S;
;   foo5 = 0;
;   return (int)foo5 + (int)bar5;
; }
;--------------------------------------------------------------------

; CHECK:  ![[FOO1:.*]] = !DILocalVariable(name: "foo1"
; CHECK:  ![[BAR1:.*]] = !DILocalVariable(name: "bar1"
; CHECK:  ![[FOO2:.*]] = !DILocalVariable(name: "foo2"
; CHECK:  ![[BAR2:.*]] = !DILocalVariable(name: "bar2"
; CHECK:  ![[FOO3:.*]] = !DILocalVariable(name: "bar3"
; CHECK:  ![[BAR3:.*]] = !DILocalVariable(name: "foo3"
; CHECK:  ![[FOO4:.*]] = !DILocalVariable(name: "foo4"
; CHECK:  ![[BAR4:.*]] = !DILocalVariable(name: "bar4"
; CHECK:  ![[BAR5:.*]] = !DILocalVariable(name: "bar5"
; CHECK:  ![[FOO5:.*]] = !DILocalVariable(name: "foo5"


source_filename = "sdag-dangling-dbgvalue.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.4.0"

%struct.SS = type { i32, i32 }

@S = global %struct.SS { i32 23, i32 -17 }, align 4, !dbg !0

; Verify that the def comes before the for foo1.
define i32 @test1() local_unnamed_addr #0 !dbg !17 {
; CHECK-LABEL: bb.0.entry1
; CHECK-NEXT:    DBG_VALUE 0, $noreg, ![[BAR1]], !DIExpression()
; CHECK-NEXT:    [[REG1:%[0-9]+]]:gr64 =
; CHECK-NEXT:    DBG_VALUE [[REG1]], $noreg, ![[FOO1]], !DIExpression()
entry1:
  call void @llvm.dbg.value(metadata %struct.SS* @S, metadata !20, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata %struct.SS* null, metadata !22, metadata !DIExpression()), !dbg !24
  ret i32 ptrtoint (%struct.SS* @S to i32), !dbg !25
}

; Verify that the def comes before the for foo2 and bar2.
define i32 @test2() local_unnamed_addr #0 !dbg !26 {
; CHECK-LABEL: bb.0.entry2
; CHECK-NEXT:    [[REG2:%[0-9]+]]:gr64 =
; CHECK-NEXT:    DBG_VALUE [[REG2]], $noreg, ![[FOO2]], !DIExpression()
; CHECK-NEXT:    DBG_VALUE [[REG2]], $noreg, ![[BAR2]], !DIExpression()
entry2:
  call void @llvm.dbg.value(metadata %struct.SS* @S, metadata !28, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata %struct.SS* @S, metadata !29, metadata !DIExpression()), !dbg !31
  ret i32 add (i32 ptrtoint (%struct.SS* @S to i32), i32 ptrtoint (%struct.SS* @S to i32)), !dbg !32
}

; Verify that the def comes before the for foo3 and bar3.
define i32 @test3() local_unnamed_addr #0 !dbg !33 {
; CHECK-LABEL: bb.0.entry3
; CHECK-NEXT:    [[REG3:%[0-9]+]]:gr64 =
; CHECK-NEXT:    DBG_VALUE [[REG3]], $noreg, ![[BAR3]], !DIExpression()
; CHECK-NEXT:    DBG_VALUE [[REG3]], $noreg, ![[FOO3]], !DIExpression()
entry3:
  call void @llvm.dbg.value(metadata %struct.SS* @S, metadata !36, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata %struct.SS* @S, metadata !35, metadata !DIExpression()), !dbg !37
  ret i32 add (i32 ptrtoint (%struct.SS* @S to i32), i32 ptrtoint (%struct.SS* @S to i32)), !dbg !39
}

; Verify that the def comes before the for bar4.
define i32 @test4() local_unnamed_addr #0 !dbg !40 {
; CHECK-LABEL: bb.0.entry4
; CHECK-NEXT:    DBG_VALUE $noreg, $noreg, ![[FOO4]], !DIExpression()
; CHECK-NEXT:    DBG_VALUE 0, $noreg, ![[FOO4]], !DIExpression()
; CHECK-NEXT:    [[REG4:%[0-9]+]]:gr64 =
; CHECK-NEXT:    DBG_VALUE [[REG4]], $noreg, ![[BAR4]], !DIExpression()
entry4:
  call void @llvm.dbg.value(metadata %struct.SS* @S, metadata !42, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata %struct.SS* @S, metadata !43, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata %struct.SS* null, metadata !42, metadata !DIExpression()), !dbg !44
  ret i32 ptrtoint (%struct.SS* @S to i32), !dbg !46
}

; Verify that we do not get a DBG_VALUE that maps foo5 to @S here.
define i32 @test5() local_unnamed_addr #0 !dbg !47 {
; CHECK-LABEL: bb.0.entry5:
; CHECK-NEXT:    DBG_VALUE $noreg, $noreg, ![[FOO5]], !DIExpression()
; CHECK-NEXT:    DBG_VALUE 0, $noreg, ![[FOO5]], !DIExpression()
; CHECK-NEXT:    [[REG5:%[0-9]+]]:gr64 =
; CHECK-NEXT:    DBG_VALUE [[REG5]], $noreg, ![[BAR5]], !DIExpression()
; CHECK-NOT:     DBG_VALUE [[REG5]], $noreg, ![[FOO5]], !DIExpression()
; CHECK:         RET
entry5:
  call void @llvm.dbg.value(metadata %struct.SS* @S, metadata !49, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata %struct.SS* @S, metadata !50, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata %struct.SS* null, metadata !50, metadata !DIExpression()), !dbg !52
  ret i32 ptrtoint (%struct.SS* @S to i32), !dbg !53
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "S", scope: !2, file: !3, line: 4, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 327229) (llvm/trunk 327239)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7)
!3 = !DIFile(filename: "sdag-dangling-dbgvalue.c", directory: "/repo/uabbpet/llvm-master")
!4 = !{}
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!0}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "SS", file: !3, line: 1, size: 64, elements: !9)
!9 = !{!10, !11}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !8, file: !3, line: 2, baseType: !6, size: 32)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !8, file: !3, line: 3, baseType: !6, size: 32, offset: 32)
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"PIC Level", i32 2}
!16 = !{!"clang version 7.0.0 (trunk 327229) (llvm/trunk 327239)"}
!17 = distinct !DISubprogram(name: "test1", scope: !3, file: !3, line: 6, type: !18, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !2, retainedNodes: !19)
!18 = !DISubroutineType(types: !5)
!19 = !{!20, !22}
!20 = !DILocalVariable(name: "foo1", scope: !17, file: !3, line: 7, type: !21)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!22 = !DILocalVariable(name: "bar1", scope: !17, file: !3, line: 8, type: !21)
!23 = !DILocation(line: 7, column: 14, scope: !17)
!24 = !DILocation(line: 8, column: 14, scope: !17)
!25 = !DILocation(line: 9, column: 3, scope: !17)
!26 = distinct !DISubprogram(name: "test2", scope: !3, file: !3, line: 12, type: !18, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: true, unit: !2, retainedNodes: !27)
!27 = !{!28, !29}
!28 = !DILocalVariable(name: "foo2", scope: !26, file: !3, line: 13, type: !21)
!29 = !DILocalVariable(name: "bar2", scope: !26, file: !3, line: 14, type: !21)
!30 = !DILocation(line: 13, column: 14, scope: !26)
!31 = !DILocation(line: 14, column: 14, scope: !26)
!32 = !DILocation(line: 15, column: 3, scope: !26)
!33 = distinct !DISubprogram(name: "test3", scope: !3, file: !3, line: 18, type: !18, isLocal: false, isDefinition: true, scopeLine: 18, isOptimized: true, unit: !2, retainedNodes: !34)
!34 = !{!35, !36}
!35 = !DILocalVariable(name: "bar3", scope: !33, file: !3, line: 19, type: !21)
!36 = !DILocalVariable(name: "foo3", scope: !33, file: !3, line: 20, type: !21)
!37 = !DILocation(line: 19, column: 14, scope: !33)
!38 = !DILocation(line: 20, column: 14, scope: !33)
!39 = !DILocation(line: 21, column: 3, scope: !33)
!40 = distinct !DISubprogram(name: "test4", scope: !3, file: !3, line: 24, type: !18, isLocal: false, isDefinition: true, scopeLine: 24, isOptimized: true, unit: !2, retainedNodes: !41)
!41 = !{!42, !43}
!42 = !DILocalVariable(name: "foo4", scope: !40, file: !3, line: 25, type: !21)
!43 = !DILocalVariable(name: "bar4", scope: !40, file: !3, line: 26, type: !21)
!44 = !DILocation(line: 25, column: 14, scope: !40)
!45 = !DILocation(line: 26, column: 14, scope: !40)
!46 = !DILocation(line: 28, column: 3, scope: !40)
!47 = distinct !DISubprogram(name: "test5", scope: !3, file: !3, line: 31, type: !18, isLocal: false, isDefinition: true, scopeLine: 31, isOptimized: true, unit: !2, retainedNodes: !48)
!48 = !{!49, !50}
!49 = !DILocalVariable(name: "bar5", scope: !47, file: !3, line: 32, type: !21)
!50 = !DILocalVariable(name: "foo5", scope: !47, file: !3, line: 33, type: !21)
!51 = !DILocation(line: 32, column: 14, scope: !47)
!52 = !DILocation(line: 33, column: 14, scope: !47)
!53 = !DILocation(line: 35, column: 3, scope: !47)
