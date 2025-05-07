; RUN: %llc_dwarf -filetype=obj -O0 %s -o - | llvm-dwarfdump -debug-info - | FileCheck %s

; Check that DW_AT_abstract_origin is generated for concrete lexical block.

; Generated from:
; inline __attribute__((always_inline)) int foo(int x) {
;   {
;     int y = x + 5;
;     return y - 10;
;   }
; }
;
; int bar(int x) {
;   int y = foo(7);
;   return y + 8;
; }

; CHECK:      DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_name  ("foo")
; CHECK-NOT:    {{DW_TAG|NULL}}
; CHECK:        [[LB:.*]]: DW_TAG_lexical_block

; CHECK:        DW_TAG_inlined_subroutine
; CHECK-NEXT:     DW_AT_abstract_origin {{.*}} "foo"
; CHECK-NOT:      {{DW_TAG|NULL}}
; CHECK:          DW_TAG_lexical_block
; CHECK-NOT:        {{DW_TAG|NULL}}
; CHECK:            DW_AT_abstract_origin {{.*}}[[LB]]

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

define i32 @bar() !dbg !9 {
entry:
  %y.i = alloca i32, align 4
    #dbg_declare(ptr %y.i, !22, !DIExpression(), !24)
  store i32 0, ptr %y.i, align 4, !dbg !24
  %1 = load i32, ptr %y.i, align 4
  ret i32 %1
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 8, type: !10, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !13)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!19 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!21 = distinct !DILocation(line: 9, column: 11, scope: !9)
!22 = !DILocalVariable(name: "y", scope: !23, file: !1, line: 3, type: !12)
!23 = distinct !DILexicalBlock(scope: !19, file: !1, line: 2, column: 3)
!24 = !DILocation(line: 3, column: 9, scope: !23, inlinedAt: !21)
