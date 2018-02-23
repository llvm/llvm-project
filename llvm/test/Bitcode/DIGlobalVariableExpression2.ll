; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: @g = common global i32 0, align 4, !dbg ![[G:[0-9]+]]
; CHECK-DAG: ![[G]] = distinct !DIGlobalVariableExpression(var: ![[GVAR:[0-9]+]])
; CHECK-DAG: distinct !DICompileUnit({{.*}}, globals: ![[GLOBS:[0-9]+]]
; CHECK-DAG: ![[GLOBS]] = !{![[GEXPR:[0-9]+]]}
; CHECK-DAG: ![[GEXPR]] = distinct !DIGlobalVariableExpression(var: ![[GVAR]])
; CHECK-DAG: ![[GVAR]] = !DIGlobalVariable(name: "g",

; Test the bitcode upgrade for DIGlobalVariable -> DIGlobalVariableExpression.

; ModuleID = 'a.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@g = common global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (clang-stage1-configure-RA_build 241111)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "g", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, variable: i32* @g)
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 2}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"PIC Level", i32 2}
!9 = !{!"clang version 3.7.0 (clang-stage1-configure-RA_build 241111)"}
