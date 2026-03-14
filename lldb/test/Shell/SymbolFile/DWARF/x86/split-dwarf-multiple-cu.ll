; Check handling of dwo files with multiple compile units. Right now this is not
; supported, but it should not cause us to crash or misbehave either...

; RUN: llc %s -filetype=obj -o %t.o --split-dwarf-file=%t.o -split-dwarf-cross-cu-references
; RUN: %lldb %t.o -o "image lookup -s x1 -v" -o "image lookup -s x2 -v" -b | FileCheck %s

; CHECK: image lookup -s x1
; CHECK: 1 symbols match 'x1'
; CHECK-NOT: CompileUnit:
; CHECK-NOT: Variable:

; CHECK: image lookup -s x2
; CHECK: 1 symbols match 'x2'
; CHECK-NOT: CompileUnit:
; CHECK-NOT: Variable:

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x1 = dso_local global i32 42, align 4, !dbg !10
@x2 = dso_local global i32 47, align 4, !dbg !20

!llvm.dbg.cu = !{!12, !22}
!llvm.module.flags = !{!8, !9, !1}

!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "x1", scope: !12, type: !7)
!12 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, globals: !15)
!15 = !{!10}

!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "x2", scope: !22, type: !7)
!22 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, globals: !25)
!25 = !{!20}

!3 = !DIFile(filename: "-", directory: "/tmp")
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
