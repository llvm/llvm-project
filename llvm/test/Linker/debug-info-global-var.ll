; RUN: llvm-link -S %s %S/debug-info-version-a.ll | FileCheck %s
source_filename = "debug-info-global-var.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; CHECK: @g = global i32 0, align 4, !dbg ![[G:[0-9]+]]{{$}}
@g = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

; CHECK: ![[G]] = !DIGlobalVariableExpression(var: ![[GVAR:.*]], expr: !DIExpression())
; CHECK: ![[GVAR]] = distinct !DIGlobalVariable(name: "g"
!0 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "debug-info-global-var.c", directory: "/")
!3 = !{}
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"PIC Level", i32 2}
!9 = !{!"clang version 4.0.0 (trunk 286129) (llvm/trunk 286128)"}
!10 = distinct !DIGlobalVariable(name: "g", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true)
