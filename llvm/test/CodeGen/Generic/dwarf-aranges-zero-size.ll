; Ensures that the AsmPrinter doesn't emit zero-sized symbols into `.debug_aranges`.
;
; RUN: llc --generate-arange-section < %s | FileCheck %s
; CHECK: .section .debug_aranges
; CHECK: .quad EXAMPLE
; CHECK-NEXT: .quad 1
; CHECK: .section

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@EXAMPLE = constant <{ [0 x i8] }> zeroinitializer, align 1, !dbg !0

!llvm.module.flags = !{!3}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "EXAMPLE", linkageName: "EXAMPLE", scope: null, file: null, line: 161, type: !2, isLocal: false, isDefinition: true, align: 1)
!2 = !DIBasicType(name: "()", encoding: DW_ATE_unsigned)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !5, producer: "rustc", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: null, globals: !6)
!5 = !DIFile(filename: "foo", directory: "")
!6 = !{!0}
