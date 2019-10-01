; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test the mechanism for defining user custom sections.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

!0 = !{ !"red", !"foo" }
!1 = !{ !"green", !"bar" }
!2 = !{ !"green", !"qux" }
!wasm.custom_sections = !{ !0, !1, !2 }

!llvm.ident = !{!3}
!3 = !{!"clang version 123"}

!llvm.module.flags = !{!4}
!4 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!5}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6)
!6 = !DIFile(filename: "test", directory: "testdir")

; CHECK:      .section	.custom_section.red,"",@
; CHECK-NEXT: .ascii	"foo"

; CHECK:      .section	.custom_section.green,"",@
; CHECK-NEXT: .ascii	"bar"

; CHECK:      .section	.custom_section.green,"",@
; CHECK-NEXT: .ascii	"qux"

; CHECK:      .section	.custom_section.producers,"",@
; CHECK-NEXT: .int8	2
; CHECK-NEXT: .int8	8
; CHECK-NEXT: .ascii	"language"
; CHECK-NEXT: .int8	1
; CHECK-NEXT: .int8	3
; CHECK-NEXT: .ascii	"C99"
; CHECK-NEXT: .int8	0
; CHECK-NEXT: .int8	12
; CHECK-NEXT: .ascii	"processed-by"
; CHECK-NEXT: .int8	1
; CHECK-NEXT: .int8	5
; CHECK-NEXT: .ascii	"clang"
; CHECK-NEXT: .int8	3
; CHECK-NEXT: .ascii	"123"
