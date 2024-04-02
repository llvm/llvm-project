; Ensure that symbols from module ASM are properly exported.
;
; Regression test for https://github.com/llvm/llvm-project/issues/85578.

; RUN: llc -mtriple=wasm32-unknown-unknown -filetype=obj %s -o - | obj2yaml | FileCheck %s

module asm "test_func:"
module asm "    .globl test_func"
module asm "    .functype test_func (i32) -> (i32)"
module asm "    .export_name test_func, test_export"
module asm "    end_function"

; CHECK:       - Type:            TYPE
; CHECK-NEXT:      Signatures:
; CHECK-NEXT:        - Index:           0
; CHECK-NEXT:          ParamTypes:
; CHECK-NEXT:            - I32
; CHECK-NEXT:          ReturnTypes:
; CHECK-NEXT:            - I32

; CHECK:        - Type:            EXPORT
; CHECK-NEXT:     Exports:
; CHECK-NEXT:       - Name:            test_export
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           0
