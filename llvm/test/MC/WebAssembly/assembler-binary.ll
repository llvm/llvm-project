; RUN: llc -filetype=asm -asm-verbose=false %s -o %t.s
; RUN: FileCheck -check-prefix=ASM -input-file %t.s %s
; RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=asm %t.s -o - | FileCheck -check-prefix=ASM %s
; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s
; RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %t.s -o - | obj2yaml | FileCheck %s

; This specifically tests that we can generate a binary from the assembler
; that produces the same binary as the backend would.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @bar()

define void @foo(i32 %n) {
entry:
  call void @bar()
  ret void
}

; Checking assembly is not the point of this test, but if something breaks
; it is easier to spot it here than in the yaml output.

; ASM:     	.text
; ASM:      	.file	"assembler-binary.ll"
; ASM:      	.globl	foo
; ASM:      foo:
; ASM-NEXT: 	.functype	foo (i32) -> ()
; ASM-NEXT: 	call	bar@FUNCTION
; ASM-NEXT: 	end_function
; ASM-NEXT: .Lfunc_end0:
; ASM-NEXT: 	.size	foo, .Lfunc_end0-foo
; ASM:       	.functype	bar () -> ()


; CHECK:      --- !WASM
; CHECK-NEXT: FileHeader:
; CHECK-NEXT:   Version:         0x00000001
; CHECK-NEXT: Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ReturnType:      NORESULT
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ReturnType:      NORESULT
; CHECK-NEXT:         ParamTypes:      []
; CHECK-NEXT:   - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __linear_memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Memory:
; CHECK-NEXT:           Initial:         0x00000000
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __indirect_function_table
; CHECK-NEXT:         Kind:            TABLE
; CHECK-NEXT:         Table:
; CHECK-NEXT:           ElemType:        FUNCREF
; CHECK-NEXT:           Limits:
; CHECK-NEXT:             Initial:         0x00000000
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           bar
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        1
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0 ]
; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Relocations:
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x00000004
; CHECK-NEXT:     Functions:
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Locals:          []
; CHECK-NEXT:         Body:            1080808080000B
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     Version:         2
; CHECK-NEXT:     SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            foo
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Function:        1
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            bar
; CHECK-NEXT:         Flags:           [ UNDEFINED ]
; CHECK-NEXT:         Function:        0
; CHECK-NEXT: ...
