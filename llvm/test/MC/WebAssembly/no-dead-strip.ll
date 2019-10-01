; RUN: llc -filetype=obj -wasm-keep-registers %s -o - | llvm-readobj --symbols | FileCheck %s

target triple = "wasm32-unknown-unknown"

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @foo to i8*)], section "llvm.metadata"

define i32 @foo() {
entry:
    ret i32 0
}

; CHECK:      Symbols [
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: foo
; CHECK-NEXT:     Type: FUNCTION (0x0)
; CHECK-NEXT:     Flags [ (0x20)
; CHECK-NEXT:       EXPORTED (0x20)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x0
; CHECK-NEXT:   }
; CHECK-NEXT: ]
