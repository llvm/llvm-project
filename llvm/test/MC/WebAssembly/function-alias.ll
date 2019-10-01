; RUN: llc -filetype=obj %s -o - | llvm-readobj --symbols | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

@foo = alias i8, bitcast (i8* ()* @func to i8*)
@bar = alias i8* (), i8* ()* @func
@bar2 = alias i8* (), i8* ()* @bar

define i8* @func() {
  call i8* @bar2();
  ret i8* @foo;
}

; CHECK:      Symbols [
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: func
; CHECK-NEXT:     Type: FUNCTION (0x0)
; CHECK-NEXT:     Flags [ (0x0)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x0
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: bar2
; CHECK-NEXT:     Type: FUNCTION (0x0)
; CHECK-NEXT:     Flags [ (0x0)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x0
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: foo
; CHECK-NEXT:     Type: FUNCTION (0x0)
; CHECK-NEXT:     Flags [ (0x0)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x0
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: bar
; CHECK-NEXT:     Type: FUNCTION (0x0)
; CHECK-NEXT:     Flags [ (0x0)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x0
; CHECK-NEXT:   }
; CHECK-NEXT: ]
