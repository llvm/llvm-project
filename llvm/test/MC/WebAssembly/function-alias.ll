; RUN: llc -filetype=obj %s -o - | llvm-readobj --symbols - | FileCheck %s
; RUN: llc -filetype=obj %s -mattr=+reference-types -o - | llvm-readobj --symbols - | FileCheck --check-prefix=REF %s

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

; REF:      Symbols [
; REF-NEXT:   Symbol {
; REF-NEXT:     Name: func
; REF-NEXT:     Type: FUNCTION (0x0)
; REF-NEXT:     Flags [ (0x0)
; REF-NEXT:     ]
; REF-NEXT:     ElementIndex: 0x0
; REF-NEXT:   }
; REF-NEXT:   Symbol {
; REF-NEXT:     Name: bar2
; REF-NEXT:     Type: FUNCTION (0x0)
; REF-NEXT:     Flags [ (0x0)
; REF-NEXT:     ]
; REF-NEXT:     ElementIndex: 0x0
; REF-NEXT:   }
; REF-NEXT:   Symbol {
; REF-NEXT:     Name: foo
; REF-NEXT:     Type: FUNCTION (0x0)
; REF-NEXT:     Flags [ (0x0)
; REF-NEXT:     ]
; REF-NEXT:     ElementIndex: 0x0
; REF-NEXT:   }
; REF-NEXT:   Symbol {
; REF-NEXT:     Name: bar
; REF-NEXT:     Type: FUNCTION (0x0)
; REF-NEXT:     Flags [ (0x0)
; REF-NEXT:     ]
; REF-NEXT:     ElementIndex: 0x0
; REF-NEXT:   }
; REF-NEXT:   Symbol {
; REF-NEXT:     Name: __indirect_function_table
; REF-NEXT:     Type: TABLE (0x5)
; REF-NEXT:     Flags [ (0x90)
; REF-NEXT:       NO_STRIP (0x80)
; REF-NEXT:       UNDEFINED (0x10)
; REF-NEXT:     ]
; REF-NEXT:     ImportModule: env
; REF-NEXT:     ElementIndex: 0x0
; REF-NEXT:   }
; REF-NEXT: ]
