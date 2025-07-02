; RUN: llc -mcpu=mvp -filetype=obj %s -mattr=+branch-hinting -wasm-branch-prob-high=.5 -wasm-branch-prob-low=0.5 -o - | obj2yaml | FileCheck --check-prefixes=C1 %s
; RUN: llc -mcpu=mvp -filetype=obj %s -mattr=+branch-hinting -wasm-branch-prob-high=.76 -wasm-branch-prob-low=0 -o - | obj2yaml | FileCheck --check-prefixes=C2 %s
; RUN: llc -mcpu=mvp -filetype=obj %s -mattr=+branch-hinting -wasm-branch-prob-high=.75 -wasm-branch-prob-low=0 -o - | obj2yaml | FileCheck --check-prefixes=C3 %s

; This test checks that branch weight metadata (!prof) is correctly translated to webassembly branch hints
; We set the prob-thresholds so that "likely" branches are only emitted if prob > 75% and "unlikely" branches
; if prob <= 0%.

; C1:   - Type:            CUSTOM
; C1-NEXT:     Relocations:
; C1-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; C1-NEXT:         Index:           0
; C1-NEXT:         Offset:          0x5
; C1-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; C1-NEXT:         Index:           1
; C1-NEXT:         Offset:          0xE
; C1-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; C1-NEXT:         Index:           2
; C1-NEXT:         Offset:          0x17
; C1-NEXT:     Name:            metadata.code.branch_hint
; C1-NEXT:     Entries:
; C1-NEXT:       - FuncIdx:         0
; C1-NEXT:         Hints:
; C1-NEXT:           - Offset:          5
; C1-NEXT:             Size:            1
; C1-NEXT:             Data:            LIKELY
; C1-NEXT:       - FuncIdx:         1
; C1-NEXT:         Hints:
; C1-NEXT:           - Offset:          5
; C1-NEXT:             Size:            1
; C1-NEXT:             Data:            UNLIKELY
; C1-NEXT:       - FuncIdx:         2
; C1-NEXT:         Hints:
; C1-NEXT:           - Offset:          5
; C1-NEXT:             Size:            1
; C1-NEXT:             Data:            UNLIKELY

; C2:   - Type:            CUSTOM
; C2-NEXT:     Relocations:
; C2-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; C2-NEXT:         Index:           2
; C2-NEXT:         Offset:          0x5
; C2-NEXT:     Name:            metadata.code.branch_hint
; C2-NEXT:     Entries:
; C2-NEXT:       - FuncIdx:         2
; C2-NEXT:         Hints:
; C2-NEXT:           - Offset:          5
; C2-NEXT:             Size:            1
; C2-NEXT:             Data:            UNLIKELY

; C3:   - Type:            CUSTOM
; C3-NEXT:     Relocations:
; C3-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; C3-NEXT:         Index:           0
; C3-NEXT:         Offset:          0x5
; C3-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; C3-NEXT:         Index:           2
; C3-NEXT:         Offset:          0xE
; C3-NEXT:     Name:            metadata.code.branch_hint
; C3-NEXT:     Entries:
; C3-NEXT:       - FuncIdx:         0
; C3-NEXT:         Hints:
; C3-NEXT:           - Offset:          5
; C3-NEXT:             Size:            1
; C3-NEXT:             Data:            LIKELY
; C3-NEXT:       - FuncIdx:         2
; C3-NEXT:         Hints:
; C3-NEXT:           - Offset:          5
; C3-NEXT:             Size:            1
; C3-NEXT:             Data:            UNLIKELY

; CHECK:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     Version:         2
; CHECK-NEXT:     SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            test0
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Function:        0
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            test1
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Function:        1
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            test2
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Function:        2

target triple = "wasm32-unknown-unknown"

define i32 @test0(i32 %a) {
entry:
  %cmp0 = icmp eq i32 %a, 0
  br i1 %cmp0, label %if_then, label %if_else, !prof !0
if_then:
  ret i32 1
if_else:
  ret i32 0
}

define i32 @test1(i32 %a) {
entry:
  %cmp0 = icmp eq i32 %a, 0
  br i1 %cmp0, label %if_then, label %if_else, !prof !1
if_then:
  ret i32 1
if_else:
  ret i32 0
}

define i32 @test2(i32 %a) {
entry:
  %cmp0 = icmp eq i32 %a, 0
  br i1 %cmp0, label %if_then, label %if_else, !prof !2
if_then:
  ret i32 1
if_else:
  ret i32 0
}

; the resulting branch hint is actually reversed, since llvm-br is turned into br_unless, inverting branch probs
!0 = !{!"branch_weights", !"expected", i32 100, i32 310} ; prob 75.61%
!1 = !{!"branch_weights", i32 1, i32 1}                  ; prob == 50% (no hint)
!2 = !{!"branch_weights", i32 1, i32 0}                  ; prob == 0% (unlikely hint)
