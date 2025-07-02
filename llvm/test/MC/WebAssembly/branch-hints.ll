; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=asm %s -mattr=+branch-hinting -o - | FileCheck --check-prefixes=ASM-CHECK %s
; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=asm %s -mattr=-branch-hinting -o - | FileCheck --check-prefixes=ASM-NCHECK %s
; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=asm %s -o - | FileCheck --check-prefixes=ASM-NCHECK %s
; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=obj %s -mattr=+branch-hinting -o - | obj2yaml | FileCheck --check-prefixes=YAML-CHECK %s
; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=obj %s -mattr=-branch-hinting -o - | obj2yaml | FileCheck --check-prefixes=YAML-NCHECK %s
; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=obj %s -o - | obj2yaml | FileCheck --check-prefixes=YAML-NCHECK %s

; This test checks that branch weight metadata (!prof) is correctly lowered to
; the WebAssembly branch hint custom section.

; ASM-CHECK:        test_unlikely_likely_branch:            # @test_unlikely_likely_branch
; ASM-CHECK:        .Ltmp0:
; ASM-CHECK-NEXT:     br_if           0                               # 0: down to label1
; ASM-CHECK:        .Ltmp1:
; ASM-CHECK-NEXT:     br_if           1                               # 1: down to label0

; ASM-CHECK:        test_likely_branch:                     # @test_likely_branch
; ASM-CHECK:        .Ltmp2:
; ASM-CHECK-NEXT:     br_if           0                               # 0: down to label2

; ASM-CHECK:        .section        .custom_section.target_features,"",@
; ASM-CHECK-NEXT:   .int8   1
; ASM-CHECK-NEXT:   .int8   43
; ASM-CHECK-NEXT:   .int8   14
; ASM-CHECK-NEXT:   .ascii  "branch-hinting"

; ASM-CHECK:        .section        .custom_section.metadata.code.branch_hint,"",@
; ASM-CHECK-NEXT:   .asciz  "\202\200\200\200"
; ASM-CHECK-NEXT:   .uleb128 test_unlikely_likely_branch@FUNCINDEX
; ASM-CHECK-NEXT:   .int8   2
; ASM-CHECK-NEXT:   .uleb128 .Ltmp0@DEBUGREF
; ASM-CHECK-NEXT:   .int8   1
; ASM-CHECK-NEXT:   .int8   0
; ASM-CHECK-NEXT:   .uleb128 .Ltmp1@DEBUGREF
; ASM-CHECK-NEXT:   .int8   1
; ASM-CHECK-NEXT:   .int8   1
; ASM-CHECK-NEXT:   .uleb128 test_likely_branch@FUNCINDEX
; ASM-CHECK-NEXT:   .int8   1
; ASM-CHECK-NEXT:   .uleb128 .Ltmp2@DEBUGREF
; ASM-CHECK-NEXT:   .int8   1
; ASM-CHECK-NEXT:   .int8   1

; ASM-NCHECK-NOT:   .ascii      "branch-hinting"
; ASM-NCHECK-NOT:   .section        metadata.code.branch_hint,"",@

; YAML-CHECK:        - Type:            CUSTOM
; YAML-CHECK-NEXT:     Relocations:
; YAML-CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; YAML-CHECK-NEXT:         Index:           0
; YAML-CHECK-NEXT:         Offset:          0x5
; YAML-CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; YAML-CHECK-NEXT:         Index:           1
; YAML-CHECK-NEXT:         Offset:          0x11
; YAML-CHECK-NEXT:     Name:            metadata.code.branch_hint
; YAML-CHECK-NEXT:     Entries:
; YAML-CHECK-NEXT:       - FuncIdx:         0
; YAML-CHECK-NEXT:         Hints:
; YAML-CHECK-NEXT:           - Offset:          7
; YAML-CHECK-NEXT:             Size:            1
; YAML-CHECK-NEXT:             Data:            UNLIKELY
; YAML-CHECK-NEXT:           - Offset:          14
; YAML-CHECK-NEXT:             Size:            1
; YAML-CHECK-NEXT:             Data:            LIKELY
; YAML-CHECK-NEXT:       - FuncIdx:         1
; YAML-CHECK-NEXT:         Hints:
; YAML-CHECK-NEXT:           - Offset:          5
; YAML-CHECK-NEXT:             Size:            1
; YAML-CHECK-NEXT:             Data:            LIKELY

; YAML-CHECK:        - Type:            CUSTOM
; YAML-CHECK-NEXT:     Name:            linking
; YAML-CHECK-NEXT:     Version:         2
; YAML-CHECK-NEXT:     SymbolTable:
; YAML-CHECK-NEXT:       - Index:           0
; YAML-CHECK-NEXT:         Kind:            FUNCTION
; YAML-CHECK-NEXT:         Name:            test_unlikely_likely_branch
; YAML-CHECK-NEXT:         Flags:           [  ]
; YAML-CHECK-NEXT:         Function:        0
; YAML-CHECK-NEXT:       - Index:           1
; YAML-CHECK-NEXT:         Kind:            FUNCTION
; YAML-CHECK-NEXT:         Name:            test_likely_branch
; YAML-CHECK-NEXT:         Flags:           [  ]
; YAML-CHECK-NEXT:         Function:        1

; YAML-CHECK:        - Type:            CUSTOM
; YAML-CHECK-NEXT:     Name:            target_features
; YAML-CHECK-NEXT:     Features:
; YAML-CHECK-NEXT:       - Prefix:          USED
; YAML-CHECK-NEXT:         Name:            branch-hinting

; YAML-NCHECK-NOT:     Name:            metadata.code.branch_hint
; YAML-NCHECK-NOT:     Name:            branch-hinting

target triple = "wasm32-unknown-unknown"

define i32 @test_unlikely_likely_branch(i32 %a) {
entry:
  %cmp0 = icmp eq i32 %a, 0
  ; This metadata hints that the true branch is overwhelmingly likely.
  br i1 %cmp0, label %if.then, label %ret1, !prof !0
if.then:
  %cmp1 = icmp eq i32 %a, 1
  br i1 %cmp1, label %ret1, label %ret2, !prof !1
ret1:
  ret i32 2
ret2:
  ret i32 1
}

define i32 @test_likely_branch(i32 %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else, !prof !1
if.then:
  ret i32 1
if.else:
  ret i32 2
}

; the resulting branch hint is actually reversed, since llvm-br is turned into br_unless, inverting branch probs
!0 = !{!"branch_weights", i32 2000, i32 1}
!1 = !{!"branch_weights", i32 1, i32 2000}
