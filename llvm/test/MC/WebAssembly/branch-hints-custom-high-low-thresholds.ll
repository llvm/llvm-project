; RUN: llc -mcpu=mvp -filetype=asm %s -mattr=+branch-hinting -wasm-branch-prob-high=.5 -wasm-branch-prob-low=0.5 -o - | FileCheck --check-prefixes=C1 %s
; RUN: llc -mcpu=mvp -filetype=asm %s -mattr=+branch-hinting -wasm-branch-prob-high=.76 -wasm-branch-prob-low=0 -o - | FileCheck --check-prefixes=C2 %s
; RUN: llc -mcpu=mvp -filetype=asm %s -mattr=+branch-hinting -wasm-branch-prob-high=.75 -wasm-branch-prob-low=0 -o - | FileCheck --check-prefixes=C3 %s

; C1:	.section	.custom_section.target_features,"",@
; C1-NEXT:	.int8	1
; C1-NEXT:	.int8	43
; C1-NEXT:	.int8	14
; C1-NEXT:	.ascii	"branch-hinting"
; C1-NEXT:	.section	.text.test2,"",@
; C1-NEXT:	.section	.custom_section.metadata.code.branch_hint,"",@
; C1-NEXT:	.int8	3
; C1-NEXT:	.uleb128 test0@FUNCINDEX
; C1-NEXT:	.int8	1
; C1-NEXT:	.uleb128 .Ltmp0@DEBUGREF
; C1-NEXT:	.int8	1
; C1-NEXT:	.int8	1
; C1-NEXT:	.uleb128 test1@FUNCINDEX
; C1-NEXT:	.int8	1
; C1-NEXT:	.uleb128 .Ltmp1@DEBUGREF
; C1-NEXT:	.int8	1
; C1-NEXT:	.int8	0
; C1-NEXT:	.uleb128 test2@FUNCINDEX
; C1-NEXT:	.int8	1
; C1-NEXT:	.uleb128 .Ltmp2@DEBUGREF
; C1-NEXT:	.int8	1
; C1-NEXT:	.int8	0
; C1-NEXT:	.section	.text.test2,"",@

; C2:	.section	.custom_section.target_features,"",@
; C2-NEXT:	.int8	1
; C2-NEXT:	.int8	43
; C2-NEXT:	.int8	14
; C2-NEXT:	.ascii	"branch-hinting"
; C2-NEXT:	.section	.text.test2,"",@
; C2-NEXT:	.section	.custom_section.metadata.code.branch_hint,"",@
; C2-NEXT:	.int8	1
; C2-NEXT:	.uleb128 test2@FUNCINDEX
; C2-NEXT:	.int8	1
; C2-NEXT:	.uleb128 .Ltmp2@DEBUGREF
; C2-NEXT:	.int8	1
; C2-NEXT:	.int8	0
; C2-NEXT:	.section	.text.test2,"",@

; C3:	.section	.custom_section.target_features,"",@
; C3-NEXT:	.int8	1
; C3-NEXT:	.int8	43
; C3-NEXT:	.int8	14
; C3-NEXT:	.ascii	"branch-hinting"
; C3-NEXT:	.section	.text.test2,"",@
; C3-NEXT:	.section	.custom_section.metadata.code.branch_hint,"",@
; C3-NEXT:	.int8	2
; C3-NEXT:	.uleb128 test0@FUNCINDEX
; C3-NEXT:	.int8	1
; C3-NEXT:	.uleb128 .Ltmp0@DEBUGREF
; C3-NEXT:	.int8	1
; C3-NEXT:	.int8	1
; C3-NEXT:	.uleb128 test2@FUNCINDEX
; C3-NEXT:	.int8	1
; C3-NEXT:	.uleb128 .Ltmp2@DEBUGREF
; C3-NEXT:	.int8	1
; C3-NEXT:	.int8	0
; C3-NEXT:	.section	.text.test2,"",@

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
