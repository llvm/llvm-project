; RUN: rm -rf %t; split-file %s %t

; check ll -> asm
; RUN: llc -mcpu=mvp -filetype=asm %t/1.ll -mattr=+branch-hinting -wasm-branch-prob-high=.5 -wasm-branch-prob-low=0.5 -o - | FileCheck --check-prefixes=C1 %s
; RUN: llc -mcpu=mvp -filetype=asm %t/1.ll -mattr=+branch-hinting -wasm-branch-prob-high=.76 -wasm-branch-prob-low=0 -o - | FileCheck --check-prefixes=C2 %s
; RUN: llc -mcpu=mvp -filetype=asm %t/1.ll -mattr=+branch-hinting -wasm-branch-prob-high=.75 -wasm-branch-prob-low=0 -o - | FileCheck --check-prefixes=C3 %s

; check asm -> obj -> yaml
; RUN: llvm-mc -mcpu=mvp -triple=wasm32-unknown-unknown -filetype=obj %t/1.S -o - | obj2yaml | FileCheck --check-prefixes=YAML-CHECK %s

; C1:	.section	.custom_section.target_features,"",@
; C1-NEXT:	.int8	1
; C1-NEXT:	.int8	43
; C1-NEXT:	.int8	14
; C1-NEXT:	.ascii	"branch-hinting"
; C1-NEXT:	.section	.text.test2,"",@
; C1:       .section  .custom_section.metadata.code.branch_hint,"",@
; C1-NEXT:  .int8 3
; C1-DAG:   .uleb128 test0@FUNCINDEX
; C1-DAG-NEXT:   .int8 1
; C1-DAG-NEXT:   .uleb128 .Ltmp0
; C1-DAG-NEXT:   .int8 1
; C1-DAG-NEXT:   .int8 1
; C1-DAG:   .uleb128 test1@FUNCINDEX
; C1-DAG-NEXT:   .int8 1
; C1-DAG-NEXT:   .uleb128 .Ltmp1
; C1-DAG-NEXT:   .int8 1
; C1-DAG-NEXT:   .int8 0
; C1-DAG:   .uleb128 test2@FUNCINDEX
; C1-DAG-NEXT:   .int8 1
; C1-DAG-NEXT:   .uleb128 .Ltmp2
; C1-DAG-NEXT:   .int8 1
; C1-DAG-NEXT:   .int8 0
; C1:  .section  .text.test2,"",@

; C2:	.section	.custom_section.target_features,"",@
; C2-NEXT:	.int8	1
; C2-NEXT:	.int8	43
; C2-NEXT:	.int8	14
; C2-NEXT:	.ascii	"branch-hinting"
; C2-NEXT:	.section	.text.test2,"",@
; C2-NEXT:	.section	.custom_section.metadata.code.branch_hint,"",@
; C2-NEXT:	.int8	1
; C2-DAG:	.uleb128 test2@FUNCINDEX
; C2-DAG-NEXT:	.int8	1
; C2-DAG-NEXT:	.uleb128 .Ltmp2
; C2-DAG-NEXT:	.int8	1
; C2-DAG-NEXT:	.int8	0
; C2:	.section	.text.test2,"",@

; C3:	.section	.custom_section.target_features,"",@
; C3-NEXT:	.int8	1
; C3-NEXT:	.int8	43
; C3-NEXT:	.int8	14
; C3-NEXT:	.ascii	"branch-hinting"
; C3-NEXT:	.section	.text.test2,"",@
; C3-NEXT:	.section	.custom_section.metadata.code.branch_hint,"",@
; C3-NEXT:	.int8	2
; C3-DAG:	.uleb128 test0@FUNCINDEX
; C3-DAG-NEXT:	.int8	1
; C3-DAG-NEXT:	.uleb128 .Ltmp0
; C3-DAG-NEXT:	.int8	1
; C3-DAG-NEXT:	.int8	1
; C3-DAG:	.uleb128 test2@FUNCINDEX
; C3-DAG-NEXT:	.int8	1
; C3-DAG-NEXT:	.uleb128 .Ltmp2
; C3-DAG-NEXT:	.int8	1
; C3-DAG-NEXT:	.int8	0
; C3:	.section	.text.test2,"",@

; YAML-CHECK:          - Type:            CUSTOM
; YAML-CHECK-NEXT:       Relocations:
; YAML-CHECK-NEXT:         - Type:            R_WASM_FUNCTION_INDEX_LEB
; YAML-CHECK-NEXT:           Index:           0
; YAML-CHECK-NEXT:           Offset:          0x1
; YAML-CHECK-NEXT:         - Type:            R_WASM_FUNCTION_INDEX_LEB
; YAML-CHECK-NEXT:           Index:           2
; YAML-CHECK-NEXT:           Offset:          0xE
; YAML-CHECK-NEXT:       Name:            metadata.code.branch_hint
; YAML-CHECK-NEXT:       Entries:
; YAML-CHECK-NEXT:         - FuncIdx:         0
; YAML-CHECK-NEXT:           Hints:
; YAML-CHECK-NEXT:             - Offset:          5
; YAML-CHECK-NEXT:               Size:            1
; YAML-CHECK-NEXT:               Data:            LIKELY
; YAML-CHECK-NEXT:         - FuncIdx:         2
; YAML-CHECK-NEXT:           Hints:
; YAML-CHECK-NEXT:             - Offset:          5
; YAML-CHECK-NEXT:               Size:            1
; YAML-CHECK-NEXT:               Data:            UNLIKELY

; YAML-CHECK:          - Type:            CUSTOM
; YAML-CHECK-NEXT:       Name:            linking

; YAML-CHECK:          - Type:            CUSTOM
; YAML-CHECK-NEXT:       Name:            target_features
; YAML-CHECK-NEXT:       Features:
; YAML-CHECK-NEXT:         - Prefix:          USED
; YAML-CHECK-NEXT:           Name:            branch-hinting

#--- 1.ll
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

#--- 1.S
# Assembly generated based on 1.ll and
# `llc -mcpu=mvp -filetype=asm %s -mattr=+branch-hinting -wasm-branch-prob-high=.75 -wasm-branch-prob-low=0 -o 1.S`
        .file   "1.ll"
        .functype       test0 (i32) -> (i32)
        .functype       test1 (i32) -> (i32)
        .functype       test2 (i32) -> (i32)
        .section        .text.test0,"",@
        .globl  test0                           # -- Begin function test0
        .type   test0,@function
test0:                                  # @test0
        .functype       test0 (i32) -> (i32)
# %bb.0:                                # %entry
        block
        local.get       0
.Ltmp0:
        br_if           0                               # 0: down to label0
# %bb.1:                                # %if_then
        i32.const       1
        return
.LBB0_2:                                # %if_else
        end_block                               # label0:
        i32.const       0
                                        # fallthrough-return
        end_function
                                        # -- End function
        .section        .text.test1,"",@
        .globl  test1                           # -- Begin function test1
        .type   test1,@function
test1:                                  # @test1
        .functype       test1 (i32) -> (i32)
# %bb.0:                                # %entry
        block
        local.get       0
.Ltmp1:
        br_if           0                               # 0: down to label1
# %bb.1:                                # %if_then
        i32.const       1
        return
.LBB1_2:                                # %if_else
        end_block                               # label1:
        i32.const       0
                                        # fallthrough-return
        end_function
                                        # -- End function
        .section        .text.test2,"",@
        .globl  test2                           # -- Begin function test2
        .type   test2,@function
test2:                                  # @test2
        .functype       test2 (i32) -> (i32)
# %bb.0:                                # %entry
        block
        local.get       0
.Ltmp2:
        br_if           0                               # 0: down to label2
# %bb.1:                                # %if_then
        i32.const       1
        return
.LBB2_2:                                # %if_else
        end_block                               # label2:
        i32.const       0
                                        # fallthrough-return
        end_function
                                        # -- End function
        .section        .custom_section.target_features,"",@
        .int8   1
        .int8   43
        .int8   14
        .ascii  "branch-hinting"
        .section        .text.test2,"",@
        .section        .custom_section.metadata.code.branch_hint,"",@
        .int8   2
        .uleb128 test0@FUNCINDEX
        .int8   1
        .uleb128 .Ltmp0
        .int8   1
        .int8   1
        .uleb128 test2@FUNCINDEX
        .int8   1
        .uleb128 .Ltmp2
        .int8   1
        .int8   0
        .section        .text.test2,"",@
