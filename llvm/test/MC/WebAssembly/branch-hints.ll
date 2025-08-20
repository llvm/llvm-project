# RUN: rm -rf %t; split-file %s %t

; check ll -> asm
; RUN: llc -mcpu=mvp -filetype=asm %t/1.ll -mattr=+branch-hinting -o - | FileCheck --check-prefixes=ASM-CHECK %s
; RUN: llc -mcpu=mvp -filetype=asm %t/1.ll -mattr=-branch-hinting -o - | FileCheck --check-prefixes=ASM-NCHECK %s
; RUN: llc -mcpu=mvp -filetype=asm %t/1.ll -o - | FileCheck --check-prefixes=ASM-NCHECK %s

; check asm -> obj -> yaml
; RUN: llvm-mc -mcpu=mvp -triple=wasm32-unknown-unknown -filetype=obj %t/1-bh.S -o - | obj2yaml | FileCheck --check-prefixes=YAML-CHECK %s
; RUN: llvm-mc -mcpu=mvp -triple=wasm32-unknown-unknown -filetype=obj %t/1-no-bh.S -o - | obj2yaml | FileCheck --check-prefixes=YAML-NCHECK %s

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
; ASM-CHECK-NEXT:   .int8   2
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
; YAML-CHECK-NEXT:         Offset:          0x1
; YAML-CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; YAML-CHECK-NEXT:         Index:           1
; YAML-CHECK-NEXT:         Offset:          0xD
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

; YAML-CHECK:        - Type:            CUSTOM
; YAML-CHECK-NEXT:     Name:            target_features
; YAML-CHECK-NEXT:     Features:
; YAML-CHECK-NEXT:       - Prefix:          USED
; YAML-CHECK-NEXT:         Name:            branch-hinting

; YAML-NCHECK-NOT:     Name:            metadata.code.branch_hint
; YAML-NCHECK-NOT:     Name:            branch-hinting

#--- 1.ll
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

#--- 1-no-bh.S
# Assembly generated based on 1.ll and
# `llc -mcpu=mvp -filetype=asm ./1.ll -o 1bh.S -mattr=+branch-hinting`
# `llc -mcpu=mvp -filetype=asm ./1.ll -o 1-no-bh.S -mattr=-branch-hinting`
	.file	"1.ll"
	.functype	test_unlikely_likely_branch (i32) -> (i32)
	.functype	test_likely_branch (i32) -> (i32)
	.section	.text.test_unlikely_likely_branch,"",@
	.globl	test_unlikely_likely_branch     # -- Begin function test_unlikely_likely_branch
	.type	test_unlikely_likely_branch,@function
test_unlikely_likely_branch:            # @test_unlikely_likely_branch
	.functype	test_unlikely_likely_branch (i32) -> (i32)
# %bb.0:                                # %entry
	block
	block
	local.get	0
	br_if   	0                               # 0: down to label1
# %bb.1:                                # %if.then
	local.get	0
	i32.const	1
	i32.ne
	br_if   	1                               # 1: down to label0
.LBB0_2:                                # %ret1
	end_block                               # label1:
	i32.const	2
	return
.LBB0_3:                                # %ret2
	end_block                               # label0:
	i32.const	1
                                        # fallthrough-return
	end_function
                                        # -- End function
	.section	.text.test_likely_branch,"",@
	.globl	test_likely_branch              # -- Begin function test_likely_branch
	.type	test_likely_branch,@function
test_likely_branch:                     # @test_likely_branch
	.functype	test_likely_branch (i32) -> (i32)
# %bb.0:                                # %entry
	block
	local.get	0
	br_if   	0                               # 0: down to label2
# %bb.1:                                # %if.then
	i32.const	1
	return
.LBB1_2:                                # %if.else
	end_block                               # label2:
	i32.const	2
                                        # fallthrough-return
	end_function
                                        # -- End function
#--- 1-bh.S
	.file	"1.ll"
	.functype	test_unlikely_likely_branch (i32) -> (i32)
	.functype	test_likely_branch (i32) -> (i32)
	.section	.text.test_unlikely_likely_branch,"",@
	.globl	test_unlikely_likely_branch     # -- Begin function test_unlikely_likely_branch
	.type	test_unlikely_likely_branch,@function
test_unlikely_likely_branch:            # @test_unlikely_likely_branch
	.functype	test_unlikely_likely_branch (i32) -> (i32)
# %bb.0:                                # %entry
	block
	block
	local.get	0
.Ltmp0:
	br_if   	0                               # 0: down to label1
# %bb.1:                                # %if.then
	local.get	0
	i32.const	1
	i32.ne
.Ltmp1:
	br_if   	1                               # 1: down to label0
.LBB0_2:                                # %ret1
	end_block                               # label1:
	i32.const	2
	return
.LBB0_3:                                # %ret2
	end_block                               # label0:
	i32.const	1
                                        # fallthrough-return
	end_function
                                        # -- End function
	.section	.text.test_likely_branch,"",@
	.globl	test_likely_branch              # -- Begin function test_likely_branch
	.type	test_likely_branch,@function
test_likely_branch:                     # @test_likely_branch
	.functype	test_likely_branch (i32) -> (i32)
# %bb.0:                                # %entry
	block
	local.get	0
.Ltmp2:
	br_if   	0                               # 0: down to label2
# %bb.1:                                # %if.then
	i32.const	1
	return
.LBB1_2:                                # %if.else
	end_block                               # label2:
	i32.const	2
                                        # fallthrough-return
	end_function
                                        # -- End function
	.section	.custom_section.target_features,"",@
	.int8	1
	.int8	43
	.int8	14
	.ascii	"branch-hinting"
	.section	.text.test_likely_branch,"",@
	.section	.custom_section.metadata.code.branch_hint,"",@
	.int8	2
	.uleb128 test_unlikely_likely_branch@FUNCINDEX
	.int8	2
	.uleb128 .Ltmp0@DEBUGREF
	.int8	1
	.int8	0
	.uleb128 .Ltmp1@DEBUGREF
	.int8	1
	.int8	1
	.uleb128 test_likely_branch@FUNCINDEX
	.int8	1
	.uleb128 .Ltmp2@DEBUGREF
	.int8	1
	.int8	1
	.section	.text.test_likely_branch,"",@
