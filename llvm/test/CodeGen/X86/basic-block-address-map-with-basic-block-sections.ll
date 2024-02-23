;; Check that the -basic-block-address-map option works when used along with -basic-block-sections.
;; Let a function with 4 basic blocks get split into 2 sections.
; RUN: echo '!_Z3bazb' > %t
; RUN: echo '!!0 2' >> %t
; RUN: llc < %s -mtriple=x86_64 -basic-block-address-map -basic-block-sections=%t | FileCheck %s

define void @_Z3bazb(i1 zeroext) personality ptr @__gxx_personality_v0 {
  br i1 %0, label %2, label %7

2:
  %3 = invoke i32 @_Z3barv()
          to label %7 unwind label %5
  br label %9

5:
  landingpad { ptr, i32 }
          catch ptr null
  br label %9

7:
  %8 = call i32 @_Z3foov()
  br label %9

9:
  ret void
}

declare i32 @_Z3barv() #1

declare i32 @_Z3foov() #1

declare i32 @__gxx_personality_v0(...)

; CHECK:		.text
; CHECK-LABEL:	_Z3bazb:
; CHECK-LABEL:	.Lfunc_begin0:
; CHECK-LABEL:	.LBB_END0_0:
; CHECK-LABEL:	.LBB0_1:
; CHECK-LABEL:	.LBB_END0_1:
; CHECK:		.section .text.split._Z3bazb,"ax",@progbits
; CHECK-LABEL:	_Z3bazb.cold:
; CHECK-LABEL:	.LBB_END0_2:
; CHECK-LABEL:	.LBB0_3:
; CHECK-LABEL:	.LBB_END0_3:
; CHECK-LABEL:	.Lfunc_end0:

; CHECK:		.section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text.hot._Z3bazb
; CHECK-NEXT:   .byte   2               # version
; CHECK-NEXT:   .byte   8               # feature
; CHECK-NEXT:   .byte   2               # number of basic block ranges
; CHECK-NEXT:	.quad	.Lfunc_begin0   # base address
; CHECK-NEXT:	.byte	2               # number of basic blocks
; CHECK-NEXT:	.byte	0               # BB id
; CHECK-NEXT:	.uleb128 .Lfunc_begin0-.Lfunc_begin0
; CHECK-NEXT:	.uleb128 .LBB_END0_0-.Lfunc_begin0
; CHECK-NEXT:	.byte	0
; CHECK-NEXT:	.byte	2               # BB id
; CHECK-NEXT:	.uleb128 .LBB0_1-.LBB_END0_0
; CHECK-NEXT:	.uleb128 .LBB_END0_1-.LBB0_1
; CHECK-NEXT:	.byte	5
; CHECK-NEXT:	.quad	_Z3bazb.cold    # base address
; CHECK-NEXT:	.byte	2               # number of basic blocks
; CHECK-NEXT:	.byte	1               # BB id
; CHECK-NEXT:	.uleb128 _Z3bazb.cold-_Z3bazb.cold
; CHECK-NEXT:	.uleb128 .LBB_END0_2-_Z3bazb.cold
; CHECK-NEXT:	.byte	8
; CHECK-NEXT:	.byte	3               # BB id
; CHECK-NEXT:	.uleb128 .LBB0_3-.LBB_END0_2
; CHECK-NEXT:	.uleb128 .LBB_END0_3-.LBB0_3
; CHECK-NEXT:	.byte	1

