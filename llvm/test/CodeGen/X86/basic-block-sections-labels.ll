; Check the basic block sections labels option
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-sections=labels | FileCheck %s --check-prefixes=CHECK,UNIQ
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=false -basic-block-sections=labels | FileCheck %s --check-prefixes=CHECK,NOUNIQ
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-sections=labels -split-machine-functions | FileCheck %s --check-prefixes=CHECK,UNIQ

define void @_Z3bazb(i1 zeroext) personality i32 (...)* @__gxx_personality_v0 {
  br i1 %0, label %2, label %7

2:
  %3 = invoke i32 @_Z3barv()
          to label %7 unwind label %5
  br label %9

5:
  landingpad { i8*, i32 }
          catch i8* null
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

; UNIQ:			.section .text._Z3bazb,"ax",@progbits{{$}}
; NOUNIQ:		.section .text,"ax",@progbits,unique,1
; CHECK-LABEL:	_Z3bazb:
; CHECK-LABEL:	.Lfunc_begin0:
; CHECK-LABEL:	.LBB_END0_0:
; CHECK-LABEL:	.LBB0_1:
; CHECK-LABEL:	.LBB_END0_1:
; CHECK-LABEL:	.LBB0_2:
; CHECK-LABEL:	.LBB_END0_2:
; CHECK-LABEL:	.LBB0_3:
; CHECK-LABEL:	.LBB_END0_3:
; CHECK-LABEL:	.Lfunc_end0:

; UNIQ:			.section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text._Z3bazb{{$}}
;; Verify that with -unique-section-names=false, the unique id of the text section gets assigned to the llvm_bb_addr_map section.
; NOUNIQ:		.section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text,unique,1
; CHECK-NEXT:   .byte   1		# version
; CHECK-NEXT:   .byte   0		# feature
; CHECK-NEXT:	.quad	.Lfunc_begin0	# function address
; CHECK-NEXT:	.byte	4		# number of basic blocks
; CHECK-NEXT:	.uleb128 .Lfunc_begin0-.Lfunc_begin0
; CHECK-NEXT:	.uleb128 .LBB_END0_0-.Lfunc_begin0
; CHECK-NEXT:	.byte	8
; CHECK-NEXT:	.uleb128 .LBB0_1-.LBB_END0_0
; CHECK-NEXT:	.uleb128 .LBB_END0_1-.LBB0_1
; CHECK-NEXT:	.byte	8
; CHECK-NEXT:	.uleb128 .LBB0_2-.LBB_END0_1
; CHECK-NEXT:	.uleb128 .LBB_END0_2-.LBB0_2
; CHECK-NEXT:	.byte	1
; CHECK-NEXT:	.uleb128 .LBB0_3-.LBB_END0_2
; CHECK-NEXT:	.uleb128 .LBB_END0_3-.LBB0_3
; CHECK-NEXT:	.byte	5
