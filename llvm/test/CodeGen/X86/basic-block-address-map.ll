; Check the basic block sections labels option
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map | FileCheck %s --check-prefixes=CHECK,UNIQ
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=false -basic-block-address-map | FileCheck %s --check-prefixes=CHECK,NOUNIQ
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -split-machine-functions | FileCheck %s --check-prefixes=CHECK,UNIQ

define void @_Z3bazb(i1 zeroext, i1 zeroext) personality ptr @__gxx_personality_v0 {
  br i1 %0, label %3, label %8

3:
  %4 = invoke i32 @_Z3barv()
          to label %8 unwind label %6
  br label %10

6:
  landingpad { ptr, i32 }
          catch ptr null
  br label %12

8:
  %9 = call i32 @_Z3foov()
  br i1 %1, label %12, label %10

10:
  %11 = select i1 %1, ptr blockaddress(@_Z3bazb, %3), ptr blockaddress(@_Z3bazb, %12) ; <ptr> [#uses=1]
  indirectbr ptr %11, [label %3, label %12]

12:
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
; CHECK-NEXT:   .byte   2		# version
; CHECK-NEXT:   .byte   0		# feature
; CHECK-NEXT:	.quad	.Lfunc_begin0	# function address
; CHECK-NEXT:	.byte	6		# number of basic blocks
; CHECK-NEXT:   .byte	0		# BB id
; CHECK-NEXT:	.uleb128 .Lfunc_begin0-.Lfunc_begin0
; CHECK-NEXT:	.uleb128 .LBB_END0_0-.Lfunc_begin0
; CHECK-NEXT:	.byte	8
; CHECK-NEXT:   .byte	1		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_1-.LBB_END0_0
; CHECK-NEXT:	.uleb128 .LBB_END0_1-.LBB0_1
; CHECK-NEXT:	.byte	8
; CHECK-NEXT:   .byte	3		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_2-.LBB_END0_1
; CHECK-NEXT:	.uleb128 .LBB_END0_2-.LBB0_2
; CHECK-NEXT:	.byte	8
; CHECK-NEXT:   .byte	4		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_3-.LBB_END0_2
; CHECK-NEXT:	.uleb128 .LBB_END0_3-.LBB0_3
; CHECK-NEXT:	.byte	16
; CHECK-NEXT:   .byte	5		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_4-.LBB_END0_3
; CHECK-NEXT:	.uleb128 .LBB_END0_4-.LBB0_4
; CHECK-NEXT:	.byte	1
; CHECK-NEXT:   .byte	2		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_5-.LBB_END0_4
; CHECK-NEXT:	.uleb128 .LBB_END0_5-.LBB0_5
; CHECK-NEXT:	.byte	5
