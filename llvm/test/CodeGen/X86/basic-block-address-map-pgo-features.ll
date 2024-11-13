; Check the basic block sections labels option
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map | FileCheck %s --check-prefixes=CHECK,BASIC,PGO-NONE
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -pgo-analysis-map=none | FileCheck %s --check-prefixes=CHECK,BASIC,PGO-NONE

;; Also verify this holds for all PGO features enabled
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -pgo-analysis-map=func-entry-count,bb-freq,br-prob | FileCheck %s --check-prefixes=CHECK,PGO-ALL,PGO-FEC,PGO-BBF,PGO-BRP
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -pgo-analysis-map=all | FileCheck %s --check-prefixes=CHECK,PGO-ALL,PGO-FEC,PGO-BBF,PGO-BRP

;; Also verify that pgo extension only includes the enabled feature
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -pgo-analysis-map=func-entry-count | FileCheck %s --check-prefixes=CHECK,PGO-FEC,FEC-ONLY
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -pgo-analysis-map=bb-freq | FileCheck %s --check-prefixes=CHECK,PGO-BBF,BBF-ONLY
; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -pgo-analysis-map=br-prob | FileCheck %s --check-prefixes=CHECK,PGO-BRP,BRP-ONLY

; RUN: llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -pgo-analysis-map=func-entry-count -basic-block-address-map-skip-bb-entries | FileCheck %s --check-prefixes=SKIP-BB-ENTRIES
; RUN: not llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -pgo-analysis-map=bb-freq -basic-block-address-map-skip-bb-entries 2>&1 | FileCheck %s --check-prefixes=SKIP-BB-ENTRIES-ERROR
; RUN: not llc < %s -mtriple=x86_64 -function-sections -unique-section-names=true -basic-block-address-map -pgo-analysis-map=br-prob -basic-block-address-map-skip-bb-entries 2>&1 | FileCheck %s --check-prefixes=SKIP-BB-ENTRIES-ERROR

;; Verify that we emit an error if we try and specify values in addition to all/none
; RUN: not llc < %s -mtriple=x86_64 -basic-block-address-map -pgo-analysis-map=all,bb-freq
; RUN: not llc < %s -mtriple=x86_64 -basic-block-address-map -pgo-analysis-map=none,bb-freq


define void @_Z3bazb(i1 zeroext, i1 zeroext) personality ptr @__gxx_personality_v0 !prof !0  {
  br i1 %0, label %3, label %8, !prof !1

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
  br i1 %1, label %12, label %10, !prof !2

10:
  %11 = select i1 %1, ptr blockaddress(@_Z3bazb, %3), ptr blockaddress(@_Z3bazb, %12) ; <ptr> [#uses=1]
  indirectbr ptr %11, [label %3, label %12], !prof !3

12:
  ret void
}

declare i32 @_Z3barv() #1

declare i32 @_Z3foov() #1

declare i32 @__gxx_personality_v0(...)

!0 = !{!"function_entry_count", i64 100}
!1 = !{!"branch_weights", i32 80, i32 20}
!2 = !{!"branch_weights", i32 70, i32 10}
!3 = !{!"branch_weights", i32 15, i32 5}

; CHECK:	.section .text._Z3bazb,"ax",@progbits{{$}}
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

; CHECK: 	.section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text._Z3bazb{{$}}
; CHECK-NEXT:	.byte	2		# version
; BASIC-NEXT:	.byte	0		# feature
; PGO-ALL-NEXT:	.byte	7		# feature
; FEC-ONLY-NEXT:.byte	1		# feature
; BBF-ONLY-NEXT:.byte	2		# feature
; BRP-ONLY-NEXT:.byte	4		# feature
; CHECK-NEXT:	.quad	.Lfunc_begin0	# function address
; CHECK-NEXT:	.byte	6		# number of basic blocks
; CHECK-NEXT:	.byte	0		# BB id
; CHECK-NEXT:	.uleb128 .Lfunc_begin0-.Lfunc_begin0
; CHECK-NEXT:	.uleb128 .LBB_END0_0-.Lfunc_begin0
; CHECK-NEXT:	.byte	8
; CHECK-NEXT:	.byte	1		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_1-.LBB_END0_0
; CHECK-NEXT:	.uleb128 .LBB_END0_1-.LBB0_1
; CHECK-NEXT:	.byte	8
; CHECK-NEXT:	.byte	3		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_2-.LBB_END0_1
; CHECK-NEXT:	.uleb128 .LBB_END0_2-.LBB0_2
; CHECK-NEXT:	.byte	8
; CHECK-NEXT:	.byte	5		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_3-.LBB_END0_2
; CHECK-NEXT:	.uleb128 .LBB_END0_3-.LBB0_3
; CHECK-NEXT:	.byte	1
; CHECK-NEXT:	.byte	4		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_4-.LBB_END0_3
; CHECK-NEXT:	.uleb128 .LBB_END0_4-.LBB0_4
; CHECK-NEXT:	.byte	16
; CHECK-NEXT:	.byte	2		# BB id
; CHECK-NEXT:	.uleb128 .LBB0_5-.LBB_END0_4
; CHECK-NEXT:	.uleb128 .LBB_END0_5-.LBB0_5
; CHECK-NEXT:	.byte	4

;; PGO Analysis Map
; PGO-NONE-NOT: .byte	100		# function entry count
; PGO-NONE-NOT: .ascii	"\271\235\376\332\245\200\356\017"	# basic block frequency
; PGO-NONE-NOT: .byte	2		# basic block successor count
; PGO-FEC-NEXT:	.byte	100		# function entry count
; PGO-BBF-NEXT:	.ascii	"\271\235\376\332\245\200\356\017"	# basic block frequency
; PGO-BRP-NEXT:	.byte	2		# basic block successor count
; PGO-BRP-NEXT:	.byte	1		# successor BB ID
; PGO-BRP-NEXT:	.ascii	"\346\314\231\263\006"	# successor branch probability
; PGO-BRP-NEXT:	.byte	3		# successor BB ID
; PGO-BRP-NEXT:	.ascii	"\232\263\346\314\001"	# successor branch probability
; PGO-BBF-NEXT:	.ascii	"\202\301\341\375\205\200\200\003"	# basic block frequency
; PGO-BRP-NEXT:	.byte	2		# basic block successor count
; PGO-BRP-NEXT:	.byte	3		# successor BB ID
; PGO-BRP-NEXT:	.ascii	"\200\360\377\377\007"	# successor branch probability
; PGO-BRP-NEXT:	.byte	2		# successor BB ID
; PGO-BRP-NEXT:	.ascii	"\200\020"	# successor branch probability
; PGO-BBF-NEXT:	.ascii	"\200\200\200\200\200\200\200 "	# basic block frequency
; PGO-BRP-NEXT:	.byte	2		# basic block successor count
; PGO-BRP-NEXT:	.byte	5		# successor BB ID
; PGO-BRP-NEXT:	.ascii	"\200\200\200\200\007"	# successor branch probability
; PGO-BRP-NEXT:	.byte 	4		# successor BB ID
; PGO-BRP-NEXT:	.ascii	"\200\200\200\200\001"	# successor branch probability
; PGO-BBF-NEXT:	.ascii	"\271\235\376\332\245\200\356\017"	# basic block frequency
; PGO-BRP-NEXT:	.byte	0		# basic block successor count
; PGO-BBF-NEXT:	.ascii	"\210\214\356\257\200\200\230\002"	# basic block frequency
; PGO-BRP-NEXT:	.byte	2		# basic block successor count
; PGO-BRP-NEXT:	.byte 	1		# successor BB ID
; PGO-BRP-NEXT:	.ascii	"\200\200\200\200\006"	# successor branch probability
; PGO-BRP-NEXT:	.byte	5		# successor BB ID
; PGO-BRP-NEXT:	.ascii	"\200\200\200\200\002"	# successor branch probability
; PGO-BBF-NEXT:	.ascii	"\235\323\243\200#"	# basic block frequency
; PGO-BRP-NEXT:	.byte	1		# basic block successor count
; PGO-BRP-NEXT:	.byte	5		# successor BB ID
; PGO-BRP-NEXT:	.ascii	"\200\200\200\200\b"	# successor branch probability

; SKIP-BB-ENTRIES:      .byte	17                              # feature
; SKIP-BB-ENTRIES-NEXT:	.quad	.Lfunc_begin0                   # function address
; SKIP-BB-ENTRIES-NEXT:	.byte	6                               # number of basic blocks
; SKIP-BB-ENTRIES-NEXT:	.byte	100                             # function entry count
; SKIP-BB-ENTRIES-NOT:  # BB id

; SKIP-BB-ENTRIES-ERROR: error: BB entries info is required for BBFreq and BrProb features
