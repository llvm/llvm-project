; Verify PGO analysis map features with basic block sections profile.
;
; RUN: echo 'v1' > %t
; RUN: echo 'f foo' >> %t
; RUN: echo 'g 0:1000,1:800,2:200 1:800,3:800 2:200,3:200 3:1000' >> %t
; RUN: echo 'c 0 1 2' >> %t
;
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t -basic-block-address-map -pgo-analysis-map=all -pgo-analysis-map-emit-bb-sections-cfg | FileCheck %s

define void @foo(i1 %cond) nounwind !prof !0 {
entry:
  br label %bb1

bb1:
  br i1 %cond, label %bb2, label %bb3, !prof !1

bb2:
  br label %bb3

bb3:
  ret void
}

!0 = !{!"function_entry_count", i64 1500}
!1 = !{!"branch_weights", i32 1200, i32 300}

;; Verify that foo's PGO map contains both IRPGO and Propeller CFG profiles.

; CHECK: 	.section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text.foo
; CHECK-NEXT:	.byte	5		# version
; CHECK-NEXT:	.short	143		# feature
; CHECK:	.quad	.Lfunc_begin0	# base address
; CHECK:	.byte	0		# BB id
; CHECK:	.byte	1		# BB id
; CHECK:	.byte	2		# BB id
; CHECK:	.byte	3		# BB id

; CHECK:	.ascii	"\334\013"				# function entry count
; CHECK-NEXT:	.ascii	"\200\200\200\200\200\200\200 "		# basic block frequency
; CHECK-NEXT:	.ascii	"\350\007"				# basic block frequency (propeller)
; CHECK-NEXT:	.byte	1					# basic block successor count
; CHECK-NEXT:	.byte	1					# successor BB ID
; CHECK-NEXT:	.ascii	"\200\200\200\200\b"			# successor branch probability
; CHECK-NEXT:	.ascii	"\240\006"				# successor branch frequency (propeller)
; CHECK-NEXT:	.ascii	"\200\200\200\200\200\200\200 " 	# basic block frequency
; CHECK-NEXT:	.ascii	"\240\006"				# basic block frequency (propeller)
; CHECK-NEXT:	.byte	2					# basic block successor count
; CHECK-NEXT:	.byte	2					# successor BB ID
; CHECK-NEXT:	.ascii	"\200\200\200\200\004"			# successor branch probability
; CHECK-NEXT:	.byte	0					# successor branch frequency (propeller)
; CHECK-NEXT:	.byte	3					# successor BB ID
; CHECK-NEXT:	.ascii	"\200\200\200\200\004"			# successor branch probability
; CHECK-NEXT:	.ascii	"\240\006"				# successor branch frequency (propeller)
; CHECK-NEXT:	.ascii	"\200\200\200\200\200\200\200\020"	# basic block frequency
; CHECK-NEXT:	.ascii	"\310\001"				# basic block frequency (propeller)
; CHECK-NEXT:	.byte	1					# basic block successor count
; CHECK-NEXT:	.byte	3					# successor BB ID
; CHECK-NEXT:	.ascii	"\200\200\200\200\b"			# successor branch probability
; CHECK-NEXT:	.ascii	"\310\001"				# successor branch frequency (propeller)
; CHECK-NEXT:	.ascii	"\200\200\200\200\200\200\200 "		# basic block frequency
; CHECK-NEXT:	.ascii	"\350\007"				# basic block frequency (propeller)
; CHECK-NEXT:	.byte	0					# basic block successor count

define void @bar(i1 %cond) nounwind !prof !2 {
entry:
  br i1 %cond, label %bb1, label %bb2, !prof !3

bb1:
  ret void

bb2:
  ret void
}

!2 = !{!"function_entry_count", i64 80}
!3 = !{!"branch_weights", i32 2, i32 78}

;; Verify that the PGO map for bar only includes IRPGO data since it doesn't
;; have Propeller profile.

; CHECK: 	.section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text.bar
; CHECK-NEXT:	.byte	5		# version
; CHECK-NEXT:	.short	7		# feature
; CHECK:	.quad	.Lfunc_begin1	# function address
; CHECK:	.byte	0		# BB id
; CHECK:	.byte	1		# BB id
; CHECK:	.byte	2		# BB id

; CHECK:	.byte	80                              # function entry count
; CHECK-NEXT:	.ascii	"\200\200\200\200\200\200\200 " # basic block frequency
; CHECK-NEXT:	.byte	2                               # basic block successor count
; CHECK-NEXT:	.byte	1                               # successor BB ID
; CHECK-NEXT:	.ascii	"\200\200\200\200\004"          # successor branch probability
; CHECK-NEXT:	.byte	2                               # successor BB ID
; CHECK-NEXT:	.ascii	"\200\200\200\200\004"          # successor branch probability
; CHECK-NEXT:	.ascii	"\200\200\200\200\200\200\200\020" # basic block frequency
; CHECK-NEXT:	.byte	0                               # basic block successor count
; CHECK-NEXT:	.ascii	"\200\200\200\200\200\200\200\020" # basic block frequency
; CHECK-NEXT:	.byte	0                               # basic block successor count

