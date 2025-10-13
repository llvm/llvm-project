; Verify PGO analysis map features with basic block sections profile.
;
; RUN: echo 'v1' > %t
; RUN: echo 'f foo' >> %t
; RUN: echo 'g 0:1000,1:800,2:200 1:800,3:800 2:200,3:200 3:1000' >> %t
; RUN: echo 'c 0 1 2' >> %t
;
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t -basic-block-address-map -pgo-analysis-map=all | FileCheck %s

define void @foo() nounwind !prof !0 {
entry:
  br label %bb1

bb1:
  br i1 undef, label %bb2, label %bb3, !prof !1

bb2:
  br label %bb3

bb3:
  ret void
}

!0 = !{!"function_entry_count", i64 1500}
!1 = !{!"branch_weights", i32 1200, i32 300}

; CHECK: .section .text.foo,"ax",@progbits
; CHECK-LABEL: foo:
; CHECK: .LBB_END0_0:
; CHECK-LABEL: .LBB0_1:
; CHECK: .LBB_END0_1:
; CHECK-LABEL: .LBB0_2:
; CHECK: .LBB_END0_2:
; CHECK-LABEL: foo.cold:
; CHECK: .LBB_END0_3:

; CHECK: 	.section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text.foo
; CHECK-NEXT:	.byte	3		# version
; CHECK-NEXT:	.byte	15		# feature
; CHECK:	.quad	.Lfunc_begin0	# base address
; CHECK:	.byte	0		# BB id
; CHECK:	.byte	1		# BB id
; CHECK:	.byte	2		# BB id
; CHECK:	.byte	3		# BB id

; PGO Analysis Map
; CHECK:	.ascii	"\350\007"	# function entry count
; CHECK-NEXT:	.ascii	"\350\007"	# basic block frequency (from BBSPR)
; CHECK-NEXT:	.ascii	"\240\006"	# basic block frequency (from BBSPR)
; CHECK-NEXT:	.ascii	"\310\001"	# basic block frequency (from BBSPR)
; CHECK-NEXT:	.ascii	"\350\007"	# basic block frequency (from BBSPR)
; CHECK-NEXT:	.byte	1		# basic block successor count
; CHECK-NEXT:	.byte	1		# successor BB ID
; CHECK-NEXT:	.ascii	"\240\006"	# successor branch probability (from BBSPR)
; CHECK-NEXT:	.byte	2		# basic block successor count
; CHECK-NEXT:	.byte	2		# successor BB ID
; CHECK-NEXT:	.byte	0		# successor branch probability (from BBSPR)
; CHECK-NEXT:	.byte	3		# successor BB ID
; CHECK-NEXT:	.ascii	"\240\006"	# successor branch probability (from BBSPR)
; CHECK-NEXT:	.byte	1		# basic block successor count
; CHECK-NEXT:	.byte	3		# successor BB ID
; CHECK-NEXT:	.ascii	"\310\001"	# successor branch probability (from BBSPR)
; CHECK-NEXT:	.byte	0		# basic block successor count

