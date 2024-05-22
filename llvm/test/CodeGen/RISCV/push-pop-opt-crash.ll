; RUN: llc -mattr=+zcmp -verify-machineinstrs  \
; RUN: -mtriple=riscv32 -target-abi ilp32 < %s \
; RUN: | FileCheck %s -check-prefixes=RV32IZCMP
; RUN: llc -mattr=+zcmp -verify-machineinstrs  \
; RUN: -mtriple=riscv64 -target-abi ilp64 < %s \
; RUN: | FileCheck %s -check-prefixes=RV64IZCMP

; This source code exposed a crash in the RISC-V Zcmp Push/Pop optimization
; pass. The root cause was: Not doing a bounds check before using a returned
; iterator.

declare dso_local void @f1() local_unnamed_addr
declare dso_local void @f2() local_unnamed_addr
define  dso_local void @f0() local_unnamed_addr {
; RV32IZCMP-LABEL: f0:
; RV32IZCMP: 	.cfi_startproc
; RV32IZCMP-NEXT: # %bb.0:                                # %entry
; RV32IZCMP-NEXT: 	bnez	zero, .LBB0_2
; RV32IZCMP-NEXT: # %bb.1:                                # %if.T
; RV32IZCMP-NEXT: 	cm.push	{ra}, -16
; RV32IZCMP-NEXT: 	.cfi_def_cfa_offset 16
; RV32IZCMP-NEXT: 	.cfi_offset ra, -4
; RV32IZCMP-NEXT: 	call	f1
; RV32IZCMP-NEXT: 	cm.pop	{ra}, 16
; RV32IZCMP-NEXT: .LBB0_2:                                # %if.F
; RV32IZCMP-NEXT: 	tail	f2
; RV32IZCMP-NEXT: .Lfunc_end0:

; RV64IZCMP-LABEL: f0:
; RV64IZCMP: 	.cfi_startproc
; RV64IZCMP-NEXT: # %bb.0:                                # %entry
; RV64IZCMP-NEXT: 	bnez	zero, .LBB0_2
; RV64IZCMP-NEXT: # %bb.1:                                # %if.T
; RV64IZCMP-NEXT: 	cm.push	{ra}, -16
; RV64IZCMP-NEXT: 	.cfi_def_cfa_offset 16
; RV64IZCMP-NEXT: 	.cfi_offset ra, -8
; RV64IZCMP-NEXT: 	call	f1
; RV64IZCMP-NEXT: 	cm.pop	{ra}, 16
; RV64IZCMP-NEXT: .LBB0_2:                                # %if.F
; RV64IZCMP-NEXT: 	tail	f2
; RV64IZCMP-NEXT: .Lfunc_end0:
entry:
  br i1 poison, label %if.T, label %if.F

if.T:
  tail call void @f1()
  br label %if.F

if.F:
  tail call void @f2()
  ret void
}
