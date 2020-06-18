; RUN: llc < %s -verify-machineinstrs -stack-symbol-ordering=0 -mtriple="x86_64-pc-linux-gnu" | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -stack-symbol-ordering=0 -mtriple="x86_64-pc-unknown-elf" | FileCheck %s

; This test is a sanity check to ensure statepoints are generating StackMap
; sections correctly.  This is not intended to be a rigorous test of the 
; StackMap format (see the stackmap tests for that).

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"

declare zeroext i1 @return_i1()

define i1 @test(i32 addrspace(1)* %ptr_base, i32 %arg)
  gc "statepoint-example" {
; CHECK-LABEL: test:
; Do we see two spills for the local values and the store to the
; alloca?
; CHECK: subq	$40, %rsp
; CHECK: movq	$0,   24(%rsp)
; CHECK: movq	%rdi, 16(%rsp)
; CHECK: movq	%rax, 8(%rsp)
; CHECK: callq return_i1
; CHECK: addq	$40, %rsp
; CHECK: retq
entry:
  %metadata1 = alloca i32 addrspace(1)*, i32 2, align 8
  store i32 addrspace(1)* null, i32 addrspace(1)** %metadata1
  %ptr_derived = getelementptr i32, i32 addrspace(1)* %ptr_base, i32 %arg
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %ptr_base, i32 addrspace(1)* %ptr_derived, i32 addrspace(1)* null) ["deopt" (i32 addrspace(1)* %ptr_base, i32 addrspace(1)* null)]
  %call1 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  %a = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 7, i32 7)
  %b = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 7, i32 8)
  %c = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 9, i32 9)
; 
  ret i1 %call1
}

; This is similar to the previous test except that we have derived pointer as
; argument to the function. Despite that this can not happen after the
; RewriteSafepointForGC pass, lowering should be able to handle it anyway.
define i1 @test_derived_arg(i32 addrspace(1)* %ptr_base,
                            i32 addrspace(1)* %ptr_derived)
  gc "statepoint-example" {
; CHECK-LABEL: test_derived_arg
; Do we see two spills for the local values and the store to the
; alloca?
; CHECK: subq	$40, %rsp
; CHECK: movq	$0,   24(%rsp)
; CHECK: movq	%rdi, 16(%rsp)
; CHECK: movq	%rsi, 8(%rsp)
; CHECK: callq return_i1
; CHECK: addq	$40, %rsp
; CHECK: retq
entry:
  %metadata1 = alloca i32 addrspace(1)*, i32 2, align 8
  store i32 addrspace(1)* null, i32 addrspace(1)** %metadata1
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %ptr_base, i32 addrspace(1)* %ptr_derived, i32 addrspace(1)* null) ["deopt" (i32 addrspace(1)* %ptr_base, i32 addrspace(1)* null)]
  %call1 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  %a = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 7, i32 7)
  %b = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 7, i32 8)
  %c = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 9, i32 9)
; 
  ret i1 %call1
}

; Simple test case to check that we emit the ID field correctly
define i1 @test_id() gc "statepoint-example" {
; CHECK-LABEL: test_id
entry:
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 237, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0)
  %call1 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  ret i1 %call1
}

; This test checks that when SP is changed in the function
; (e.g. passing arguments on stack), the stack map entry
; takes this adjustment into account.
declare void @many_arg(i64, i64, i64, i64, i64, i64, i64, i64)

define i32 @test_spadj(i32 addrspace(1)* %p) gc "statepoint-example" {
  ; CHECK-LABEL: test_spadj
  ; CHECK: movq %rdi, (%rsp)
  ; CHECK: xorl %edi, %edi
  ; CHECK: xorl %esi, %esi
  ; CHECK: xorl %edx, %edx
  ; CHECK: xorl %ecx, %ecx
  ; CHECK: xorl %r8d, %r8d
  ; CHECK: xorl %r9d, %r9d
  ; CHECK: pushq $0
  ; CHECK: pushq $0
  ; CHECK: callq many_arg
  ; CHECK: addq $16, %rsp
  ; CHECK: movq (%rsp)
  %statepoint_token = call token (i64, i32, void (i64, i64, i64, i64, i64, i64, i64, i64)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidi64i64i64i64i64i64i64i64f(i64 0, i32 0, void (i64, i64, i64, i64, i64, i64, i64, i64)* @many_arg, i32 8, i32 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0, i32 0, i32 addrspace(1)* %p)
  %p.relocated = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %statepoint_token, i32 15, i32 15) ; (%p, %p)
  %ld = load i32, i32 addrspace(1)* %p.relocated
  ret i32 %ld
}

; Test that function arguments at fixed stack offset
; can be directly encoded in the stack map, without
; spilling.
%struct = type { i64, i64, i64 }

declare void @use(%struct*)

define void @test_fixed_arg(%struct* byval %x) gc "statepoint-example" {
; CHECK-LABEL: test_fixed_arg
; CHECK: pushq %rax
; CHECK: leaq 16(%rsp), %rdi
; Should not spill fixed stack address.
; CHECK-NOT: movq %rdi, (%rsp)
; CHECK: callq use
; CHECK: popq %rax
; CHECK: retq
entry:
  br label %bb

bb:                                               ; preds = %entry
  %statepoint_token = call token (i64, i32, void (%struct*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp0s_structsf(i64 0, i32 0, void (%struct*)* @use, i32 1, i32 0, %struct* %x, i32 0, i32 0) ["deopt" (%struct* %x)]
  ret void
}

declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidi64i64i64i64i64i64i64i64f(i64, i32, void (i64, i64, i64, i64, i64, i64, i64, i64)*, i32, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidp0s_structsf(i64, i32, void (%struct*)*, i32, i32, ...)
declare i1 @llvm.experimental.gc.result.i1(token)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32) #3

; CHECK-LABEL: .section .llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 5
; Num LargeConstants
; CHECK-NEXT:   .long 0
; Num Callsites
; CHECK-NEXT:   .long 5

; Functions and stack size
; CHECK-NEXT:   .quad test
; CHECK-NEXT:   .quad 40
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad test_derived_arg
; CHECK-NEXT:   .quad 40
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad test_id
; CHECK-NEXT:   .quad 8
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad test_spadj
; CHECK-NEXT:   .quad 8
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad test_fixed_arg
; CHECK-NEXT:   .quad 8
; CHECK-NEXT:   .quad 1

;
; test
;

; Statepoint ID
; CHECK-NEXT: .quad	0

; Callsites
; Constant arguments
; CHECK-NEXT: .long	.Ltmp0-test
; CHECK: .short	0
; CHECK: .short	11
; SmallConstant (0)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0
; SmallConstant (0)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0
; SmallConstant (2)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	2
; Indirect Spill Slot [RSP+0]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16
; SmallConstant  (0)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0
; SmallConstant  (0)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0
; SmallConstant  (0)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0
; Indirect Spill Slot [RSP+16]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16
; Indirect Spill Slot [RSP+8]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	8
; Indirect Spill Slot [RSP+16]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16
; Indirect Spill Slot [RSP+16]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16

; No Padding or LiveOuts
; CHECK: .short	0
; CHECK: .short	0
; CHECK: .p2align	3

;
; test_derived_arg

; Statepoint ID
; CHECK-NEXT: .quad	0

; Callsites
; Constant arguments
; CHECK-NEXT: .long	.Ltmp1-test_derived_arg
; CHECK: .short	0
; CHECK: .short	11
; SmallConstant (0)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0
; SmallConstant (2)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	2
; Indirect Spill Slot [RSP+0]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16
; SmallConstant  (0)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0
; SmallConstant  (0)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0
; SmallConstant  (0)
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0
; Indirect Spill Slot [RSP+16]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16
; Indirect Spill Slot [RSP+8]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	8
; Indirect Spill Slot [RSP+16]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16
; Indirect Spill Slot [RSP+16]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16

; No Padding or LiveOuts
; CHECK: .short	0
; CHECK: .short	0
; CHECK: .p2align	3

; Records for the test_id function:

; The Statepoint ID:
; CHECK-NEXT: .quad	237

; Instruction Offset
; CHECK-NEXT: .long	.Ltmp2-test_id

; Reserved:
; CHECK: .short	0

; NumLocations:
; CHECK: .short	3

; StkMapRecord[0]:
; SmallConstant(0):
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0

; StkMapRecord[1]:
; SmallConstant(0):
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0

; StkMapRecord[2]:
; SmallConstant(0):
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0

; No padding or LiveOuts
; CHECK: .short	0
; CHECK: .short	0
; CHECK: .p2align	3

;
; test_spadj

; Statepoint ID
; CHECK-NEXT: .quad	0

; Instruction Offset
; CHECK-NEXT: .long	.Ltmp3-test_spadj

; Reserved:
; CHECK: .short	0

; NumLocations:
; CHECK: .short	5

; StkMapRecord[0]:
; SmallConstant(0):
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0

; StkMapRecord[1]:
; SmallConstant(0):
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0

; StkMapRecord[2]:
; SmallConstant(0):
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0

; StkMapRecord[3]:
; Indirect Spill Slot [RSP+16]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16

; StkMapRecord[4]:
; Indirect Spill Slot [RSP+16]
; CHECK: .byte	3
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16

; No padding or LiveOuts
; CHECK: .short	0
; CHECK: .short	0
; CHECK: .p2align	3

;
; test_fixed_arg

; Statepoint ID
; CHECK-NEXT: .quad	0

; Instruction Offset
; CHECK-NEXT: .long	.Ltmp4-test_fixed_arg

; Reserved:
; CHECK: .short	0

; NumLocations:
; CHECK: .short	4

; StkMapRecord[0]:
; SmallConstant(0):
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0

; StkMapRecord[1]:
; SmallConstant(0):
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	0

; StkMapRecord[2]:
; SmallConstant(1):
; CHECK: .byte	4
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	0
; CHECK-NEXT:   .short  0
; CHECK: .long	1

; StkMapRecord[3]:
; Direct RSP+16
; CHECK: .byte	2
; CHECK-NEXT:   .byte   0
; CHECK: .short 8
; CHECK: .short	7
; CHECK-NEXT:   .short  0
; CHECK: .long	16

; No padding or LiveOuts
; CHECK: .short	0
; CHECK: .short	0
; CHECK: .p2align	3
