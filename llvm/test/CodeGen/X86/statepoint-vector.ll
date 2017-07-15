; RUN: llc -verify-machineinstrs -stack-symbol-ordering=0 -mcpu=nehalem -debug-only=stackmaps < %s | FileCheck %s
; REQUIRES: asserts

target triple = "x86_64-pc-linux-gnu"

; Can we lower a single vector?
define <2 x i8 addrspace(1)*> @test(<2 x i8 addrspace(1)*> %obj) gc "statepoint-example" {
entry:
; CHECK-LABEL: @test
; CHECK: subq	$24, %rsp
; CHECK: movaps	%xmm0, (%rsp)
; CHECK: callq	do_safepoint
; CHECK: movaps	(%rsp), %xmm0
; CHECK: addq	$24, %rsp
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, <2 x i8 addrspace(1)*> %obj)
  %obj.relocated = call coldcc <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %safepoint_token, i32 7, i32 7) ; (%obj, %obj)
  ret <2 x i8 addrspace(1)*> %obj.relocated
}

; Can we lower the base, derived pairs if both are vectors?
define <2 x i8 addrspace(1)*> @test2(<2 x i8 addrspace(1)*> %obj, i64 %offset) gc "statepoint-example" {
entry:
; CHECK-LABEL: @test2
; CHECK: subq	$40, %rsp
; CHECK: movq	%rdi, %xmm1
; CHECK: pshufd	$68, %xmm1, %xmm1       # xmm1 = xmm1[0,1,0,1]
; CHECK: paddq	%xmm0, %xmm1
; CHECK: movdqa	%xmm0, 16(%rsp)
; CHECK: movdqa	%xmm1, (%rsp)
; CHECK: callq	do_safepoint
; CHECK: movaps	(%rsp), %xmm0
; CHECK: addq	$40, %rsp
  %derived = getelementptr i8, <2 x i8 addrspace(1)*> %obj, i64 %offset
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, <2 x i8 addrspace(1)*> %obj, <2 x i8 addrspace(1)*> %derived)
  %derived.relocated = call coldcc <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %safepoint_token, i32 7, i32 8) ; (%obj, %derived)
  ret <2 x i8 addrspace(1)*> %derived.relocated
}

; Originally, this was just a variant of @test2 above, but it ends up 
; covering a bunch of interesting missed optimizations.  Specifically:
; - We waste a stack slot for a value that a backend transform pass
;   CSEd to another spilled one.
; - We don't remove the testb even though it serves no purpose
; - We could in principal reuse the argument memory (%rsi) and do away
;   with stack slots entirely.
define <2 x i64 addrspace(1)*> @test3(i1 %cnd, <2 x i64 addrspace(1)*>* %ptr) gc "statepoint-example" {
entry:
; CHECK-LABEL: @test3
; CHECK: subq	$40, %rsp
; CHECK: testb	$1, %dil
; CHECK: movaps	(%rsi), %xmm0
; CHECK-DAG: movaps	%xmm0, (%rsp)
; CHECK-DAG: movaps	%xmm0, 16(%rsp)
; CHECK: callq	do_safepoint
; CHECK: movaps	(%rsp), %xmm0
; CHECK: addq	$40, %rsp
  br i1 %cnd, label %taken, label %untaken

taken:                                            ; preds = %entry
  %obja = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge

untaken:                                          ; preds = %entry
  %objb = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge

merge:                                            ; preds = %untaken, %taken
  %obj.base = phi <2 x i64 addrspace(1)*> [ %obja, %taken ], [ %objb, %untaken ]
  %obj = phi <2 x i64 addrspace(1)*> [ %obja, %taken ], [ %objb, %untaken ]
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, <2 x i64 addrspace(1)*> %obj, <2 x i64 addrspace(1)*> %obj.base)
  %obj.relocated = call coldcc <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %safepoint_token, i32 8, i32 7) ; (%obj.base, %obj)
  %obj.relocated.casted = bitcast <2 x i8 addrspace(1)*> %obj.relocated to <2 x i64 addrspace(1)*>
  %obj.base.relocated = call coldcc <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %safepoint_token, i32 8, i32 8) ; (%obj.base, %obj.base)
  %obj.base.relocated.casted = bitcast <2 x i8 addrspace(1)*> %obj.base.relocated to <2 x i64 addrspace(1)*>
  ret <2 x i64 addrspace(1)*> %obj.relocated.casted
}

; Can we handle vector constants?  At the moment, we don't appear to actually
; get selection dag nodes for these.
define <2 x i8 addrspace(1)*> @test4() gc "statepoint-example" {
entry:
; CHECK-LABEL: @test4
; CHECK: subq	$24, %rsp
; CHECK: xorps %xmm0, %xmm0
; CHECK: movaps	%xmm0, (%rsp)
; CHECK: callq	do_safepoint
; CHECK: movaps	(%rsp), %xmm0
; CHECK: addq	$24, %rsp
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, <2 x i8 addrspace(1)*> zeroinitializer)
  %obj.relocated = call coldcc <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %safepoint_token, i32 7, i32 7) ; (%obj, %obj)
  ret <2 x i8 addrspace(1)*> %obj.relocated
}

; Check that we can lower a constant typed as i128 correctly.  Note that the
; actual value is representable in 64 bits.  We don't have a representation 
; of larger than 64 bit constant in the StackMap format.
define void @test5() gc "statepoint-example" {
entry:
; CHECK-LABEL: @test5
; CHECK: push
; CHECK: callq	do_safepoint
; CHECK: pop
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 1, i128 0)
  ret void
}

; CHECK: __LLVM_StackMaps:

; CHECK: .Ltmp0-test
; Check for the two spill slots
; Stack Maps: 		Loc 3: Indirect 7+0	[encoding: .byte 3, .byte 0, .short 16, .short 7, .short 0, .int 0]
; Stack Maps: 		Loc 4: Indirect 7+0	[encoding: .byte 3, .byte 0, .short 16, .short 7, .short 0, .int 0]
; CHECK: .byte	3
; CHECK: .byte	0
; CHECK: .short 16
; CHECK: .short	7
; CHECK: .short	0
; CHECK: .long	0
; CHECK: .byte	3
; CHECK: .byte	0
; CHECK: .short 16
; CHECK: .short	7
; CHECK: .short	0
; CHECK: .long	0

; CHECK: .Ltmp1-test2
; Check for the two spill slots
; Stack Maps: 		Loc 3: Indirect 7+16	[encoding: .byte 3, .byte 0, .short 16, .short 7, .short 0, .int 16]
; Stack Maps: 		Loc 4: Indirect 7+0	[encoding: .byte 3, .byte 0, .short 16, .short 7, .short 0, .int 0]
; CHECK: .byte	3
; CHECK: .byte	0
; CHECK: .short 16
; CHECK: .short	7
; CHECK: .short	0
; CHECK: .long	16
; CHECK: .byte	3
; CHECK: .byte	0
; CHECK: .short 16
; CHECK: .short	7
; CHECK: .short	0
; CHECK: .long	0

; CHECK: .Ltmp2-test3
; Check for the four spill slots
; Stack Maps: 		Loc 3: Indirect 7+16	[encoding: .byte 3, .byte 0, .short 16, .short 7, .short 0, .int 16]
; Stack Maps: 		Loc 4: Indirect 7+16	[encoding: .byte 3, .byte 0, .short 16, .short 7, .short 0, .int 16]
; Stack Maps: 		Loc 5: Indirect 7+16	[encoding: .byte 3, .byte 0, .short 16, .short 7, .short 0, .int 16]
; Stack Maps: 		Loc 6: Indirect 7+0	[encoding: .byte 3, .byte 0, .short 16, .short 7, .short 0, .int 0]
; CHECK: .byte	3
; CHECK: .byte	0
; CHECK: .short 16
; CHECK: .short	7
; CHECK: .short	0
; CHECK: .long	16
; CHECK: .byte	3
; CHECK: .byte	 0
; CHECK: .short 16
; CHECK: .short	7
; CHECK: .short	0
; CHECK: .long	16
; CHECK: .byte	3
; CHECK: .byte	 0
; CHECK: .short 16
; CHECK: .short	7
; CHECK: .short	0
; CHECK: .long	16
; CHECK: .byte	3
; CHECK: .byte	 0
; CHECK: .short 16
; CHECK: .short	7
; CHECK: .short	0
; CHECK: .long	0

declare void @do_safepoint()

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32)
declare <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token, i32, i32)
