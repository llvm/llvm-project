; RUN: llc < %s -mtriple="x86_64-pc-linux-gnu" | FileCheck %s

declare void @callee()

define void @f_0(<1024 x i64> %val) {
; CHECK:      .quad	2882400015
; CHECK-NEXT: .long	.Ltmp0-f_0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	4
; Constant(0)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0 
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Constant(0)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Constant(1)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	1
; Indirect
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8192
; CHECK-NEXT: .short	7
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Padding
; CHECK-NEXT: .p2align	3
  call void @callee() [ "deopt"(<1024 x i64> %val) ]
  ret void
}

define void @f_1(<1024 x ptr> %val) {
; CHECK:      .quad	2882400015
; CHECK-NEXT: .long	.Ltmp1-f_1
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	4
; Constant(0)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Constant(0)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Constant(1)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	1
; Indirect
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8192
; CHECK-NEXT: .short	7
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Padding
; CHECK-NEXT: .p2align	3
  call void @callee() [ "deopt"(<1024 x ptr> %val) ]
  ret void
}

define void @f_2(<99 x ptr> %val) {
; CHECK:      .quad	2882400015
; CHECK-NEXT: .long	.Ltmp2-f_2
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	4
; Constant(0)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Constant(0)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Constant(1)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	1
; Indirect
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	792
; CHECK-NEXT: .short	7
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; CHECK-NEXT: .p2align	3
  call void @callee() [ "deopt"(<99 x ptr> %val) ]
  ret void
}


define <400 x ptr addrspace(1)> @f_3(<400 x ptr addrspace(1)> %obj) gc "statepoint-example" {
; CHECK:      .quad	4242
; CHECK-NEXT: .long	.Ltmp3-f_3
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	5
; Constant(0)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Constant(0)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Constant(0)
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Indirect
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	3200
; CHECK-NEXT: .short	7
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Indirect
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .short	3200
; CHECK-NEXT: .short	7
; CHECK-NEXT: .short	0
; CHECK-NEXT: .long	0
; Padding
; CHECK-NEXT: .p2align	3
  %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 4242, i32 0, ptr elementtype(void ()) @do_safepoint, i32 0, i32 0, i32 0, i32 0) ["gc-live"(<400 x ptr addrspace(1)> %obj)]
  %obj.r = call coldcc <400 x ptr addrspace(1)> @llvm.experimental.gc.relocate.v400p1(token %tok, i32 0, i32 0)
  ret <400 x ptr addrspace(1)> %obj.r
}

declare void @do_safepoint()

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare <400 x ptr addrspace(1)> @llvm.experimental.gc.relocate.v400p1(token, i32, i32)
