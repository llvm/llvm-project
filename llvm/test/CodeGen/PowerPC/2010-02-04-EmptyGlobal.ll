; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic -frame-pointer=all | FileCheck %s
; <rdar://problem/7604010>

%cmd.type = type { }

@_cmd = constant %cmd.type zeroinitializer

; CHECK:      .globl _cmd
; CHECK-NEXT: .p2align 3
; CHECK-NEXT: _cmd:
; CHECK-NEXT: .size _cmd, 0

; PR6340

%Ty = type { i32, {}, i32 }
@k = global %Ty { i32 1, {} zeroinitializer, i32 3 }

; CHECK: k:
; CHECK-NEXT:	.long	1
; CHECK-NEXT:	.long	3
