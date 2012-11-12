; RUN: llc < %s -mtriple=powerpc-apple-darwin10 -relocation-model=pic -disable-fp-elim | FileCheck %s
; <rdar://problem/7604010>

%cmd.type = type { }

@_cmd = constant %cmd.type zeroinitializer

; CHECK:      .globl __cmd
; CHECK-NEXT: .align 3
; CHECK-NEXT: __cmd:
; CHECK-NEXT: .byte 0

; PR6340

%Ty = type { i32, {}, i32 }
@k = global %Ty { i32 1, {} zeroinitializer, i32 3 }

; CHECK: _k:
; CHECK-NEXT:	.long	1
; CHECK-NEXT:	.long	3
