; RUN: llc < %s | FileCheck %s
; RUN: llc -debug-only=stackmaps < %s 2>&1 | FileCheck --check-prefix=STACKMAPS %s
; REQUIRES: asserts

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare webkit_jscc i64  @llvm.experimental.deoptimize.i64(...)

define i64 @caller_1() {
; CHECK-LABEL: _caller_1:
; CHECK-NEXT: {{.+cfi.+}}
; CHECK-NEXT: ##{{.+}}
; CHECK-NEXT: pushq	%rax
; CHECK-NEXT: {{Ltmp[0-9]+}}:
; CHECK-NEXT: {{.+cfi.+}}
; CHECK-NEXT: movl	$1140457472, (%rsp)     ## imm = 0x43FA0000
; CHECK-NEXT: movl	$42, %eax
; CHECK-NEXT: callq	___llvm_deoptimize
; CHECK-NEXT: {{Ltmp[0-9]+}}:

entry:
  %v = call webkit_jscc i64(...) @llvm.experimental.deoptimize.i64(i32 42, float 500.0) [ "deopt"(i32 3) ]
  ret i64 %v
}

; STACKMAPS: Stack Maps: callsites:
; STACKMAPS-NEXT: Stack Maps: callsite 2882400015
; STACKMAPS-NEXT: Stack Maps:   has 4 locations
; STACKMAPS-NEXT: Stack Maps: 		Loc 0: Constant 12	[encoding: .byte 4, .byte 8, .short 0, .int 12]
; STACKMAPS-NEXT: Stack Maps: 		Loc 1: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 2: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 		Loc 3: Constant 3	[encoding: .byte 4, .byte 8, .short 0, .int 3]
; STACKMAPS-NEXT: Stack Maps: 	has 0 live-out registers
