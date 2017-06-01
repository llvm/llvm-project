; RUN: llc < %s | FileCheck %s
; RUN: llc -debug-only=stackmaps < %s 2>&1 | FileCheck --check-prefix=STACKMAPS %s
; REQUIRES: asserts

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare i32 @llvm.experimental.deoptimize.i32(...)
declare i8  @llvm.experimental.deoptimize.i8(...)

define i32 @caller_0() {
; CHECK-LABEL: _caller_0:
; CHECK-NEXT: {{.+cfi.+}}
; CHECK-NEXT: ##{{.+}}
; CHECK-NEXT: pushq   %rax
; CHECK-NEXT: {{Lcfi[0-9]+}}:
; CHECK-NEXT: {{.+cfi.+}}
; CHECK-NEXT: callq	___llvm_deoptimize
; CHECK-NEXT: {{Ltmp[0-9]+}}:
entry:
  %v = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 0) ]
  ret i32 %v
}

define i8 @caller_1() {
; CHECK-LABEL: _caller_1:
; CHECK-NEXT: {{.+cfi.+}}
; CHECK-NEXT: ##{{.+}}
; CHECK-NEXT: pushq	%rax
; CHECK-NEXT: {{Lcfi[0-9]+}}:
; CHECK-NEXT: {{.+cfi.+}}
; CHECK-NEXT: movss	{{[a-zA-Z0-9_]+}}(%rip), %xmm0    ## xmm0 = mem[0],zero,zero,zero
; CHECK-NEXT: movl	$42, %edi
; CHECK-NEXT: callq	___llvm_deoptimize
; CHECK-NEXT: {{Ltmp[0-9]+}}:

entry:
  %v = call i8(...) @llvm.experimental.deoptimize.i8(i32 42, float 500.0) [ "deopt"(i32 1) ]
  ret i8 %v
}

; STACKMAPS: Stack Maps: callsites:
; STACKMAPS-NEXT: Stack Maps: callsite 2882400015
; STACKMAPS-NEXT: Stack Maps:   has 4 locations
; STACKMAPS-NEXT: Stack Maps: 		Loc 0: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 1: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 2: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 		Loc 3: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 	has 0 live-out registers
; STACKMAPS-NEXT: Stack Maps: callsite 2882400015
; STACKMAPS-NEXT: Stack Maps:   has 4 locations
; STACKMAPS-NEXT: Stack Maps: 		Loc 0: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 1: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 2: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 		Loc 3: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 	has 0 live-out registers
