; RUN: llc -verify-machineinstrs -filetype=obj -o - -mtriple=x86_64-apple-macosx < %s | llvm-objdump --no-print-imm-hex --triple=x86_64-apple-macosx -d - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=x86_64-apple-macosx < %s | FileCheck %s --check-prefix=CHECK-ALIGN
; RUN: llc -verify-machineinstrs -show-mc-encoding -mtriple=i386 < %s | FileCheck %s --check-prefixes=X86,X86CFI,XCHG
; RUN: llc -verify-machineinstrs -show-mc-encoding -mtriple=i386-windows-msvc < %s | FileCheck %s --check-prefixes=X86,MOV
; RUN: llc -verify-machineinstrs -show-mc-encoding -mtriple=i386-windows-msvc -mcpu=pentium3 < %s | FileCheck %s --check-prefixes=X86,MOV
; RUN: llc -verify-machineinstrs -show-mc-encoding -mtriple=i386-windows-msvc -mcpu=pentium4 < %s | FileCheck %s --check-prefixes=X86,XCHG
; RUN: llc -verify-machineinstrs -show-mc-encoding -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=X64

declare void @callee(ptr)

define void @f0() "patchable-function"="prologue-short-redirect" {
; CHECK-LABEL: _f0{{>?}}:
; CHECK-NEXT:  66 90 	nop

; CHECK-ALIGN: 	.p2align	4
; CHECK-ALIGN: _f0:

; X86: f0:
; X86CFI-NEXT: .cfi_startproc
; X86-NEXT: # %bb.0:
; XCHG-NEXT: xchgw   %ax, %ax                # encoding: [0x66,0x90]
; MOV-NEXT: movl    %edi, %edi              # encoding: [0x8b,0xff]
; X86-NEXT: retl

; X64: f0:
; X64-NEXT: # %bb.0:
; X64-NEXT: xchgw   %ax, %ax                # encoding: [0x66,0x90]
; X64-NEXT: retq

  ret void
}

define void @f1() "patchable-function"="prologue-short-redirect" "frame-pointer"="all" {
; CHECK-LABEL: _f1
; CHECK-NEXT: 66 90     nop
; CHECK-NEXT: 55		pushq	%rbp

; CHECK-ALIGN: 	.p2align	4
; CHECK-ALIGN: _f1:

; X86: f1:
; X86CFI-NEXT: .cfi_startproc
; X86-NEXT: # %bb.0:
; XCHG-NEXT: xchgw   %ax, %ax                # encoding: [0x66,0x90]
; MOV-NEXT: movl    %edi, %edi              # encoding: [0x8b,0xff]
; X86-NEXT: pushl   %ebp

; X64: f1:
; X64-NEXT: .seh_proc f1
; X64-NEXT: # %bb.0:
; X64-NEXT: xchgw %ax, %ax
; X64-NEXT: pushq   %rbp

  ret void
}

define void @f2() "patchable-function"="prologue-short-redirect" {
; CHECK-LABEL: _f2
; CHECK-NEXT: 48 81 ec a8 00 00 00 	subq	$168, %rsp

; CHECK-ALIGN: 	.p2align	4
; CHECK-ALIGN: _f2:

; X86: f2:
; X86CFI-NEXT: .cfi_startproc
; X86-NEXT: # %bb.0:
; XCHG-NEXT: xchgw   %ax, %ax                # encoding: [0x66,0x90]
; MOV-NEXT: movl    %edi, %edi              # encoding: [0x8b,0xff]
; X86-NEXT: pushl   %ebp

; X64: f2:
; X64-NEXT: .seh_proc f2
; X64-NEXT: # %bb.0:
; X64-NEXT: subq    $200, %rsp

  %ptr = alloca i64, i32 20
  call void @callee(ptr %ptr)
  ret void
}

define void @f3() "patchable-function"="prologue-short-redirect" optsize {
; CHECK-LABEL: _f3
; CHECK-NEXT: 66 90 	nop

; CHECK-ALIGN: 	.p2align	4
; CHECK-ALIGN: _f3:

; X86: f3:
; X86CFI-NEXT: .cfi_startproc
; X86-NEXT: # %bb.0:
; XCHG-NEXT: xchgw   %ax, %ax
; MOV-NEXT: movl   %edi, %edi
; X86-NEXT: retl

; X64: f3:
; X64-NEXT: # %bb.0:
; X64-NEXT: xchgw   %ax, %ax
; X64-NEXT: retq

  ret void
}

; This testcase happens to produce a KILL instruction at the beginning of the
; first basic block. In this case the 2nd instruction should be turned into a
; patchable one.
; CHECK-LABEL: f4{{>?}}:
; CHECK-NEXT: 8b 0c 37  movl  (%rdi,%rsi), %ecx
; X86: f4:
; X86CFI-NEXT: .cfi_startproc
; X86-NEXT: # %bb.0:
; XCHG-NEXT: xchgw   %ax, %ax
; MOV-NEXT: movl   %edi, %edi
; X86-NEXT: pushl   %ebx

; X64: f4:
; X64-NEXT: # %bb.0:
; X64-NOT: xchgw   %ax, %ax

define i32 @f4(ptr %arg1, i64 %arg2, i32 %arg3) "patchable-function"="prologue-short-redirect" {
bb:
  %tmp10 = getelementptr i8, ptr %arg1, i64 %arg2
  %tmp12 = load i32, ptr %tmp10, align 4
  fence acquire
  %tmp13 = add i32 %tmp12, %arg3
  %tmp14 = cmpxchg ptr %tmp10, i32 %tmp12, i32 %tmp13 seq_cst monotonic
  %tmp15 = extractvalue { i32, i1 } %tmp14, 1
  br i1 %tmp15, label %bb21, label %bb16

bb16:
  br label %bb21

bb21:
  %tmp22 = phi i32 [ %tmp12, %bb ], [ %arg3, %bb16 ]
  ret i32 %tmp22
}

; This testcase produces an empty function (not even a ret on some targets).
; This scenario can happen with undefined behavior.
; Ensure that the "patchable-function" pass supports this case.
; CHECK-LABEL: _emptyfunc
; CHECK-NEXT: 0f 0b 	ud2

; CHECK-ALIGN: 	.p2align	4
; CHECK-ALIGN: _emptyfunc:

; X86: emptyfunc:
; X86CFI-NEXT: .cfi_startproc
; X86-NEXT: # %bb.0:
; XCHG-NEXT: xchgw   %ax, %ax
; MOV-NEXT: movl   %edi, %edi

; X64: emptyfunc:
; X64-NEXT: # %bb.0:
; X64-NEXT: xchgw   %ax, %ax

; From code: int emptyfunc() {}
define i32 @emptyfunc() "patchable-function"="prologue-short-redirect" {
  unreachable
}


; Hotpatch feature must ensure no jump within the function goes to the first instruction.
; From code:
; void jmp_to_start(char *b) {
;   do {
;   } while ((++(*b++)));
; }

; CHECK-ALIGN: 	.p2align	4
; CHECK-ALIGN: _jmp_to_start:

; X86: jmp_to_start:
; X86CFI-NEXT: .cfi_startproc
; X86-NEXT: # %bb.0:
; XCHG-NEXT: xchgw   %ax, %ax
; MOV-NEXT: movl   %edi, %edi

; X64: jmp_to_start:
; X64-NEXT: # %bb.0:
; X64-NEXT: xchgw   %ax, %ax

define dso_local void @jmp_to_start(ptr inreg nocapture noundef %b) "patchable-function"="prologue-short-redirect" {
entry:
  br label %do.body
do.body:                                          ; preds = %do.body, %entry
  %b.addr.0 = phi ptr [ %b, %entry ], [ %incdec.ptr, %do.body ]
  %incdec.ptr = getelementptr inbounds i8, ptr %b.addr.0, i64 1
  %0 = load i8, ptr %b.addr.0, align 1
  %inc = add i8 %0, 1
  store i8 %inc, ptr %b.addr.0, align 1
  %tobool.not = icmp eq i8 %inc, 0
  br i1 %tobool.not, label %do.end, label %do.body
do.end:                                           ; preds = %do.body
  ret void
}


; Test that inline asm is properly hotpatched. We currently don't examine the
; asm instruction when printing it, thus we always emit patching NOPs.

; X64: inline_asm:
; X64-NEXT: # %bb.0:
; X64-NEXT: xchgw   %ax, %ax                        # encoding: [0x66,0x90]
; X64-NEXT: #APP
; X64-NEXT: int3                                    # encoding: [0xcc]
; X64-NEXT: #NO_APP

define dso_local void @inline_asm() "patchable-function"="prologue-short-redirect" {
entry:
  call void asm sideeffect "int3", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}
