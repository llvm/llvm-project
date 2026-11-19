; RUN: llc -mtriple=aarch64-pc-windows-msvc -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-pc-windows-msvc -verify-machineinstrs -filetype=obj %s -o %t.obj
; RUN: llvm-readobj --unwind %t.obj | FileCheck %s --check-prefix=UNWIND
; RUN: llc -mtriple=aarch64-pc-windows-msvc -enable-shrink-wrap=false -filetype=obj %s -o %t-disabled.obj
; RUN: llvm-readobj --unwind %t-disabled.obj | FileCheck %s --check-prefix=DISABLED

; FP is established before SP is realigned. The adjustment belongs to the
; range body, where unwinding uses FP rather than the aligned SP.
define i32 @realigned_stack(i1 %skip) uwtable {
; ASM-LABEL: realigned_stack:
; ASM-NOT: .seh_proc
; ASM: tbz w0, #0, [[SLOW:\.LBB[0-9_]+]]
; ASM-NOT: .seh_proc
; ASM: ret
; ASM-NEXT: [[SLOW]]:
; ASM-NEXT: .seh_proc realigned_stack
; ASM: mov x29, sp
; ASM-NEXT: .seh_set_fp
; ASM-NEXT: .seh_endprologue
; ASM: and sp,
; ASM: bl work
; ASM: .seh_startepilogue
; ASM-NEXT: mov sp, x29
; ASM-NEXT: .seh_set_fp
;
; UNWIND: Function: realigned_stack +0xC (0xC)
; UNWIND: FunctionLength: 40
; UNWIND: Prologue [
; UNWIND-NEXT: mov x29, sp
; UNWIND-NEXT: stp x29, lr, [sp, #-16]!
; UNWIND-NEXT: end
; DISABLED: Function: realigned_stack (0x0)
entry:
  %buffer = alloca [32 x i8], align 64
  br i1 %skip, label %return, label %slow

slow:
  %called = call i32 @work(ptr %buffer)
  %result = add i32 %called, 1
  br label %return

return:
  %value = phi i32 [ 0, %entry ], [ %result, %slow ]
  ret i32 %value
}

; A large realigned frame must also keep the probe inside the range body.
define i32 @realigned_stack_probe(i1 %skip) uwtable {
; ASM-LABEL: realigned_stack_probe:
; ASM-NOT: .seh_proc
; ASM: tbz w0, #0, [[PROBE:\.LBB[0-9_]+]]
; ASM-NOT: .seh_proc
; ASM: ret
; ASM-NEXT: [[PROBE]]:
; ASM-NEXT: .seh_proc realigned_stack_probe
; ASM: add x29, sp, #8
; ASM-NEXT: .seh_add_fp 8
; ASM-NEXT: .seh_endprologue
; ASM: bl __chkstk
; ASM-NEXT: sub sp, sp, x15, lsl #4
; ASM: and sp,
; ASM: bl work
; ASM: .seh_startepilogue
; ASM-NEXT: sub sp, x29, #8
; ASM-NEXT: .seh_add_fp 8
;
; UNWIND: Function: realigned_stack_probe +0xC
; UNWIND: FunctionLength: 60
; UNWIND: EpiloguePacked: Yes
; UNWIND-NEXT: EpilogueOffset: 0
; UNWIND: Prologue [
; UNWIND-NEXT: 0xe201              ; add fp, sp, #8
; UNWIND-NEXT: 0x41                ; stp x29, x30, [sp, #8]
; UNWIND-NEXT: 0xd523              ; str x28, [sp, #-32]!
; UNWIND-NEXT: 0xe4                ; end
; DISABLED: Function: realigned_stack_probe (
entry:
  %buffer = alloca [8192 x i8], align 64
  br i1 %skip, label %return, label %slow

slow:
  %called = call i32 @work(ptr %buffer)
  %result = add i32 %called, 1
  br label %return

return:
  %value = phi i32 [ 0, %entry ], [ %result, %slow ]
  ret i32 %value
}

declare i32 @work(ptr)
