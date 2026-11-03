; RUN: llc -mtriple=aarch64-pc-windows-msvc -mattr=+pauth -verify-machineinstrs \
; RUN:   -stop-before=aarch64-win-cfi %s -o - | FileCheck %s --check-prefix=BEFORE
; RUN: llc -mtriple=aarch64-pc-windows-msvc -mattr=+pauth -verify-machineinstrs \
; RUN:   -stop-after=aarch64-win-cfi %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=aarch64-pc-windows-msvc -mattr=+pauth -verify-machineinstrs %s -o %t.s
; RUN: FileCheck %s --check-prefix=ASM --input-file=%t.s
; RUN: llc -mtriple=aarch64-pc-windows-msvc -mattr=+pauth -verify-machineinstrs \
; RUN:   -filetype=obj %s -o %t.obj
; RUN: llvm-readobj --unwind %t.obj | FileCheck %s --check-prefix=UNWIND
; RUN: llvm-mc -triple=aarch64-pc-windows-msvc -mattr=+pauth -filetype=obj %t.s -o %t-asm.obj
; RUN: llvm-readobj --unwind %t-asm.obj | FileCheck %s --check-prefix=UNWIND

; A single MIR terminator can expand to multiple instructions. AUTH_TCRETURN
; materializes the discriminator before branching, so the epilogue's implicit
; final instruction cannot cover the whole expansion. Exclude both instructions
; from the unwind record to avoid unwinding the restored frame again at BRAA.
define void @auth_tail(ptr %target, i1 %skip) uwtable {
; BEFORE: frame-destroy SEH_EpilogEnd
; BEFORE-NEXT: AUTH_TCRETURN {{.*}}, 0, 0, 42,
;
; MIR: frame-destroy SEH_EpilogEnd
; MIR-NEXT: SEH_EndProc
; MIR-NEXT: AUTH_TCRETURN {{.*}}, 0, 0, 42,
;
; ASM-LABEL: auth_tail:
; ASM-NOT: .seh_proc
; ASM: tbz w1, #0,
; ASM: .seh_proc auth_tail
; ASM-NEXT: str x30, [sp, #-16]!
; ASM-NEXT: .seh_save_reg_x x30, 16
; ASM-NEXT: .seh_endprologue
; ASM: bl work
; ASM: .seh_startepilogue
; ASM-NEXT: ldr x30, [sp], #16
; ASM-NEXT: .seh_save_reg_x x30, 16
; ASM-NEXT: .seh_endepilogue
; ASM-NEXT: .seh_endfunclet
; ASM-NEXT: .seh_endproc
; ASM-NEXT: mov x16, #42
; ASM-NEXT: braa x0, x16
;
; UNWIND: RuntimeFunction {
; UNWIND-NEXT: Function: auth_tail +0x8 (0x8)
; UNWIND: FunctionLength: 20
; UNWIND: EpilogueScopes: 1
; UNWIND: Prologue [
; UNWIND-NEXT: 0xd561              ; str x30, [sp, #-16]!
; UNWIND-NEXT: 0xe4                ; end
; UNWIND: StartOffset: 4
; UNWIND-NEXT: EpilogueStartIndex: 0
; UNWIND-NEXT: Opcodes [
; UNWIND-NEXT: 0xd561              ; ldr x30, [sp], #16
; UNWIND-NEXT: 0xe4                ; end
; UNWIND-NOT: RuntimeFunction {
entry:
  br i1 %skip, label %return, label %slow

slow:
  call void @work()
  tail call void %target() [ "ptrauth"(i32 0, i64 42) ]
  ret void

return:
  ret void
}

declare void @work()
