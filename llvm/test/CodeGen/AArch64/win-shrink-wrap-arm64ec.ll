; RUN: llc -mtriple=arm64ec-pc-windows-msvc -verify-machineinstrs %s -o %t.s
; RUN: FileCheck %s --check-prefix=ASM --input-file=%t.s
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -verify-machineinstrs -filetype=obj %s -o %t.obj
; RUN: llvm-readobj --unwind %t.obj | FileCheck %s --check-prefix=UNWIND
; RUN: llvm-mc -triple=arm64ec-pc-windows-msvc -filetype=obj %t.s -o %t-asm.obj
; RUN: llvm-readobj --unwind %t-asm.obj | FileCheck %s --check-prefix=UNWIND
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -enable-shrink-wrap=false \
; RUN:   -verify-machineinstrs -filetype=obj %s -o %t-disabled.obj
; RUN: llvm-readobj --unwind %t-disabled.obj | FileCheck %s --check-prefix=DISABLED

; Native ARM64EC code can shrink-wrap its frame while its x64 entry thunk and
; exit thunks retain their own unwind records and extended register-save codes.
define i32 @ec_shrinkwrap(i32 %skip) uwtable {
; ASM-LABEL: "#ec_shrinkwrap":
; ASM-NOT: .seh_proc
; ASM: cbz w0,
; ASM: .seh_proc "#ec_shrinkwrap"
; ASM-NEXT: str x30, [sp, #-16]!
; ASM-NEXT: .seh_save_reg_x x30, 16
; ASM-NEXT: .seh_endprologue
; ASM: bl "#work"
; ASM-NEXT: add w0, w0, #1
; ASM-NEXT: .seh_startepilogue
; ASM-NEXT: ldr x30, [sp], #16
; ASM-NEXT: .seh_save_reg_x x30, 16
; ASM-NEXT: .seh_endepilogue
; ASM-NEXT: .seh_endfunclet
; ASM-NEXT: .seh_endproc
; ASM-NEXT: [[RETURN:.LBB[0-9_]+]]:
; ASM-NEXT: ret{{$}}
;
; UNWIND: Format: COFF-ARM64EC
; UNWIND: Function: #ec_shrinkwrap +0x4 (0x4)
; UNWIND: FunctionLength: 16
; UNWIND: Prologue [
; UNWIND-NEXT: 0xd561              ; str x30, [sp, #-16]!
; UNWIND-NEXT: 0xe4                ; end
; UNWIND: StartOffset: 3
; UNWIND-NEXT: EpilogueStartIndex: 0
; UNWIND-NEXT: Opcodes [
; UNWIND-NEXT: 0xd561              ; ldr x30, [sp], #16
; UNWIND-NEXT: 0xe4                ; end
; DISABLED: Function: #ec_shrinkwrap (0x0)
entry:
  %skip_call = icmp eq i32 %skip, 0
  br i1 %skip_call, label %return, label %slow

slow:
  %called = call i32 @work()
  %result = add i32 %called, 1
  br label %return

return:
  %value = phi i32 [ 0, %entry ], [ %result, %slow ]
  ret i32 %value
}

declare i32 @work()

; ASM-LABEL: $ientry_thunk$cdecl$i8$i8:
; ASM-NEXT: .seh_proc $ientry_thunk$cdecl$i8$i8
; ASM: .seh_save_any_reg_px q6, 176
; ASM: .seh_endprologue
; ASM-NOT: .seh_end_c
; ASM: .seh_startepilogue
; ASM: .seh_save_any_reg_px q6, 176
; ASM-NEXT: .seh_endepilogue
; ASM-NEXT: br x1
; ASM-NEXT: .seh_endfunclet
; ASM-NEXT: .seh_endproc
; ASM-LABEL: $iexit_thunk$cdecl$i8$v:
; ASM-NEXT: .seh_proc $iexit_thunk$cdecl$i8$v
; ASM: .seh_endprologue
; ASM-NOT: .seh_end_c
; ASM: .seh_startepilogue
; ASM: .seh_endepilogue
; ASM-NEXT: ret
; ASM-NEXT: .seh_endfunclet
; ASM-NEXT: .seh_endproc
; ASM-LABEL: "#work$exit_thunk":
; ASM: .seh_proc "#work$exit_thunk"
; ASM: .seh_endprologue
; ASM-NOT: .seh_end_c
; ASM: .seh_startepilogue
; ASM: .seh_endepilogue
; ASM-NEXT: br x11
; ASM-NEXT: .seh_endfunclet
; ASM-NEXT: .seh_endproc

; UNWIND: Function: $ientry_thunk$cdecl$i8$i8 (0x0)
; UNWIND: Prologue [
; UNWIND-NEXT: 0xe214              ; add fp, sp, #160
; UNWIND-NEXT: 0x54                ; stp x29, x30, [sp, #160]
; UNWIND-NEXT: 0xe74e88            ; stp q14, q15, [sp, #128]
; UNWIND-NEXT: 0xe74c86            ; stp q12, q13, [sp, #96]
; UNWIND-NEXT: 0xe74a84            ; stp q10, q11, [sp, #64]
; UNWIND-NEXT: 0xe74882            ; stp q8, q9, [sp, #32]
; UNWIND-NEXT: 0xe7668a            ; stp q6, q7, [sp, #-176]!
; UNWIND-NEXT: 0xe4                ; end
; UNWIND: Epilogue [
; UNWIND-NEXT: 0x54                ; ldp x29, x30, [sp, #160]
; UNWIND-NEXT: 0xe74e88            ; ldp q14, q15, [sp, #128]
; UNWIND-NEXT: 0xe74c86            ; ldp q12, q13, [sp, #96]
; UNWIND-NEXT: 0xe74a84            ; ldp q10, q11, [sp, #64]
; UNWIND-NEXT: 0xe74882            ; ldp q8, q9, [sp, #32]
; UNWIND-NEXT: 0xe7668a            ; ldp q6, q7, [sp], #176
; UNWIND-NEXT: 0xe4                ; end
; UNWIND: Function: $iexit_thunk$cdecl$i8$v (0x0)
; UNWIND: Prologue [
; UNWIND-NEXT: 0xe204              ; add fp, sp, #32
; UNWIND-NEXT: 0x44                ; stp x29, x30, [sp, #32]
; UNWIND-NEXT: 0x03                ; sub sp, #48
; UNWIND-NEXT: 0xe4                ; end
; UNWIND: Epilogue [
; UNWIND-NEXT: 0x44                ; ldp x29, x30, [sp, #32]
; UNWIND-NEXT: 0x03                ; add sp, #48
; UNWIND-NEXT: 0xe4                ; end
; UNWIND: Function: #work$exit_thunk (0x0)
; UNWIND: Prologue [
; UNWIND-NEXT: str lr, [sp, #-16]!
; UNWIND-NEXT: end
; UNWIND-NOT: RuntimeFunction {
