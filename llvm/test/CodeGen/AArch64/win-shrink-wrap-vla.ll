; RUN: llc -mtriple=aarch64-pc-windows-msvc -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-pc-windows-msvc -verify-machineinstrs -filetype=obj %s -o %t.obj
; RUN: llvm-readobj --unwind %t.obj | FileCheck %s --check-prefix=UNWIND
; RUN: llc -mtriple=aarch64-pc-windows-msvc -enable-shrink-wrap=false -filetype=obj %s -o %t-disabled.obj
; RUN: llvm-readobj --unwind %t-disabled.obj | FileCheck %s --check-prefix=DISABLED

; The early-return path must remain outside the unwind range. Establishing FP
; ends the SEH prologue; probing, dynamic allocation, and stacksave/stackrestore
; stay inside the range without adding SP-dependent unwind operations.
define void @vla_probe(i64 %size, i1 %skip) uwtable {
; ASM-LABEL: vla_probe:
; ASM-NOT: .seh_proc
; ASM: tbnz w1, #0, [[RETURN:.LBB[0-9_]+]]
; ASM: .seh_proc vla_probe
; ASM: stp x29, x30,
; ASM: add x29, sp, #8
; ASM-NEXT: .seh_add_fp 8
; ASM-NEXT: .seh_endprologue
; ASM: mov [[SAVED:x[0-9]+]], sp
; ASM: bl __chkstk
; ASM: sub [[ALLOC:x[0-9]+]], sp, x15, lsl #4
; ASM-NEXT: mov sp, [[ALLOC]]
; ASM-NOT: .seh_
; ASM: bl consume
; ASM-NEXT: mov sp, [[SAVED]]
; ASM-NEXT: bl after_restore
; ASM-NEXT: .seh_startepilogue
; ASM-NEXT: sub sp, x29, #8
; ASM-NEXT: .seh_add_fp 8
; ASM: .seh_endepilogue
; ASM-NEXT: .seh_endfunclet
; ASM-NEXT: .seh_endproc
; ASM-NEXT: [[RETURN]]:
; ASM-NEXT: ret{{$}}
;
; UNWIND: Function: vla_probe +0x4 (0x4)
; UNWIND: FunctionLength: 60
; UNWIND: Prologue [
; UNWIND-NEXT: 0xe201              ; add fp, sp, #8
; UNWIND-NEXT: 0x41                ; stp x29, x30, [sp, #8]
; UNWIND-NEXT: 0xd403              ; str x19, [sp, #-32]!
; UNWIND-NEXT: 0xe4                ; end
; UNWIND: EpilogueStartIndex: 0
; UNWIND-NEXT: Opcodes [
; UNWIND-NEXT: 0xe201              ; sub sp, fp, #8
; UNWIND-NEXT: 0x41                ; ldp x29, x30, [sp, #8]
; UNWIND-NEXT: 0xd403              ; ldr x19, [sp], #32
; UNWIND-NEXT: 0xe4                ; end
; DISABLED: Function: vla_probe (0x0)
entry:
  br i1 %skip, label %return, label %allocate

allocate:
  %saved = call ptr @llvm.stacksave()
  %buffer = alloca i8, i64 %size, align 16
  call void @consume(ptr %buffer)
  call void @llvm.stackrestore(ptr %saved)
  call void @after_restore()
  br label %return

return:
  ret void
}

; Disabling stack probing must not disable shrink wrapping or FP-based recovery.
define void @vla_no_probe(i64 %size, i1 %skip) uwtable "no-stack-arg-probe" {
; ASM-LABEL: vla_no_probe:
; ASM-NOT: .seh_proc
; ASM: tbnz w1, #0, [[RETURN:.LBB[0-9_]+]]
; ASM: .seh_proc vla_no_probe
; ASM: stp x29, x30, [sp, #-16]!
; ASM-NEXT: .seh_save_fplr_x 16
; ASM-NEXT: mov x29, sp
; ASM-NEXT: .seh_set_fp
; ASM-NEXT: .seh_endprologue
; ASM-NOT: __chkstk
; ASM: mov sp, {{x[0-9]+}}
; ASM-NOT: .seh_
; ASM-NOT: __chkstk
; ASM: bl consume
; ASM-NEXT: .seh_startepilogue
; ASM-NEXT: mov sp, x29
; ASM-NEXT: .seh_set_fp
; ASM-NEXT: ldp x29, x30, [sp], #16
; ASM-NEXT: .seh_save_fplr_x 16
; ASM-NEXT: .seh_endepilogue
; ASM-NEXT: .seh_endfunclet
; ASM-NEXT: .seh_endproc
; ASM-NEXT: [[RETURN]]:
; ASM-NEXT: ret{{$}}
;
; UNWIND: Function: vla_no_probe +0x4
; UNWIND: FunctionLength: 40
; UNWIND: Prologue [
; UNWIND-NEXT: 0xe1                ; mov fp, sp
; UNWIND-NEXT: 0x81                ; stp x29, x30, [sp, #-16]!
; UNWIND-NEXT: 0xe4                ; end
; UNWIND: EpilogueStartIndex: 0
; UNWIND-NEXT: Opcodes [
; UNWIND-NEXT: 0xe1                ; mov sp, fp
; UNWIND-NEXT: 0x81                ; ldp x29, x30, [sp], #16
; UNWIND-NEXT: 0xe4                ; end
; DISABLED: Function: vla_no_probe (
entry:
  br i1 %skip, label %return, label %allocate

allocate:
  %buffer = alloca i8, i64 %size, align 16
  call void @consume(ptr %buffer)
  br label %return

return:
  ret void
}

declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr)
declare void @consume(ptr)
declare void @after_restore()
