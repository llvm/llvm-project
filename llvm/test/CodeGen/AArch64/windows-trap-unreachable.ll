; Test the trap emitted after a noreturn call for each exception model.
; The exception model changes whether the trap is emitted by default.

; RUN: llc -mtriple=aarch64-windows-gnu < %s | FileCheck --check-prefix=SEH %s
; RUN: llc -mtriple=aarch64-windows-gnu -exception-model=dwarf < %s | FileCheck --check-prefix=DWARF %s
; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck --check-prefix=ELF %s

declare void @abort() noreturn

; SEH:      .seh_proc f
; SEH:      bl abort
; SEH-NEXT: brk #0x1

; DWARF:      .cfi_startproc
; DWARF:      bl abort
; DWARF-NEXT: .cfi_endproc

; ELF:      .cfi_startproc
; ELF:      bl abort
; ELF-NEXT: .Lfunc_end0:
define void @f() {
  call void @abort()
  unreachable
}
