; Test the code after a noreturn call for each exception model.

; RUN: llc -mtriple=thumbv7-windows-gnu < %s | FileCheck --check-prefix=SEH %s
; RUN: llc -mtriple=thumbv7-windows-gnu -exception-model=dwarf < %s | FileCheck --check-prefix=DWARF %s
; RUN: llc -mtriple=thumbv7-linux-gnueabihf < %s | FileCheck --check-prefix=ELF %s

declare void @abort() noreturn

; SEH:      .seh_proc f
; SEH:      bl abort
; SEH-NEXT: .seh_endproc

; DWARF:      .cfi_startproc
; DWARF:      bl abort
; DWARF-NEXT: .cfi_endproc

; ELF:      .fnstart
; ELF:      bl abort
; ELF-NEXT: .Lfunc_end0:
define void @f() {
  call void @abort()
  unreachable
}
