; Test the code after a noreturn call for each exception model.

; RUN: llc -mtriple=x86_64-windows-gnu < %s | FileCheck --check-prefix=SEH %s
; RUN: llc -mtriple=x86_64-windows-gnu -exception-model=dwarf < %s | FileCheck --check-prefix=DWARF %s
; RUN: llc -mtriple=x86_64-linux-gnu < %s | FileCheck --check-prefix=ELF %s

declare void @abort() noreturn

; SEH:      .seh_proc f
; SEH:      callq abort
; SEH-NEXT: int3
; SEH-NEXT: .seh_endproc

; DWARF:      .cfi_startproc
; DWARF:      callq abort
; DWARF-NEXT: .cfi_endproc

; ELF:      .cfi_startproc
; ELF:      callq abort@PLT
; ELF-NEXT: .Lfunc_end0:
define void @f() {
  call void @abort()
  unreachable
}
