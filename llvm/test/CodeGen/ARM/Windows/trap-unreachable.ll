; Test the code after a noreturn call for each exception model. With Windows
; unwind info, a trap follows a noreturn call that would end a function or
; funclet.

; RUN: split-file %s %t
; RUN: llc -mtriple=thumbv7-windows-gnu < %t/noreturn.ll | FileCheck --check-prefix=SEH %s
; RUN: llc -mtriple=thumbv7-windows-gnu -exception-model=dwarf < %t/noreturn.ll | FileCheck --check-prefix=DWARF %s
; RUN: llc -mtriple=thumbv7-linux-gnueabihf < %t/noreturn.ll | FileCheck --check-prefix=ELF %s
; RUN: llc -mtriple=thumbv7-windows-msvc < %t/funclet.ll | FileCheck --check-prefix=FUNCLET %s

;--- noreturn.ll
declare void @abort() noreturn
declare void @exit(i32) noreturn

; SEH:      .seh_proc f
; SEH:      bl abort
; SEH-NEXT: .inst.n 0xdefe
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

; Only the call that ends the function needs a trap.
; SEH:      .seh_proc g
; SEH:      bleq abort
; SEH-NEXT: movs r0, #1
; SEH-NEXT: bl exit
; SEH-NEXT: .inst.n 0xdefe
; SEH-NEXT: .seh_endproc

; DWARF:      .cfi_startproc
; DWARF:      bl exit
; DWARF-NEXT: .cfi_endproc

; ELF:      .fnstart
; ELF:      bl exit
; ELF-NEXT: .Lfunc_end1:
define void @g(i32 %x) {
entry:
  %c = icmp eq i32 %x, 0
  br i1 %c, label %a, label %b
a:
  call void @abort()
  unreachable
b:
  call void @exit(i32 1)
  unreachable
}

;--- funclet.ll
declare void @may_throw()
declare void @abort() noreturn
declare void @note()
declare i32 @__C_specific_handler(...)

; The parent's code ends with a noreturn call right before the cleanup funclet,
; and the funclet ends with one too.
; FUNCLET:      .seh_proc h
; FUNCLET:      bl abort
; FUNCLET-NEXT: .inst.n 0xdefe
; FUNCLET:      .seh_proc "?dtor$2@?0?h@4HA"
; FUNCLET:      bl abort
; FUNCLET-NEXT: .inst.n 0xdefe
define void @h() personality ptr @__C_specific_handler {
entry:
  invoke void @may_throw() to label %cont unwind label %cleanup
cont:
  call void @abort()
  unreachable
cleanup:
  %cp = cleanuppad within none []
  call void @abort() [ "funclet"(token %cp) ]
  unreachable
}

; The invoke falls through to its empty unreachable block, which ends the
; function.
; FUNCLET:      .seh_proc inv
; FUNCLET:      bl abort
; FUNCLET-NEXT: @ EH_LABEL
; FUNCLET-NEXT: %unreach
; FUNCLET-NEXT: .inst.n 0xdefe
define void @inv() personality ptr @__C_specific_handler {
entry:
  invoke void @abort() to label %unreach unwind label %cleanup
unreach:
  unreachable
cleanup:
  %cp = cleanuppad within none []
  call void @note() [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}
