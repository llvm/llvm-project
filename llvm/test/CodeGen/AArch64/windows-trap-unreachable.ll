; Test the trap emitted after a noreturn call for each exception model.
; Windows unwind info needs a trap only when the call would otherwise end a
; function or funclet.

; RUN: split-file %s %t
; RUN: llc -mtriple=aarch64-windows-gnu < %t/noreturn.ll | FileCheck --check-prefix=SEH %s
; RUN: llc -mtriple=aarch64-windows-gnu -trap-unreachable < %t/noreturn.ll | FileCheck --check-prefix=SEH %s
; RUN: llc -mtriple=aarch64-windows-gnu -exception-model=dwarf < %t/noreturn.ll | FileCheck --check-prefix=DWARF %s
; RUN: llc -mtriple=aarch64-linux-gnu < %t/noreturn.ll | FileCheck --check-prefix=ELF %s
; RUN: llc -mtriple=aarch64-windows-msvc < %t/funclet.ll | FileCheck --check-prefix=FUNCLET %s
; RUN: llc -mtriple=aarch64-windows-msvc -trap-unreachable < %t/funclet.ll | FileCheck --check-prefix=FUNCLET %s

;--- noreturn.ll
declare void @abort() noreturn
declare void @exit(i32) noreturn

; With -trap-unreachable, the trap is not emitted twice.
; SEH-LABEL: f:
; SEH:       bl abort
; SEH-NEXT:  brk #0x1
; SEH-NEXT:  .seh_endfunclet

; DWARF-LABEL: f:
; DWARF:       bl abort
; DWARF-NEXT:  .cfi_endproc

; ELF-LABEL: f:
; ELF:       bl abort
; ELF-NEXT:  .Lfunc_end0:
define void @f() {
  call void @abort()
  unreachable
}

; The first call is followed by more of the function, so it needs no trap.
; SEH-LABEL:      g:
; SEH:            bl abort
; SEH:            bl exit
; SEH-NEXT:       brk #0x1
; SEH-NEXT:       .seh_endfunclet

; DWARF-LABEL: g:
; DWARF:       bl abort
; DWARF-NEXT:  .LBB1_2:
; DWARF:       bl exit
; DWARF-NEXT:  .cfi_endproc

; ELF-LABEL: g:
; ELF:       bl abort
; ELF-NEXT:  .LBB1_2:
; ELF:       bl exit
; ELF-NEXT:  .Lfunc_end1:
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
declare i32 @__CxxFrameHandler3(...)

; Each catch funclet ends with the noreturn call, so each needs a trap.
; FUNCLET-LABEL: "?catch$2@?0?h@4HA":
; FUNCLET:       bl abort
; FUNCLET-NEXT:  brk #0x1
; FUNCLET-NEXT:  .seh_endfunclet
; FUNCLET-LABEL: "?catch$3@?0?h@4HA":
; FUNCLET:       bl abort
; FUNCLET-NEXT:  brk #0x1
; FUNCLET-NEXT:  .Lfunc_end0:
define void @h() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @may_throw() to label %ret unwind label %dispatch
dispatch:
  %cs = catchswitch within none [label %catch1, label %catch2] unwind to caller
catch1:
  %p1 = catchpad within %cs [ptr null, i32 0, ptr null]
  call void @abort() [ "funclet"(token %p1) ]
  unreachable
catch2:
  %p2 = catchpad within %cs [ptr null, i32 64, ptr null]
  call void @abort() [ "funclet"(token %p2) ]
  unreachable
ret:
  ret void
}
