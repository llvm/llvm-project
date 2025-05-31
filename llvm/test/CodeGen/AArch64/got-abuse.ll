; RUN: llc -mtriple=aarch64-none-linux-gnu -relocation-model=pic -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -relocation-model=pic -filetype=obj -o - %s

; LLVM gives well-defined semantics to this horrible construct (though C says
; it's undefined). Regardless, we shouldn't crash. The important feature here is
; that in general the only way to access a GOT symbol is via a 64-bit
; load. Neither of these alternatives has the ELF relocations required to
; support it:
;    + ldr wD, [xN, #:got_lo12:func]
;    + add xD, xN, #:got_lo12:func

declare void @consume(i32)
declare void @func()

define void @foo() nounwind {
; CHECK-LABEL: foo:
entry:
  call void @consume(i32 ptrtoint (ptr @func to i32))
; CHECK:      adrp x[[ADDRHI:[0-9]+]], :got:func
; CHECK-NEXT: ldr {{x[0-9]+}}, [x[[ADDRHI]], {{#?}}:got_lo12:func]
  ret void
}

