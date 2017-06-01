; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

; An unknown personality forces us to emit an Itanium LSDA. Make sure that the
; Itanium call site table actually tells the personality to keep unwinding,
; i.e. we have an entry and it says "has no landing pad".

declare void @throwit()
declare void @__unknown_ehpersonality(...)

define void @use_unknown_ehpersonality()
    personality void (...)* @__unknown_ehpersonality {
entry:
  call void @throwit()
  unreachable
}

; CHECK-LABEL: use_unknown_ehpersonality:
; CHECK: .Lfunc_begin0:
; CHECK: .seh_handler __unknown_ehpersonality, @unwind, @except
; CHECK: callq throwit
; CHECK: .Lfunc_end0:
; CHECK: .seh_handlerdata
; CHECK: .Lexception0:
; CHECK:  .byte   255                     # @LPStart Encoding = omit
; CHECK:  .byte   0                       # @TType Encoding = absptr
; CHECK:  .asciz  "\217\200"              # @TType base offset
; CHECK:  .byte   3                       # Call site Encoding = udata4
; CHECK:  .byte   13                      # Call site table length
; CHECK:  .long   .Lfunc_begin0-.Lfunc_begin0 # >> Call Site 1 <<
; CHECK:  .long   .Lfunc_end0-.Lfunc_begin0 #   Call between .Lfunc_begin0 and .Lfunc_end0
; CHECK:  .long   0                       #     has no landing pad
; CHECK:  .byte   0                       #   On action: cleanup
