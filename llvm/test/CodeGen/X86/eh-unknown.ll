; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

; An unknown personality forces us to emit an Itanium LSDA. Make sure that the
; Itanium call site table actually tells the personality to keep unwinding,
; i.e. we have an entry and it says "has no landing pad".

declare void @throwit()
declare void @__unknown_ehpersonality(...)

define void @use_unknown_ehpersonality()
    personality ptr @__unknown_ehpersonality {
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
; CHECK:  .byte   255                     # @TType Encoding = omit
; CHECK:  .byte   1                       # Call site Encoding = uleb128
; CHECK:  .uleb128 .Lcst_end0-.Lcst_begin0
; CHECK:  .Lcst_begin0:
; CHECK:  .uleb128 .Lfunc_begin0-.Lfunc_begin0 # >> Call Site 1 <<
; CHECK:  .uleb128 .Lfunc_end0-.Lfunc_begin0 #   Call between .Lfunc_begin0 and .Lfunc_end0
; CHECK:  .byte   0                       #     has no landing pad
; CHECK:  .byte   0                       #   On action: cleanup
; CHECK:  .Lcst_end0:
