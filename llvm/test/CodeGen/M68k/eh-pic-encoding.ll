; RUN: llc -mtriple=m68k-linux-gnu -relocation-model=pic < %s | FileCheck %s --check-prefix=PIC
; RUN: llc -mtriple=m68k-linux-gnu -relocation-model=static < %s | FileCheck %s --check-prefix=STATIC

; Under PIC the personality, LSDA and type-info pointers must be PC-relative
; (the personality and type info through an indirect DW.ref), or a position-
; independent executable is left with absolute relocations in .eh_frame and
; .gcc_except_table, which are read-only. Without PIC they stay absolute.

; PIC-LABEL: foo:
; PIC:         .cfi_personality 155, DW.ref.__gxx_personality_v0
; PIC-NEXT:    .cfi_lsda 27, .Lexception0
; PIC:         .byte 155 ; @TType Encoding = indirect pcrel sdata4
; PIC:         DW.ref.__gxx_personality_v0:

; STATIC-LABEL: foo:
; STATIC:         .cfi_personality 0, __gxx_personality_v0
; STATIC-NEXT:    .cfi_lsda 0, .Lexception0
; STATIC:         .byte 0 ; @TType Encoding = absptr

@_ZTIi = external constant ptr

define i32 @foo() uwtable personality ptr @__gxx_personality_v0 {
entry:
  invoke void @bar()
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  ret i32 1

try.cont:
  ret i32 0
}

declare void @bar()

declare i32 @__gxx_personality_v0(...)
