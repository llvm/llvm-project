; RUN: llc < %s -relocation-model=pic -mtriple=x86_64-pc-solaris2.11 | FileCheck %s -check-prefix=X64
; RUN: llc < %s -relocation-model=pic -mtriple=i386-pc-solaris2.11 | FileCheck %s -check-prefix=X32
; PR1632

define void @_Z1fv() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_Z1gv()
          to label %return unwind label %unwind

unwind:                                           ; preds = %entry
  %exn = landingpad {ptr, i32}
            cleanup
  ret void

return:                                           ; preds = %eh_then, %entry
  ret void
}

declare void @_Z1gv()

declare i32 @__gxx_personality_v0(...)

; X64:      .size	DW.ref.__gxx_personality_v0, 8
; X64:      .quad	__gxx_personality_v0

; X32:      .size	DW.ref.__gxx_personality_v0, 4
; X32:      .long	__gxx_personality_v0

