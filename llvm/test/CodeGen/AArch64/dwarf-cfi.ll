; RUN: llc -mtriple aarch64-windows-gnu -exception-model=dwarf -filetype=asm -o - %s | FileCheck %s

define void @_Z1gv() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_Z1fv()
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = tail call ptr @__cxa_begin_catch(ptr %1) #2
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret void
}

declare void @_Z1fv()

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

; CHECK-LABEL: _Z1gv:
; CHECK:    .cfi_startproc
; CHECK:    .cfi_personality 0, __gxx_personality_v0
; CHECK:    .cfi_lsda 0, .Lexception0
; CHECK:    str x30, [sp, #-16]!
; CHECK:    .cfi_def_cfa_offset 16
; CHECK:    .cfi_offset w30, -16
; CHECK:    ldr x30, [sp], #16
; CHECK:    .cfi_endproc
