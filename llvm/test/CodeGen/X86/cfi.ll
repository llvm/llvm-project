; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck --check-prefix=STATIC %s
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic | FileCheck --check-prefix=PIC %s

; STATIC: .cfi_personality 3, __gxx_personality_v0
; STATIC: .cfi_lsda 3, .Lexception0

; PIC: .cfi_personality 155, DW.ref.__gxx_personality_v0
; PIC: .cfi_lsda 27, .Lexception0


define void @bar() personality ptr @__gxx_personality_v0 {
entry:
  %call = invoke i32 @foo()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %exn = landingpad {ptr, i32}
            catch ptr null
  ret void
}

declare i32 @foo()

declare i32 @__gxx_personality_v0(...)
