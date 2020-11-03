; RUN: llc -mtriple=x86_64-windows-gnu < %s | FileCheck %s

declare void @throwit()
declare void @__gxx_personality_seh0(...)
declare void @__gcc_personality_seh0(...)

define void @use_gxx_seh()
    personality void (...)* @__gxx_personality_seh0 {
entry:
  call void @throwit()
  unreachable
}

; CHECK-LABEL: use_gxx_seh:
; CHECK: .seh_proc use_gxx_seh
; CHECK-NOT: .seh_handler __gxx_personality_seh0
; CHECK: callq throwit
; CHECK: .seh_handlerdata
; CHECK: .seh_endproc

define void @use_gcc_seh()
    personality void (...)* @__gcc_personality_seh0 {
entry:
  call void @throwit()
  unreachable
}

; CHECK-LABEL: use_gcc_seh:
; CHECK: .seh_proc use_gcc_seh
; CHECK-NOT: .seh_handler __gcc_personality_seh0
; CHECK: callq throwit
; CHECK: .seh_handlerdata
; CHECK: .seh_endproc

