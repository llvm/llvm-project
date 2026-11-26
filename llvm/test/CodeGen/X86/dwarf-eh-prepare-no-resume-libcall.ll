; RUN: not opt -mtriple=x86_64-linux-gnu -dwarf-eh-prepare -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -mtriple=x86_64-linux-gnu -passes='require<libcall-lowering-info>,dwarf-eh-prepare' -disable-output %s 2>&1 | FileCheck %s

; No _Unwind_Resume is available without an exception model.

; CHECK: error: no libcall available for resume

declare void @may_throw()
declare i32 @__gxx_personality_v0(...)

define void @resume() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw()
          to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  call void @may_throw()
  resume { ptr, i32 } %lp
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"none"}
