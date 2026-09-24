; Check that the "exception-model" flag is respected. The pass should
; only only when the module selects the SjLj model, and is a no-op for
; any other model.

; RUN: split-file %s %t
; RUN: opt -passes=sjlj-eh-prepare -S %t/sjlj.ll  | FileCheck --check-prefix=SJLJ %s
; RUN: opt -passes=sjlj-eh-prepare -S %t/other.ll | FileCheck --check-prefix=OTHER %s

;--- sjlj.ll
; SJLJ-LABEL: define void @f() personality ptr @__gxx_personality_sj0 {
; SJLJ-NEXT:  entry:
; SJLJ-NEXT:    %fn_context = alloca
; SJLJ:         call void @_Unwind_SjLj_Register(
define void @f() personality ptr @__gxx_personality_sj0 {
entry:
  invoke void @g()
          to label %cont unwind label %lpad

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %lp

cont:
  ret void
}

declare void @g()
declare i32 @__gxx_personality_sj0(...)

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"sjlj"}

;--- other.ll

; A non-SjLj model, so the pass is a no-op. The entry block still
; branches straight to the invoke.

; OTHER-LABEL: define void @f() personality ptr @__gxx_personality_sj0 {
; OTHER-NEXT:  entry:
; OTHER-NEXT:    invoke void @g()
define void @f() personality ptr @__gxx_personality_sj0 {
entry:
  invoke void @g()
          to label %cont unwind label %lpad

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %lp

cont:
  ret void
}

declare void @g()
declare i32 @__gxx_personality_sj0(...)

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"dwarf"}
