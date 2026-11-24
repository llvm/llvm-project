; SjLjEHPrepare only runs for the "sjlj" module flag, even though every
; function has an SjLj personality.

; RUN: split-file %s %t

; "sjlj" flag.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=sjlj-eh-prepare \
; RUN:   %t/sjlj.ll -o - | FileCheck %s --check-prefix=SJLJ

; No flag, so the triple's default DwarfCFI model.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=sjlj-eh-prepare \
; RUN:   %t/default.ll -o - | FileCheck %s --check-prefix=NOSJLJ

; "dwarf" flag.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=sjlj-eh-prepare \
; RUN:   %t/dwarf.ll -o - | FileCheck %s --check-prefix=NOSJLJ

;--- sjlj.ll
declare void @may_throw()
declare i32 @__gxx_personality_sj0(...)

define void @f() personality ptr @__gxx_personality_sj0 {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"sjlj"}

; SJLJ: call void @_Unwind_SjLj_Register
; SJLJ: call void @llvm.eh.sjlj.callsite

;--- default.ll
declare void @may_throw()
declare i32 @__gxx_personality_sj0(...)

define void @g() personality ptr @__gxx_personality_sj0 {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

; NOSJLJ-NOT: llvm.eh.sjlj

;--- dwarf.ll
declare void @may_throw()
declare i32 @__gxx_personality_sj0(...)

define void @h() personality ptr @__gxx_personality_sj0 {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"dwarf"}
