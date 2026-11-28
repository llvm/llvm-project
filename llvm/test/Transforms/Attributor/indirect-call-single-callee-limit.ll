; RUN: opt -passes=attributor -attributor-max-specializations-per-call-base=0 -S < %s | FileCheck %s --check-prefix=INDIRECT
; RUN: opt -passes=attributor -attributor-max-specializations-per-call-base=1 -S < %s | FileCheck %s --check-prefix=DIRECT

; A single known callee must respect the specialization policy too.

declare void @callee(i32)

define void @single_callee(ptr %fp) {
; INDIRECT-LABEL: define void @single_callee(
; INDIRECT:         call void %fp(i32 42){{.*}}!callees
; INDIRECT-NEXT:    ret void
;
; DIRECT-LABEL: define void @single_callee(
; DIRECT:         call void @callee(i32 42)
; DIRECT-NEXT:    ret void
  call void %fp(i32 42), !callees !0
  ret void
}

!0 = !{ptr @callee}
