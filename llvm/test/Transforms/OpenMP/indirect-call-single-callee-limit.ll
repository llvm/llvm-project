; RUN: opt -passes=openmp-opt -openmp-opt-max-callees-for-specialization=0 -S < %s | FileCheck %s --check-prefix=INDIRECT
; RUN: opt -passes=openmp-opt -openmp-opt-max-callees-for-specialization=1 -S < %s | FileCheck %s --check-prefix=DIRECT

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
  call void %fp(i32 42), !callees !2
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 7, !"openmp", i32 50}
!1 = !{i32 7, !"openmp-device", i32 50}
!2 = !{ptr @callee}
