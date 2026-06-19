; RUN: opt -ejit-wrapper-async -passes=ejit-wrapper-gen -S %s | FileCheck %s

; CHECK: define void @static_only_entry()
; CHECK: jit_entry:
; CHECK: call i32 @ejit_taskpool_compile_or_get
; CHECK: br i1 {{.*}}, label %jit_dispatch, label %jit_fallback
; CHECK: jit_fallback:
; CHECK: ret void
; CHECK: jit_dispatch:
; CHECK: call void {{.*}}()
; CHECK: call void @ejit_taskpool_release_read(i32 {{.*}})
; CHECK: ret void

define void @static_only_entry() !ejit.metadata !0 {
entry:
  ret void
}

!0 = distinct !{!{!"ejit_entry"}}
