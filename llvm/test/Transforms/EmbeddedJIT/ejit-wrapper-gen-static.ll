; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s

; CHECK: define void @static_only_entry()
; CHECK: jit_entry:
; CHECK: call ptr @ejit_compile_or_get(ptr {{.*}}, ptr null, i32 0, ptr null)
; CHECK: icmp eq ptr {{.*}}, null
; CHECK: br i1 {{.*}}, label %jit_fallback, label %jit_dispatch
; CHECK: jit_fallback:
; CHECK: ret void
; CHECK: jit_dispatch:
; CHECK: call void {{.*}}()
; CHECK: ret void

define void @static_only_entry() !ejit.metadata !0 {
entry:
  ret void
}

!0 = distinct !{!{!"ejit_entry"}}
