; RUN: opt -ejit-wrapper-async -passes=ejit-wrapper-gen -S %s | FileCheck %s

; CHECK: @__ejit_funcidx_multi_dim_entry = internal global i32 -1
; CHECK: define i32 @multi_dim_entry(i32 %idx1, i32 %idx2)
; jit_entry loads the funcIndex and, while invalid (-1), branches straight to
; the AOT fallback WITHOUT entering the taskpool.
; CHECK: jit_entry:
; CHECK: load i32, ptr @__ejit_funcidx_multi_dim_entry
; CHECK: icmp ne i32 {{.*}}, -1
; CHECK: br i1 {{.*}}, label %jit_call, label %jit_fallback
; CHECK: jit_call:
; CHECK: call i32 @ejit_taskpool_compile_or_get
; CHECK: br i1 {{.*}}, label %jit_dispatch, label %jit_fallback
; CHECK: jit_fallback:
; CHECK: load i32, ptr @data
; CHECK: load i32, ptr @data2
; CHECK: ret i32 0
; CHECK: jit_dispatch:
; CHECK: call i32 {{.*}}(i32 %idx1, i32 %idx2)
; CHECK: call void @ejit_taskpool_release_read(i32 {{.*}})

define i32 @multi_dim_entry(i32 %idx1, i32 %idx2) !ejit.metadata !0 {
entry:
  %v1 = load i32, ptr @data
  %v2 = load i32, ptr @data2
  ret i32 0
}

@data = global i32 0, !ejit.metadata !10
@data2 = global i32 0, !ejit.metadata !11

!0 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 0}, !{!"ejit_period_arr_ind", !"trp", i32 1}}
!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 16}}
!11 = distinct !{!{!"ejit_period_arr", !"trp", i32 32}}
