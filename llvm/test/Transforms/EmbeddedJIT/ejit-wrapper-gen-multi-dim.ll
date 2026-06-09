; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s

; CHECK: define i32 @multi_dim_entry(i32 %idx1, i32 %idx2)
; CHECK: jit_entry:
; CHECK: alloca { ptr, i8 }, i32 2
; CHECK: call ptr @ejit_compile_or_get(ptr {{.*}}, ptr {{.*}}, i32 2, ptr null)
; CHECK: br i1 {{.*}}, label %jit_fallback, label %jit_dispatch
; CHECK: jit_fallback:
; CHECK: load i32, ptr @data
; CHECK: load i32, ptr @data2
; CHECK: ret i32 0
; CHECK: jit_dispatch:
; CHECK: call i32 {{.*}}(i32 %idx1, i32 %idx2)

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
