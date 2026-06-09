; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s

; Test: Edge cases for wrapper generation:
;   1. 0-dim entry (static period only, no dims)
;   2. 4-dim entry (max dimension support)
;   3. Basic noinline attached

; --- 0-dim: entry with static only ---
; CHECK-LABEL: define void @static_only_entry()
; CHECK: jit_entry:
; CHECK: call ptr @ejit_compile_or_get(ptr {{.*}}, ptr null, i32 0, ptr null)
; CHECK: jit_fallback:
; CHECK: jit_dispatch:

define void @static_only_entry() !ejit.metadata !0 {
entry:
  %v = load i32, ptr @static_cfg
  ret void
}

; --- 4-dim: max dimension support ---
; CHECK-LABEL: define void @four_dim_entry(i32 %d1, i32 %d2, i32 %d3, i32 %d4)
; CHECK: jit_entry:
; CHECK: alloca { ptr, i8 }, i32 4
; CHECK: call ptr @ejit_compile_or_get(ptr {{.*}}, ptr {{.*}}, i32 4, ptr null)

define void @four_dim_entry(i32 %d1, i32 %d2, i32 %d3, i32 %d4) !ejit.metadata !10 {
entry:
  ret void
}

@static_cfg = global i32 42, !ejit.metadata !90

!0 = distinct !{!{!"ejit_entry"}}
!10 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 0}, !{!"ejit_period_arr_ind", !"trp", i32 1}, !{!"ejit_period_arr_ind", !"slice", i32 2}, !{!"ejit_period_arr_ind", !"carrier", i32 3}}
!90 = distinct !{!{!"ejit_period", !"static"}}
