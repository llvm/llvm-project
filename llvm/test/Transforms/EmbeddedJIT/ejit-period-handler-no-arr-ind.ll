; RUN: opt -passes=ejit-period-handler -S %s 2>&1 | FileCheck %s

; Verify diagnostic warning when ejit_period_lc has no matching ejit_period_arr_ind.
; CHECK: EJit warning: function 'lc_no_arr_ind' has ejit_period_lc("cell") but no matching ejit_period_arr_ind("cell") parameter

; Function with both lc and arr_ind — no warning expected.
; CHECK-NOT: EJit warning: function 'lc_ok' has ejit_period_lc

@cell_data = global [10 x i32] zeroinitializer, !ejit.metadata !10

; Bad: ejit_period_lc without matching ejit_period_arr_ind
define void @lc_no_arr_ind(i32 %x) !ejit.metadata !20 {
entry:
  ret void
}

; Good: ejit_period_lc WITH matching ejit_period_arr_ind
define void @lc_ok(i32 %cell_idx) !ejit.metadata !22 {
entry:
  ret void
}

!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 10}}
!20 = distinct !{!{!"ejit_period_lc", !"cell"}}
!22 = distinct !{!{!"ejit_period_lc", !"cell"}, !{!"ejit_period_arr_ind", !"cell", i32 0}}
