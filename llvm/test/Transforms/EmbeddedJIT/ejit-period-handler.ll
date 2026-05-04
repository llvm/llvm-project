; RUN: opt -passes=ejit-period-handler -S %s | FileCheck %s

; CHECK: define void @lc_single(i32 %cell_idx)
; CHECK: call void @ejit_deactivate_array(ptr {{.*}}, ptr @cell_data, i32 0)
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @cell_data, i32 0)

; Multi-lc: deactivate in order (cell, trp), activate in reverse (trp, cell)
; CHECK: define void @lc_multi(i32 %cell_idx, i32 %trp_idx)
; CHECK: call void @ejit_deactivate_array(ptr {{.*}}, ptr @cell_data, i32 0)
; CHECK: call void @ejit_deactivate_array(ptr {{.*}}, ptr @trp_data, i32 1)
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @trp_data, i32 1)
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @cell_data, i32 0)

@cell_data = global [10 x i32] zeroinitializer, !ejit.metadata !10
@trp_data = global [5 x i32] zeroinitializer, !ejit.metadata !12

define void @lc_single(i32 %cell_idx) !ejit.metadata !20 {
entry:
  ret void
}

define void @lc_multi(i32 %cell_idx, i32 %trp_idx) !ejit.metadata !22 {
entry:
  ret void
}

!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 10}}
!12 = distinct !{!{!"ejit_period_arr", !"trp", i32 5}}
!20 = distinct !{!{!"ejit_period_lc", !"cell"}, !{!"ejit_period_arr_ind", !"cell", i32 0}}
!22 = distinct !{!{!"ejit_period_lc", !"cell"}, !{!"ejit_period_lc", !"trp"}, !{!"ejit_period_arr_ind", !"cell", i32 0}, !{!"ejit_period_arr_ind", !"trp", i32 1}}
