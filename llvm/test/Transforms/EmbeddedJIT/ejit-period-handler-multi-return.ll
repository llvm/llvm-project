; RUN: opt -passes=ejit-period-handler -S %s | FileCheck %s

; Test: Period handler with multiple return statements.
; Verify deactivate at entry, activate before EVERY return (in reverse order for multi-lc).

; --- Single lifecycle, multiple returns ---
; CHECK: define i32 @lc_multi_return(i32 %cell_idx)
; CHECK: entry:
; CHECK: call void @ejit_deactivate_array(ptr {{.*}}, ptr @cells, i32 %cell_idx)
; CHECK: icmp eq i32 %cell_idx, 0
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @cells, i32 %cell_idx)
; CHECK: ret i32 10
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @cells, i32 %cell_idx)
; CHECK: ret i32 20
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @cells, i32 %cell_idx)
; CHECK: ret i32 30

define i32 @lc_multi_return(i32 %cell_idx) !ejit.metadata !30 {
entry:
  %cmp = icmp eq i32 %cell_idx, 0
  br i1 %cmp, label %early_ret, label %mid_check

early_ret:
  ret i32 10

mid_check:
  %cmp2 = icmp eq i32 %cell_idx, 1
  br i1 %cmp2, label %mid_ret, label %normal_ret

mid_ret:
  ret i32 20

normal_ret:
  ret i32 30
}

; --- Multi-lifecycle, multiple returns with reverse-order activate ---
; CHECK: define void @lc_multi_multi_ret(i32 %cell_idx, i32 %trp_idx)
; CHECK: entry:
; CHECK: call void @ejit_deactivate_array(ptr {{.*}}, ptr @cells, i32 %cell_idx)
; CHECK: call void @ejit_deactivate_array(ptr {{.*}}, ptr @trps, i32 %trp_idx)
; CHECK: icmp eq i32 %cell_idx, 0
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @trps, i32 %trp_idx)
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @cells, i32 %cell_idx)
; CHECK: ret void
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @trps, i32 %trp_idx)
; CHECK: call void @ejit_activate_array(ptr {{.*}}, ptr @cells, i32 %cell_idx)
; CHECK: ret void

define void @lc_multi_multi_ret(i32 %cell_idx, i32 %trp_idx) !ejit.metadata !40 {
entry:
  %cmp = icmp eq i32 %cell_idx, 0
  br i1 %cmp, label %early, label %normal

early:
  ret void

normal:
  ret void
}

; --- No lifecycle function — pass should do nothing ---
; CHECK: define void @not_lc(i32 %idx)
; CHECK-NOT: ejit_deactivate_array
; CHECK-NOT: ejit_activate_array
; CHECK: ret void

define void @not_lc(i32 %idx) {
entry:
  ret void
}

@cells = global [10 x i32] zeroinitializer, !ejit.metadata !10
@trps = global [10 x i32] zeroinitializer, !ejit.metadata !11

!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 10}}
!11 = distinct !{!{!"ejit_period_arr", !"trp", i32 10}}
!30 = distinct !{!{!"ejit_period_lc", !"cell"}, !{!"ejit_period_arr_ind", !"cell", i32 0}}
!40 = distinct !{!{!"ejit_period_lc", !"cell"}, !{!"ejit_period_lc", !"trp"}, !{!"ejit_period_arr_ind", !"cell", i32 0}, !{!"ejit_period_arr_ind", !"trp", i32 1}}
