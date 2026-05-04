; RUN: opt -passes=ejit-aot-module -S %s | FileCheck %s

; Verify PASS2: period array registration
; CHECK: declare void @ejit_register_period_array(
; CHECK: declare void @ejit_register_static_var(
; CHECK: define internal void @ejit_auto_register()
; CHECK: call void @ejit_register_period_array(ptr {{.*}} @.str.{{.*}} @.str.{{.*}} ptr {{.*}} @cell_data{{.*}} i64 16)
; CHECK: call void @ejit_register_static_var(ptr {{.*}} @.str.{{.*}} ptr {{.*}} @static_config

; Verify PASS3: wrapper gen for ejit_entry
; CHECK: declare ptr @ejit_compile_or_get(
; Verify the ejit_entry function has jit_entry/jit_fallback/jit_dispatch blocks
; CHECK: jit_entry:
; CHECK: call ptr @ejit_compile_or_get(
; CHECK: icmp eq ptr
; CHECK: br i1 {{.*}} {{.*}} %jit_fallback{{.*}} %jit_dispatch
; CHECK: jit_dispatch:
; CHECK: ptrtoint ptr {{.*}} to void (
; CHECK: call void
; CHECK: jit_fallback:
; CHECK: ret void

; Verify PASS4: lifecycle handler deactivate/activate
; CHECK: declare void @ejit_deactivate_array(
; CHECK: declare void @ejit_activate_array(
; CHECK: call void @ejit_deactivate_array(ptr {{.*}} @.str.{{.*}} ptr {{.*}} @cell_data{{.*}} i32 0)
; CHECK: call void @ejit_activate_array(ptr {{.*}} @.str.{{.*}} ptr {{.*}} @cell_data{{.*}} i32 0)

@cell_data = global [16 x i32] zeroinitializer, !ejit.metadata !10
@static_config = global i32 42, !ejit.metadata !11

define void @process_cell(i32 %cell_idx) !ejit.metadata !20 {
entry:
  %ptr = getelementptr i32, ptr @cell_data, i32 %cell_idx
  %val = load i32, ptr %ptr
  %sum = add i32 %val, 42
  ret void
}

define void @lc_handler(i32 %cell_idx) !ejit.metadata !30 {
entry:
  ret void
}

!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 16}}
!11 = distinct !{!{!"ejit_period", !"static"}}
!20 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 0}}
!30 = distinct !{!{!"ejit_period_lc", !"cell"}, !{!"ejit_period_arr_ind", !"cell", i32 0}}
