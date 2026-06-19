; RUN: opt -ejit-wrapper-async -passes=ejit-aot-module -S %s | FileCheck %s

; Test: Full AOT pipeline on a comprehensive module.

; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @ejit_auto_register, ptr null }]

; Per-lifecycle dimType globals for the two lifecycles the entry functions use
; (slice is declared but never referenced by an ejit_period_arr_ind, so it gets
; no global). The wrapper loads these slots instead of baking constants.
; CHECK: @__ejit_dimtype_cell = internal global i32 -1
; CHECK: @__ejit_dimtype_trp = internal global i32 -1

; Per-function dense funcIndex globals (filled at registration).
; CHECK: @__ejit_funcidx_process_cell = internal global i32 -1
; CHECK: @__ejit_funcidx_process_static = internal global i32 -1

; CHECK: define{{.*}}@process_cell
; CHECK: jit_entry:
; CHECK: load i32, ptr @__ejit_funcidx_process_cell
; CHECK: icmp ne i32 {{.*}}, -1
; CHECK: br i1 {{.*}}, label %jit_call, label %jit_fallback
; CHECK: jit_call:
; CHECK: load i32, ptr @__ejit_dimtype_cell
; CHECK: load i32, ptr @__ejit_dimtype_trp
; CHECK: call i32 @ejit_taskpool_compile_or_get(i32 {{.*}}, ptr {{.*}}, i32 2, ptr {{.*}}, ptr {{.*}})

; CHECK: define{{.*}}@process_static
; CHECK: jit_entry:
; CHECK: load i32, ptr @__ejit_funcidx_process_static
; CHECK: call i32 @ejit_taskpool_compile_or_get(i32 {{.*}}, ptr {{.*}}, i32 0, ptr {{.*}}, ptr {{.*}})

; CHECK: define{{.*}}@update_config
; CHECK: call void @ejit_deactivate_array
; CHECK: call void @ejit_deactivate_array
; CHECK: call void @ejit_activate_array
; CHECK: call void @ejit_activate_array

; CHECK: define internal void @ejit_auto_register()
; CHECK: call void @ejit_register_period_array(ptr {{.*}}, ptr {{.*}}, ptr @cell_cfg, i64 10)
; CHECK: call void @ejit_register_period_array(ptr {{.*}}, ptr {{.*}}, ptr @trp_cfg, i64 8)
; CHECK: call void @ejit_register_period_array(ptr {{.*}}, ptr {{.*}}, ptr @slice_cfg, i64 4)
; CHECK: call void @ejit_register_static_var(ptr {{.*}}, ptr @board_cfg)
; CHECK: call void @ejit_register_static_var(ptr {{.*}}, ptr @sys_thresh)
; CHECK: call void @ejit_register_lifecycle(ptr {{.*}}, ptr @__ejit_dimtype_cell)
; CHECK: call void @ejit_register_lifecycle(ptr {{.*}}, ptr @__ejit_dimtype_trp)
; CHECK: call void @ejit_register_funcindex(ptr {{.*}}, ptr @__ejit_funcidx_process_cell)
; CHECK: call void @ejit_register_funcindex(ptr {{.*}}, ptr @__ejit_funcidx_process_static)

; CHECK-DAG: declare i32 @ejit_taskpool_compile_or_get(i32, ptr, i32, ptr, ptr)
; CHECK-DAG: declare void @ejit_taskpool_release_read(i32)
; CHECK-DAG: declare void @ejit_register_lifecycle(ptr, ptr)
; CHECK-DAG: declare void @ejit_register_funcindex(ptr, ptr)


@cell_cfg = global [10 x i32] zeroinitializer, !ejit.metadata !10
@trp_cfg = global [8 x i32] zeroinitializer, !ejit.metadata !11
@slice_cfg = global [4 x i32] zeroinitializer, !ejit.metadata !12
@board_cfg = global i32 1, !ejit.metadata !13
@sys_thresh = global i32 5, !ejit.metadata !14

define i32 @process_cell(i32 %cell_idx, i32 %trp_idx) !ejit.metadata !20 {
entry:
  %ptr = getelementptr i32, ptr @cell_cfg, i32 %cell_idx
  %val = load i32, ptr %ptr
  ret i32 %val
}

define i32 @process_static() !ejit.metadata !21 {
entry:
  %v = load i32, ptr @board_cfg
  ret i32 %v
}

define void @update_config(i32 %cell_idx, i32 %trp_idx) !ejit.metadata !30 {
entry:
  store i32 0, ptr @cell_cfg
  store i32 0, ptr @trp_cfg
  ret void
}

!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 10}}
!11 = distinct !{!{!"ejit_period_arr", !"trp", i32 8}}
!12 = distinct !{!{!"ejit_period_arr", !"slice", i32 4}}
!13 = distinct !{!{!"ejit_period", !"static"}}
!14 = distinct !{!{!"ejit_period", !"static"}}
!20 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 0}, !{!"ejit_period_arr_ind", !"trp", i32 1}}
!21 = distinct !{!{!"ejit_entry"}}
!30 = distinct !{!{!"ejit_period_lc", !"cell"}, !{!"ejit_period_lc", !"trp"}, !{!"ejit_period_arr_ind", !"cell", i32 0}, !{!"ejit_period_arr_ind", !"trp", i32 1}}
