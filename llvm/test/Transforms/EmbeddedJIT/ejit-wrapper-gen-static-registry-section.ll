; RUN: opt -ejit-wrapper-async -passes=ejit-wrapper-gen -S %s | FileCheck %s --implicit-check-not='@__ejit_registry_lifecycle' --implicit-check-not='@__ejit_registry_funcindex'

; PASS3's static-registry fallback must use linker-concatenated private section
; entries, not fixed external __ejit_registry_* arrays. Fixed external arrays
; collide when multiple TUs contain ejit_entry functions.

; CHECK: @__ejit_dimtype_cell = internal global i32 -1
; CHECK: @__ejit_funcidx_process_cell = internal global i32 -1
; CHECK: @{{.*}} = private constant [1 x {{.*}}] {{.*}}{ i32 5, ptr {{.*}}, ptr null, ptr @__ejit_dimtype_cell, i64 0 }{{.*}}, section ".ejit_period"
; CHECK: @{{.*}} = private constant [1 x {{.*}}] {{.*}}{ i32 6, ptr {{.*}}, ptr null, ptr @__ejit_funcidx_process_cell, i64 0 }{{.*}}, section ".ejit_period"
; CHECK: call void @ejit_register_lifecycle(ptr {{.*}}, ptr @__ejit_dimtype_cell)
; CHECK: call void @ejit_register_funcindex(ptr {{.*}}, ptr @__ejit_funcidx_process_cell)

@cell_data = global [16 x i32] zeroinitializer, !ejit.metadata !1

define void @process_cell(i32 %cell_idx) !ejit.metadata !0 {
entry:
  %ptr = getelementptr i32, ptr @cell_data, i32 %cell_idx
  %val = load i32, ptr %ptr
  ret void
}

!0 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 0}}
!1 = distinct !{!{!"ejit_period_arr", !"cell", i32 16}}
