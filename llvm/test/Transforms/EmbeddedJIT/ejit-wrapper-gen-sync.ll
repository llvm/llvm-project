; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s

; The temporary default synchronous mode keeps the established
; ejit_compile_or_get ABI, but uses the registration-assigned dense funcIndex
; rather than the old function-name hash.

; CHECK: @__ejit_funcidx_sync_entry = internal global i32 -1
; CHECK-NOT: @__ejit_dimtype_cell

; CHECK-LABEL: define i32 @sync_entry(i32 %cell)
; CHECK: jit_entry:
; CHECK: [[FUNCIDX:%.*]] = load i32, ptr @__ejit_funcidx_sync_entry
; CHECK: icmp ne i32 [[FUNCIDX]], -1
; CHECK: br i1 {{.*}}, label %jit_call, label %jit_fallback
; CHECK: jit_call:
; CHECK: [[FUNC64:%.*]] = zext i32 [[FUNCIDX]] to i64
; CHECK: [[KEYBASE:%.*]] = shl i64 [[FUNC64]], 32
; CHECK: [[CELL8:%.*]] = trunc i32 %cell to i8
; CHECK: [[CELL64:%.*]] = zext i8 [[CELL8]] to i64
; CHECK: [[KEY:%.*]] = or i64 [[KEYBASE]], [[CELL64]]
; CHECK: [[FN:%.*]] = call ptr @ejit_compile_or_get(i64 [[KEY]], ptr null)
; CHECK: icmp ne ptr [[FN]], null
; CHECK: jit_dispatch:
; CHECK: call i32 [[FN]](i32 %cell)
; CHECK-NOT: call void @ejit_taskpool_release_read
; CHECK-NOT: call i32 @ejit_taskpool_compile_or_get

define i32 @sync_entry(i32 %cell) !ejit.metadata !0 {
entry:
  %v = load i32, ptr @data
  ret i32 %v
}

@data = global i32 7, !ejit.metadata !1

!0 = distinct !{!{!"ejit_entry"},
                 !{!"ejit_period_arr_ind", !"cell", i32 0}}
!1 = distinct !{!{!"ejit_period_arr", !"cell", i32 16}}
