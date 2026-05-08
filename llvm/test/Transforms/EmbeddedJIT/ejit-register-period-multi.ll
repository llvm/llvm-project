; RUN: opt -passes=ejit-register-period -S %s | FileCheck %s

; Test: Multiple period arrays + static vars + no-metadata (plain) globals.
; Verify only period/metadata-annotated globals are registered.

; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @ejit_auto_register, ptr null }]
; CHECK: declare void @ejit_register_period_array(ptr, ptr, ptr, i64)
; CHECK: declare void @ejit_register_static_var(ptr, ptr)
; CHECK: define internal void @ejit_auto_register()

; Each period array should get a registration call
; CHECK: call void @ejit_register_period_array(ptr {{.*}}, ptr {{.*}}, ptr @cells, i64 8)
; CHECK: call void @ejit_register_period_array(ptr {{.*}}, ptr {{.*}}, ptr @trps, i64 16)
; CHECK: call void @ejit_register_period_array(ptr {{.*}}, ptr {{.*}}, ptr @slices, i64 4)
; CHECK: call void @ejit_register_period_array(ptr {{.*}}, ptr {{.*}}, ptr @carriers, i64 32)

; Each static var should get a registration call
; CHECK: call void @ejit_register_static_var(ptr {{.*}}, ptr @board_cfg)
; CHECK: call void @ejit_register_static_var(ptr {{.*}}, ptr @sys_param)

; Plain global without ejit metadata should NOT be registered
; CHECK-NOT: call void @ejit_register_.*(ptr {{.*}}, ptr @plain_data

@cells = global [8 x i32] zeroinitializer, !ejit.metadata !10
@trps = global [16 x i64] zeroinitializer, !ejit.metadata !11
@slices = global [4 x i16] zeroinitializer, !ejit.metadata !12
@carriers = global [32 x i8] zeroinitializer, !ejit.metadata !13
@board_cfg = global i32 1, !ejit.metadata !14
@sys_param = global i32 314, !ejit.metadata !15
@plain_data = global [100 x i32] zeroinitializer

define void @dummy() {
  ret void
}

!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 8}}
!11 = distinct !{!{!"ejit_period_arr", !"trp", i32 16}}
!12 = distinct !{!{!"ejit_period_arr", !"slice", i32 4}}
!13 = distinct !{!{!"ejit_period_arr", !"carrier", i32 32}}
!14 = distinct !{!{!"ejit_period", !"static"}}
!15 = distinct !{!{!"ejit_period", !"static"}}
