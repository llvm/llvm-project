; RUN: opt -passes=ejit-register-period -S %s | FileCheck %s

; llvm.global_ctors appears before function definitions
; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @ejit_auto_register, ptr null }]
; CHECK: declare void @ejit_register_period_array(ptr, ptr, ptr, i64)
; CHECK: declare void @ejit_register_static_var(ptr, ptr)
; CHECK: define internal void @ejit_auto_register()
; CHECK: call void @ejit_register_period_array(ptr {{.*}}, ptr {{.*}}, ptr @cells, i64 10)
; CHECK: call void @ejit_register_static_var(ptr {{.*}}, ptr @thresh)

@cells = global [10 x i32] zeroinitializer, !ejit.metadata !10
@thresh = global i32 5, !ejit.metadata !11

define void @dummy() {
  ret void
}

!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 10}}
!11 = distinct !{!{!"ejit_period", !"static"}}
