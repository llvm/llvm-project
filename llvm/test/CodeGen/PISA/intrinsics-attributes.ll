; Verify that the PISA work-item / control intrinsics are materialized with the
; attributes declared in IntrinsicsPISA.td.

; RUN: opt -S < %s | FileCheck %s

define void @test() {
; CHECK-LABEL: define void @test()
  %lane = call i32 @llvm.pisa.lane.id()
  %sg = call i32 @llvm.pisa.subgroup.size()
  %wd = call i32 @llvm.pisa.work.dim()
  %am = call i32 @llvm.pisa.activemask()
  call void @llvm.pisa.workgroup.barrier()
  ret void
}

; CHECK: declare i32 @llvm.pisa.activemask() [[CONVMEM:#[0-9]+]]
; CHECK: declare range(i32 0, 32) i32 @llvm.pisa.lane.id() [[NOMEM:#[0-9]+]]
; CHECK: declare range(i32 32, 33) i32 @llvm.pisa.subgroup.size() [[NOMEM]]
; CHECK: declare range(i32 1, 4) i32 @llvm.pisa.work.dim() [[NOMEM]]
; CHECK: declare void @llvm.pisa.workgroup.barrier() [[CONV:#[0-9]+]]

; CHECK-DAG: attributes [[CONVMEM]] = { convergent nounwind memory(none) }
; CHECK-DAG: attributes [[NOMEM]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; CHECK-DAG: attributes [[CONV]] = { convergent nounwind }
