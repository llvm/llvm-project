; Verify that the PISA intrinsics are materialized with the attributes declared
; in IntrinsicsPISA.td.

; RUN: opt -S < %s | FileCheck %s

define void @test(ptr %addr) {
; CHECK-LABEL: define void @test(
  %lane = call i32 @llvm.pisa.lane.id()
  %sg = call i32 @llvm.pisa.subgroup.size()
  %wd = call i32 @llvm.pisa.work.dim()
  %am = call i32 @llvm.pisa.activemask()
  %cas = call float @llvm.pisa.cas.fatom.f32.p0(ptr %addr, float 0.0, float 1.0, i8 2)
  call void @llvm.pisa.workgroup.barrier()
  ret void
}

; CHECK: declare float @llvm.pisa.cas.fatom.f32.p0(ptr captures(none), float, float, i8 immarg) [[ATOM:#[0-9]+]]
; CHECK: declare i32 @llvm.pisa.activemask() [[CONVMEM:#[0-9]+]]
; CHECK: declare range(i32 0, 32) i32 @llvm.pisa.lane.id() [[NOMEM:#[0-9]+]]
; CHECK: declare range(i32 32, 33) i32 @llvm.pisa.subgroup.size() [[NOMEM]]
; CHECK: declare range(i32 1, 4) i32 @llvm.pisa.work.dim() [[NOMEM]]
; CHECK: declare void @llvm.pisa.workgroup.barrier() [[CONV:#[0-9]+]]

; CHECK-DAG: attributes [[CONVMEM]] = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
; CHECK-DAG: attributes [[ATOM]] = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
; CHECK-DAG: attributes [[NOMEM]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; CHECK-DAG: attributes [[CONV]] = { convergent nounwind }

declare float @llvm.pisa.cas.fatom.f32.p0(ptr, float, float, i8 immarg)
