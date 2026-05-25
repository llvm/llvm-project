; RUN: llc -global-isel=0 -mtriple=amdgpu12.50 -stop-after=machine-scheduler \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.50 -stop-after=machine-scheduler \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

declare void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1), ptr addrspace(3), i32, i32)
declare void @llvm.amdgcn.tensor.store.from.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)

; CHECK-LABEL: cross_counter:%bb.0
; CHECK: SU([[LOAD:[0-9]+]]): GLOBAL_LOAD_ASYNC_TO_LDS_B32
; CHECK-SAME: implicit $asynccnt{{.*}}implicit $asyncmarker
; CHECK:   Successors:
; CHECK-NEXT:     SU({{[0-9]+}}): Anti Latency=0
; CHECK-NEXT:     SU([[MARK:[0-9]+]]): Anti Latency=0
; CHECK: SU([[MARK]]): ASYNCMARK
; CHECK-SAME: implicit-def dead $asyncmarker, implicit $asyncmarker
; CHECK:   Successors:
; CHECK:     SU([[WAIT:[0-9]+]]): Data Latency=1 Reg=$asyncmarker
; CHECK:     SU([[TENSOR:[0-9]+]]): Data Latency=1 Reg=$asyncmarker
; CHECK: SU([[TENSOR]]): TENSOR_STORE_FROM_LDS_d2
; CHECK-SAME: implicit $tensorcnt, implicit $asyncmarker
; CHECK:   Successors:
; CHECK:     SU([[WAIT]]): Anti Latency=0
; CHECK: SU([[WAIT]]): WAIT_ASYNCMARK
; CHECK-SAME: implicit-def dead $asyncmarker, implicit $asyncmarker
; CHECK:   Predecessors:
; CHECK:     SU([[TENSOR]]): Anti Latency=0
; CHECK:     SU([[MARK]]): Data Latency=1 Reg=$asyncmarker
; CHECK:     SU([[LOAD]]): Anti Latency=0
define amdgpu_ps void @cross_counter() {
  call void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1) poison, ptr addrspace(3) poison, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> poison, <8 x i32> poison, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  ret void
}
