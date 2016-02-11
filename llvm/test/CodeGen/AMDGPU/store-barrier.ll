; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs -mattr=+load-store-opt -enable-misched < %s | FileCheck  --check-prefix=CHECK %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -mattr=+load-store-opt -enable-misched < %s | FileCheck  --check-prefix=CHECK %s

; This test is for a bug in the machine scheduler where stores without
; an underlying object would be moved across the barrier.  In this
; test, the <2 x i8> store will be split into two i8 stores, so they
; won't have an underlying object.

; CHECK-LABEL: {{^}}test:
; CHECK: ds_write_b8
; CHECK: ds_write_b8
; CHECK: s_barrier
; CHECK: s_endpgm
; Function Attrs: nounwind
define void @test(<2 x i8> addrspace(3)* nocapture %arg, <2 x i8> addrspace(1)* nocapture readonly %arg1, i32 addrspace(1)* nocapture readonly %arg2, <2 x i8> addrspace(1)* nocapture %arg3, i32 %arg4, i64 %tmp9) #0 {
bb:
  %tmp10 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp9
  %tmp13 = load i32, i32 addrspace(1)* %tmp10, align 2
  %tmp14 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(3)* %arg, i32 %tmp13
  %tmp15 = load <2 x i8>, <2 x i8> addrspace(3)* %tmp14, align 2
  %tmp16 = add i32 %tmp13, 1
  %tmp17 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(3)* %arg, i32 %tmp16
  store <2 x i8> %tmp15, <2 x i8> addrspace(3)* %tmp17, align 2
  tail call void @llvm.amdgcn.s.barrier()
  %tmp25 = load i32, i32 addrspace(1)* %tmp10, align 4
  %tmp26 = sext i32 %tmp25 to i64
  %tmp27 = sext i32 %arg4 to i64
  %tmp28 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(3)* %arg, i32 %tmp25, i32 %arg4
  %tmp29 = load i8, i8 addrspace(3)* %tmp28, align 1
  %tmp30 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %arg3, i64 %tmp26, i64 %tmp27
  store i8 %tmp29, i8 addrspace(1)* %tmp30, align 1
  %tmp32 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(3)* %arg, i32 %tmp25, i32 0
  %tmp33 = load i8, i8 addrspace(3)* %tmp32, align 1
  %tmp35 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %arg3, i64 %tmp26, i64 0
  store i8 %tmp33, i8 addrspace(1)* %tmp35, align 1
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.amdgcn.s.barrier() #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
