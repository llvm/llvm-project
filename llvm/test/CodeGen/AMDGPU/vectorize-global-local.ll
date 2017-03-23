; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s
; CHECK-DAG: flat_load_dwordx4
; CHECK-DAG: flat_load_dwordx4
; CHECK-DAG: flat_load_dwordx4
; CHECK-DAG: flat_load_dwordx4
; CHECK-DAG: ds_write2_b32
; CHECK-DAG: ds_write2_b32
; CHECK-DAG: ds_write2_b32
; CHECK-DAG: ds_write2_b32
; CHECK-DAG: ds_write2_b32
; CHECK-DAG: ds_write2_b32
; CHECK-DAG: ds_write2_b32
; CHECK-DAG: ds_write2_b32

define amdgpu_kernel void @vectorize_global_local(i32 addrspace(1)* nocapture readonly %arg, i32 addrspace(3)* nocapture %arg1) {
bb:
  %tmp = load i32, i32 addrspace(1)* %arg, align 4
  store i32 %tmp, i32 addrspace(3)* %arg1, align 4
  %tmp2 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 1
  %tmp3 = load i32, i32 addrspace(1)* %tmp2, align 4
  %tmp4 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 1
  store i32 %tmp3, i32 addrspace(3)* %tmp4, align 4
  %tmp5 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 2
  %tmp6 = load i32, i32 addrspace(1)* %tmp5, align 4
  %tmp7 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 2
  store i32 %tmp6, i32 addrspace(3)* %tmp7, align 4
  %tmp8 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 3
  %tmp9 = load i32, i32 addrspace(1)* %tmp8, align 4
  %tmp10 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 3
  store i32 %tmp9, i32 addrspace(3)* %tmp10, align 4
  %tmp11 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 4
  %tmp12 = load i32, i32 addrspace(1)* %tmp11, align 4
  %tmp13 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 4
  store i32 %tmp12, i32 addrspace(3)* %tmp13, align 4
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 5
  %tmp15 = load i32, i32 addrspace(1)* %tmp14, align 4
  %tmp16 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 5
  store i32 %tmp15, i32 addrspace(3)* %tmp16, align 4
  %tmp17 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 6
  %tmp18 = load i32, i32 addrspace(1)* %tmp17, align 4
  %tmp19 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 6
  store i32 %tmp18, i32 addrspace(3)* %tmp19, align 4
  %tmp20 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 7
  %tmp21 = load i32, i32 addrspace(1)* %tmp20, align 4
  %tmp22 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 7
  store i32 %tmp21, i32 addrspace(3)* %tmp22, align 4
  %tmp23 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 8
  %tmp24 = load i32, i32 addrspace(1)* %tmp23, align 4
  %tmp25 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 8
  store i32 %tmp24, i32 addrspace(3)* %tmp25, align 4
  %tmp26 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 9
  %tmp27 = load i32, i32 addrspace(1)* %tmp26, align 4
  %tmp28 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 9
  store i32 %tmp27, i32 addrspace(3)* %tmp28, align 4
  %tmp29 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 10
  %tmp30 = load i32, i32 addrspace(1)* %tmp29, align 4
  %tmp31 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 10
  store i32 %tmp30, i32 addrspace(3)* %tmp31, align 4
  %tmp32 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 11
  %tmp33 = load i32, i32 addrspace(1)* %tmp32, align 4
  %tmp34 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 11
  store i32 %tmp33, i32 addrspace(3)* %tmp34, align 4
  %tmp35 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 12
  %tmp36 = load i32, i32 addrspace(1)* %tmp35, align 4
  %tmp37 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 12
  store i32 %tmp36, i32 addrspace(3)* %tmp37, align 4
  %tmp38 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 13
  %tmp39 = load i32, i32 addrspace(1)* %tmp38, align 4
  %tmp40 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 13
  store i32 %tmp39, i32 addrspace(3)* %tmp40, align 4
  %tmp41 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 14
  %tmp42 = load i32, i32 addrspace(1)* %tmp41, align 4
  %tmp43 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 14
  store i32 %tmp42, i32 addrspace(3)* %tmp43, align 4
  %tmp44 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 15
  %tmp45 = load i32, i32 addrspace(1)* %tmp44, align 4
  %tmp46 = getelementptr inbounds i32, i32 addrspace(3)* %arg1, i32 15
  store i32 %tmp45, i32 addrspace(3)* %tmp46, align 4
  ret void
}
