; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck %s
; CHECK-LABEL: .LBB0_1:
; CHECK: v_lshrrev_b32_e64 [[V:v[0-9]+]], 5, s33
; CHECK: v_add_nc_u32_e32 [[V]], 12, [[V]]
; CHECK: v_readfirstlane_b32 [[S:s[0-9]+]], [[V]]
; CHECK: s_cselect_b32 {{s[0-9]+}}, 0, [[S]]

%struct.wobble = type { %struct.quux }
%struct.quux = type { float, float, float }

declare %struct.wobble @foo(%struct.quux)

define void @wobble() #0 {
bb:
  %alloca = alloca %struct.wobble, align 4, addrspace(5)
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phi = phi i32 [ 0, %bb ], [ 1, %bb1 ]
  store i32 0, ptr addrspacecast (ptr addrspace(5) null to ptr), align 4
  %getelementptr = getelementptr i8, ptr addrspace(5) %alloca, i32 4
  %icmp = icmp eq i32 %phi, 0
  %load = load float, ptr addrspace(5) null, align 2147483648
  %load2 = load float, ptr addrspace(5) %alloca, align 4
  %select = select i1 %icmp, float %load, float %load2
  %insertvalue = insertvalue %struct.quux zeroinitializer, float %select, 0
  %load3 = load float, ptr addrspace(5) inttoptr (i32 4 to ptr addrspace(5)), align 4
  %load4 = load float, ptr addrspace(5) %getelementptr, align 4
  %select5 = select i1 %icmp, float %load3, float %load4
  %insertvalue6 = insertvalue %struct.quux %insertvalue, float %select5, 1
  %call = call %struct.wobble @foo(%struct.quux %insertvalue6)
  br label %bb1
}

attributes #0 = { "target-cpu"="gfx1030" }
