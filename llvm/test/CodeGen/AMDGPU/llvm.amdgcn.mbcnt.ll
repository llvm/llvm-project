;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI --check-prefix=GCN %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=VI --check-prefix=GCN %s

;GCN-LABEL: {{^}}mbcnt_intrinsics:
;GCN: v_mbcnt_lo_u32_b32_e64 [[LO:v[0-9]+]], -1, 0
;SI: v_mbcnt_hi_u32_b32_e32 {{v[0-9]+}}, -1, [[LO]]
;VI: v_mbcnt_hi_u32_b32_e64 {{v[0-9]+}}, -1, [[LO]]

define amdgpu_ps void @mbcnt_intrinsics(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg) {
main_body:
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #1
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo) #1
  %4 = bitcast i32 %hi to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %4, float %4, float %4, float %4)
  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1

declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #1 = { nounwind readnone }
