;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI --check-prefix=GCN %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=VI --check-prefix=GCN %s

;GCN: v_mbcnt_lo_u32_b32_e64
;SI: v_mbcnt_hi_u32_b32_e32
;VI: v_mbcnt_hi_u32_b32_e64

define amdgpu_ps void @main(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg) {
main_body:
  %4 = call i32 @llvm.SI.tid()
  %5 = bitcast i32 %4 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %5, float %5, float %5, float %5)
  ret void
}

declare i32 @llvm.SI.tid() readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
