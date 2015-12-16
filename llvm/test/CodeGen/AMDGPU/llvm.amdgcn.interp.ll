;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=GCN %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s

;GCN-LABEL: {{^}}v_interp:
;GCN-NOT: s_wqm
;GCN: s_mov_b32 m0, s{{[0-9]+}}
;GCN: v_interp_p1_f32
;GCN: v_interp_p2_f32
define void @v_interp(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>) #0 {
main_body:
  %i = extractelement <2 x i32> %4, i32 0
  %j = extractelement <2 x i32> %4, i32 1
  %p0_0 = call float @llvm.amdgcn.interp.p1(i32 %i, i32 0, i32 0, i32 %3)
  %p1_0 = call float @llvm.amdgcn.interp.p2(float %p0_0, i32 %j, i32 0, i32 0, i32 %3)
  %p0_1 = call float @llvm.amdgcn.interp.p1(i32 %i, i32 1, i32 0, i32 %3)
  %p1_1 = call float @llvm.amdgcn.interp.p2(float %p0_1, i32 %j, i32 1, i32 0, i32 %3)
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %p0_0, float %p0_0, float %p1_1, float %p1_1)
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p1(i32, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p2(float, i32, i32, i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }
