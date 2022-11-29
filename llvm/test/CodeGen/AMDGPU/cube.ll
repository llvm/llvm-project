; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare float @llvm.amdgcn.cubeid(float, float, float) #0
declare float @llvm.amdgcn.cubesc(float, float, float) #0
declare float @llvm.amdgcn.cubetc(float, float, float) #0
declare float @llvm.amdgcn.cubema(float, float, float) #0

; GCN-LABEL: {{^}}cube:
; GCN-DAG: v_cubeid_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_cubesc_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_cubetc_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_cubema_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: _store_dwordx4
define amdgpu_kernel void @cube(ptr addrspace(1) %out, float %a, float %b, float %c) #1 {
  %cubeid = call float @llvm.amdgcn.cubeid(float %a, float %b, float %c)
  %cubesc = call float @llvm.amdgcn.cubesc(float %a, float %b, float %c)
  %cubetc = call float @llvm.amdgcn.cubetc(float %a, float %b, float %c)
  %cubema = call float @llvm.amdgcn.cubema(float %a, float %b, float %c)

  %vec0 = insertelement <4 x float> undef, float %cubeid, i32 0
  %vec1 = insertelement <4 x float> %vec0, float %cubesc, i32 1
  %vec2 = insertelement <4 x float> %vec1, float %cubetc, i32 2
  %vec3 = insertelement <4 x float> %vec2, float %cubema, i32 3
  store <4 x float> %vec3, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
