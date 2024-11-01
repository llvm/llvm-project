; RUN:  llc -march=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN:  llc -march=amdgcn -mcpu=gfx900 -stop-after=amdgpu-isel < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX900 %s

; GCN-LABEL: name: s_abs_i32
; GCN: S_ABS_I32
define amdgpu_kernel void @s_abs_i32(ptr addrspace(1) %out, i32 %val) nounwind {
  %neg = sub i32 0, %val
  %cond = icmp sgt i32 %val, %neg
  %res = select i1 %cond, i32 %val, i32 %neg
  %res2 = add i32 %res, 2
  store i32 %res2, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: name: v_abs_i32
; SI:  V_SUB_CO_U32_e64
; GFX900: V_SUB_U32_e64
; GCN: V_MAX_I32_e64
define amdgpu_kernel void @v_abs_i32(ptr addrspace(1) %out, ptr addrspace(1) %src) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds i32, ptr addrspace(1) %src, i32 %tid
  %val = load i32, ptr addrspace(1) %gep.in, align 4
  %neg = sub i32 0, %val
  %cond = icmp sgt i32 %val, %neg
  %res = select i1 %cond, i32 %val, i32 %neg
  %res2 = add i32 %res, 2
  store i32 %res2, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: name: s_abs_v2i32
; GCN: S_ABS_I32
; GCN: S_ABS_I32
define amdgpu_kernel void @s_abs_v2i32(ptr addrspace(1) %out, <2 x i32> %val) nounwind {
  %z0 = insertelement <2 x i32> undef, i32 0, i32 0
  %z1 = insertelement <2 x i32> %z0, i32 0, i32 1
  %t0 = insertelement <2 x i32> undef, i32 2, i32 0
  %t1 = insertelement <2 x i32> %t0, i32 2, i32 1
  %neg = sub <2 x i32> %z1, %val
  %cond = icmp sgt <2 x i32> %val, %neg
  %res = select <2 x i1> %cond, <2 x i32> %val, <2 x i32> %neg
  %res2 = add <2 x i32> %res, %t1
  store <2 x i32> %res2, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: name: v_abs_v2i32
; SI:  V_SUB_CO_U32_e64
; GFX900: V_SUB_U32_e64
; GCN: V_MAX_I32_e64
; GCN: V_MAX_I32_e64
define amdgpu_kernel void @v_abs_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %src) nounwind {
  %z0 = insertelement <2 x i32> undef, i32 0, i32 0
  %z1 = insertelement <2 x i32> %z0, i32 0, i32 1
  %t0 = insertelement <2 x i32> undef, i32 2, i32 0
  %t1 = insertelement <2 x i32> %t0, i32 2, i32 1
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x i32>, ptr addrspace(1) %src, i32 %tid
  %val = load <2 x i32>, ptr addrspace(1) %gep.in, align 4
  %neg = sub <2 x i32> %z1, %val
  %cond = icmp sgt <2 x i32> %val, %neg
  %res = select <2 x i1> %cond, <2 x i32> %val, <2 x i32> %neg
  %res2 = add <2 x i32> %res, %t1
  store <2 x i32> %res2, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
