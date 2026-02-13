; RUN: llc -mtriple=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -stop-after=amdgpu-isel < %s | FileCheck -check-prefixes=GCN,FP16 %s
; RUN: llc -mtriple=amdgcn -enable-new-pm -stop-after=amdgpu-isel < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -enable-new-pm -stop-after=amdgpu-isel < %s | FileCheck -check-prefixes=GCN,FP16 %s


define amdgpu_kernel void @divergent_fneg_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fneg_f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: V_XOR_B32_e64 killed %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds float, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %val = load volatile float, ptr addrspace(1) %in.gep
  %fneg = fneg float %val
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @uniform_fneg_f32(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %idx) {
; GCN-LABEL: name:            uniform_fneg_f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: S_XOR_B32 killed %{{[0-9]+}}, killed %[[REG]]

  %in.gep = getelementptr inbounds float, ptr addrspace(1) %in, i64 %idx
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idx
  %val = load volatile float, ptr addrspace(1) %in.gep
  %fneg = fneg float %val
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @divergent_fabs_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fabs_f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 2147483647
; GCN: V_AND_B32_e64 killed %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds float, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %val = load volatile float, ptr addrspace(1) %in.gep
  %fabs = call float @llvm.fabs.f32(float %val)
  store float %fabs, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @uniform_fabs_f32(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %idx) {
; GCN-LABEL: name:            uniform_fabs_f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 2147483647
; GCN: S_AND_B32 killed %{{[0-9]+}}, killed %[[REG]]

  %in.gep = getelementptr inbounds float, ptr addrspace(1) %in, i64 %idx
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idx
  %val = load volatile float, ptr addrspace(1) %in.gep
  %fabs = call float @llvm.fabs.f32(float %val)
  store float %fabs, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @divergent_fneg_fabs_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fneg_fabs_f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: V_OR_B32_e64 killed %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds float, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %val = load volatile float, ptr addrspace(1) %in.gep
  %fabs = call float @llvm.fabs.f32(float %val)
  %fneg = fneg float %fabs
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @uniform_fneg_fabs_f32(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %idx) {
; GCN-LABEL: name:            uniform_fneg_fabs_f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: S_OR_B32 killed %{{[0-9]+}}, killed %[[REG]]

  %in.gep = getelementptr inbounds float, ptr addrspace(1) %in, i64 %idx
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idx
  %val = load volatile float, ptr addrspace(1) %in.gep
  %fabs = call float @llvm.fabs.f32(float %val)
  %fneg = fneg float %fabs
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}


define amdgpu_kernel void @divergent_fabs_f16(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; GCN-LABEL: name:            divergent_fabs_f16
; GCN-LABEL: bb.0 (%ir-block.0)
; FP16: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 32767
; FP16: V_AND_B32_e64 killed %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds half, ptr addrspace(1) %in, i64 %tid.ext
  %val = load volatile half, ptr addrspace(1) %in.gep
  %fabs = call half @llvm.fabs.f16(half %val)
  store half %fabs, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_fabs_f16(ptr addrspace(1) %in, ptr addrspace(1) %out, i64 %idx) {
; GCN-LABEL: name:            uniform_fabs_f16
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 32767
; GCN: S_AND_B32 killed %{{[0-9]+}}, killed %[[REG]]

  %in.gep = getelementptr inbounds half, ptr addrspace(1) %in, i64 %idx
  %val = load volatile half, ptr addrspace(1) %in.gep
  %fabs = call half @llvm.fabs.f16(half %val)
  store half %fabs, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @divergent_fneg_f16(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; GCN-LABEL: name:            divergent_fneg_f16
; GCN-LABEL: bb.0 (%ir-block.0)
; FP16: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 32768
; FP16: V_XOR_B32_e64 killed %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds half, ptr addrspace(1) %in, i64 %tid.ext
  %val = load volatile half, ptr addrspace(1) %in.gep
  %fneg = fneg half %val
  store half %fneg, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_fneg_f16(ptr addrspace(1) %in, ptr addrspace(1) %out, i64 %idx) {
; GCN-LABEL: name:            uniform_fneg_f16
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 32768
; GCN: S_XOR_B32 killed %{{[0-9]+}}, killed %[[REG]]

  %in.gep = getelementptr inbounds half, ptr addrspace(1) %in, i64 %idx
  %val = load volatile half, ptr addrspace(1) %in.gep
  %fneg = fneg half %val
  store half %fneg, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @divergent_fneg_fabs_f16(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; GCN-LABEL: name:            divergent_fneg_fabs_f16
; GCN-LABEL: bb.0 (%ir-block.0)
; FP16: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 32768
; FP16: V_OR_B32_e64 killed %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds half, ptr addrspace(1) %in, i64 %tid.ext
  %val = load volatile half, ptr addrspace(1) %in.gep
  %fabs = call half @llvm.fabs.f16(half %val)
  %fneg = fneg half %fabs
  store half %fneg, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_fneg_fabs_f16(ptr addrspace(1) %in, ptr addrspace(1) %out, i64 %idx) {
; GCN-LABEL: name:            uniform_fneg_fabs_f16
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 32768
; GCN: S_OR_B32 killed %{{[0-9]+}}, killed %[[REG]]

  %in.gep = getelementptr inbounds half, ptr addrspace(1) %in, i64 %idx
  %val = load volatile half, ptr addrspace(1) %in.gep
  %fabs = call half @llvm.fabs.f16(half %val)
  %fneg = fneg half %fabs
  store half %fneg, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @divergent_fneg_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fneg_v2f16
; GCN-LABEL: bb.0 (%ir-block.0)
; FP16: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147450880
; FP16: V_XOR_B32_e64 killed %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %tid
  %gep.out = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %tid
  %val = load <2 x half>, ptr addrspace(1) %gep.in, align 2
  %fneg = fneg <2 x half> %val
  store <2 x half> %fneg, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @uniform_fneg_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %idx) {
; GCN-LABEL: name:            uniform_fneg_v2f16
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147450880
; GCN: S_XOR_B32 killed %{{[0-9]+}}, killed %[[REG]]

  %gep.in = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %idx
  %gep.out = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %idx
  %val = load <2 x half>, ptr addrspace(1) %gep.in, align 2
  %fneg = fneg <2 x half> %val
  store <2 x half> %fneg, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @divergent_fabs_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fabs_v2f16
; GCN-LABEL: bb.0 (%ir-block.0)
; FP16: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 2147450879
; FP16: V_AND_B32_e64 killed %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %tid
  %gep.out = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %tid
  %val = load <2 x half>, ptr addrspace(1) %gep.in, align 2
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  store <2 x half> %fabs, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @uniform_fabs_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %idx) {
; GCN-LABEL: name:            uniform_fabs_v2f16
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 2147450879
; GCN: S_AND_B32 killed %{{[0-9]+}}, killed %[[REG]]

  %gep.in = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %idx
  %gep.out = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %idx
  %val = load <2 x half>, ptr addrspace(1) %gep.in, align 2
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  store <2 x half> %fabs, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @divergent_fneg_fabs_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fneg_fabs_v2f16
; GCN-LABEL: bb.0 (%ir-block.0)
; FP16: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147450880
; FP16: V_OR_B32_e64 killed %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %tid
  %gep.out = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %tid
  %val = load <2 x half>, ptr addrspace(1) %gep.in, align 2
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %fneg = fneg <2 x half> %fabs
  store <2 x half> %fneg, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @uniform_fneg_fabs_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %idx) {
; GCN-LABEL: name:            uniform_fneg_fabs_v2f16
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147450880
; GCN: S_OR_B32 killed %{{[0-9]+}}, killed %[[REG]]

  %gep.in = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %idx
  %gep.out = getelementptr inbounds <2 x half>, ptr addrspace(1) %in, i32 %idx
  %val = load <2 x half>, ptr addrspace(1) %gep.in, align 2
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %fneg = fneg <2 x half> %fabs
  store <2 x half> %fneg, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @divergent_fneg_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fneg_v2f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: V_XOR_B32_e64 %[[REG]]
; GCN: V_XOR_B32_e64 %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %tid
  %gep.out = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %tid
  %val = load <2 x float>, ptr addrspace(1) %gep.in, align 4
  %fneg = fneg <2 x float> %val
  store <2 x float> %fneg, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @uniform_fneg_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %idx) {
; GCN-LABEL: name:            uniform_fneg_v2f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: S_XOR_B32 killed %{{[0-9]+}}, %[[REG]]
; GCN: S_XOR_B32 killed %{{[0-9]+}}, %[[REG]]

  %gep.in = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %idx
  %gep.out = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %idx
  %val = load <2 x float>, ptr addrspace(1) %gep.in, align 4
  %fneg = fneg <2 x float> %val
  store <2 x float> %fneg, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @divergent_fabs_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fabs_v2f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 2147483647
; GCN: V_AND_B32_e64 %[[REG]]
; GCN: V_AND_B32_e64 %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %tid
  %gep.out = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %tid
  %val = load <2 x float>, ptr addrspace(1) %gep.in, align 4
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %val)
  store <2 x float> %fabs, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @uniform_fabs_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %idx) {
; GCN-LABEL: name:            uniform_fabs_v2f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 2147483647
; GCN: S_AND_B32 killed %{{[0-9]+}}, %[[REG]]
; GCN: S_AND_B32 killed %{{[0-9]+}}, %[[REG]]

  %gep.in = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %idx
  %gep.out = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %idx
  %val = load <2 x float>, ptr addrspace(1) %gep.in, align 4
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %val)
  store <2 x float> %fabs, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @divergent_fneg_fabs_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fneg_fabs_v2f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: V_OR_B32_e64 %[[REG]]
; GCN: V_OR_B32_e64 %[[REG]]

  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %tid
  %gep.out = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %tid
  %val = load <2 x float>, ptr addrspace(1) %gep.in, align 4
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %val)
  %fneg = fneg <2 x float> %fabs
  store <2 x float> %fneg, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @uniform_fneg_fabs_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %idx) {
; GCN-LABEL: name:            uniform_fneg_fabs_v2f32
; GCN-LABEL: bb.0 (%ir-block.0)
; GCN: %[[REG:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: S_OR_B32 killed %{{[0-9]+}}, %[[REG]]
; GCN: S_OR_B32 killed %{{[0-9]+}}, %[[REG]]

  %gep.in = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %idx
  %gep.out = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %idx
  %val = load <2 x float>, ptr addrspace(1) %gep.in, align 4
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %val)
  %fneg = fneg <2 x float> %fabs
  store <2 x float> %fneg, ptr addrspace(1) %gep.out
  ret void
}

define amdgpu_kernel void @divergent_fneg_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fneg_f64
; GCN-LABEL: bb.0 (%ir-block.0)
; SI: %[[VREG64:[0-9]+]]:vreg_64 = BUFFER_LOAD_DWORDX2_ADDR64
; FP16: %[[VREG64:[0-9]+]]:vreg_64 = GLOBAL_LOAD_DWORDX2_SADDR
; GCN: %[[HI32:[0-9]+]]:vgpr_32 = COPY %[[VREG64]].sub1
; GCN: %[[SREG_MASK:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: %[[XOR:[0-9]+]]:vgpr_32 = V_XOR_B32_e64 killed %[[SREG_MASK]], killed  %[[HI32]]
; GCN: %[[LO32:[0-9]+]]:vgpr_32 = COPY %[[VREG64]].sub0
; GCN: REG_SEQUENCE killed %[[LO32]], %subreg.sub0, killed %[[XOR]], %subreg.sub1


  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds double, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i64 %tid.ext
  %val = load volatile double, ptr addrspace(1) %in.gep
  %fneg = fneg double %val
  store double %fneg, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @uniform_fneg_f64(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %idx) {
; GCN-LABEL: name:            uniform_fneg_f64
; GCN-LABEL: bb.0 (%ir-block.0)
; SI: %[[VREG64:[0-9]+]]:vreg_64 = BUFFER_LOAD_DWORDX2_ADDR64
; FP16: %[[VREG64:[0-9]+]]:vreg_64 = GLOBAL_LOAD_DWORDX2_SADDR
; GCN: %[[LO32:[0-9]+]]:sreg_32 = COPY %[[VREG64]].sub0
; GCN: %[[HI32:[0-9]+]]:sreg_32 = COPY %[[VREG64]].sub1
; GCN: %[[SREG_MASK:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: %[[XOR:[0-9]+]]:sreg_32 = S_XOR_B32 killed %[[HI32]], killed %[[SREG_MASK]]
; GCN: %[[XOR_COPY:[0-9]+]]:sreg_32 = COPY killed %[[XOR]]
; GCN: REG_SEQUENCE killed %[[LO32]], %subreg.sub0, killed %[[XOR_COPY]], %subreg.sub1

  %in.gep = getelementptr inbounds double, ptr addrspace(1) %in, i64 %idx
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i64 %idx
  %val = load volatile double, ptr addrspace(1) %in.gep
  %fneg = fneg double %val
  store double %fneg, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @divergent_fabs_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fabs_f64
; GCN-LABEL: bb.0 (%ir-block.0)
; SI: %[[VREG64:[0-9]+]]:vreg_64 = BUFFER_LOAD_DWORDX2_ADDR64
; FP16: %[[VREG64:[0-9]+]]:vreg_64 = GLOBAL_LOAD_DWORDX2_SADDR
; GCN: %[[HI32:[0-9]+]]:vgpr_32 = COPY %[[VREG64]].sub1
; GCN: %[[SREG_MASK:[0-9]+]]:sreg_32 = S_MOV_B32 2147483647
; GCN: %[[AND:[0-9]+]]:vgpr_32 = V_AND_B32_e64 killed %[[SREG_MASK]], killed  %[[HI32]]
; GCN: %[[LO32:[0-9]+]]:vgpr_32 = COPY %[[VREG64]].sub0
; GCN: REG_SEQUENCE killed %[[LO32]], %subreg.sub0, killed %[[AND]], %subreg.sub1


  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds double, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i64 %tid.ext
  %val = load volatile double, ptr addrspace(1) %in.gep
  %fabs = call double @llvm.fabs.f64(double %val)
  store double %fabs, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @uniform_fabs_f64(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %idx) {
; GCN-LABEL: name:            uniform_fabs_f64
; GCN-LABEL: bb.0 (%ir-block.0)
; SI: %[[VREG64:[0-9]+]]:vreg_64 = BUFFER_LOAD_DWORDX2_ADDR64
; FP16: %[[VREG64:[0-9]+]]:vreg_64 = GLOBAL_LOAD_DWORDX2_SADDR
; GCN: %[[LO32:[0-9]+]]:sreg_32 = COPY %[[VREG64]].sub0
; GCN: %[[HI32:[0-9]+]]:sreg_32 = COPY %[[VREG64]].sub1
; GCN: %[[SREG_MASK:[0-9]+]]:sreg_32 = S_MOV_B32 2147483647
; GCN: %[[AND:[0-9]+]]:sreg_32 = S_AND_B32 killed %[[HI32]], killed %[[SREG_MASK]]
; GCN: %[[AND_COPY:[0-9]+]]:sreg_32 = COPY killed %[[AND]]
; GCN: REG_SEQUENCE killed %[[LO32]], %subreg.sub0, killed %[[AND_COPY]], %subreg.sub1


  %in.gep = getelementptr inbounds double, ptr addrspace(1) %in, i64 %idx
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i64 %idx
  %val = load volatile double, ptr addrspace(1) %in.gep
  %fabs = call double @llvm.fabs.f64(double %val)
  store double %fabs, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @divergent_fneg_fabs_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; GCN-LABEL: name:            divergent_fneg_fabs_f64
; GCN-LABEL: bb.0 (%ir-block.0)
; SI: %[[VREG64:[0-9]+]]:vreg_64 = BUFFER_LOAD_DWORDX2_ADDR64
; FP16: %[[VREG64:[0-9]+]]:vreg_64 = GLOBAL_LOAD_DWORDX2_SADDR
; GCN: %[[HI32:[0-9]+]]:vgpr_32 = COPY %[[VREG64]].sub1
; GCN: %[[SREG_MASK:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: %[[OR:[0-9]+]]:vgpr_32 = V_OR_B32_e64 killed %[[SREG_MASK]], killed  %[[HI32]]
; GCN: %[[LO32:[0-9]+]]:vgpr_32 = COPY %[[VREG64]].sub0
; GCN: REG_SEQUENCE killed %[[LO32]], %subreg.sub0, killed %[[OR]], %subreg.sub1


  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds double, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i64 %tid.ext
  %val = load volatile double, ptr addrspace(1) %in.gep
  %fabs = call double @llvm.fabs.f64(double %val)
  %fneg = fneg double %fabs
  store double %fneg, ptr addrspace(1) %out.gep
  ret void
}

define amdgpu_kernel void @uniform_fneg_fabs_f64(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %idx) {
; GCN-LABEL: name:            uniform_fneg_fabs_f64
; GCN-LABEL: bb.0 (%ir-block.0)
; SI: %[[VREG64:[0-9]+]]:vreg_64 = BUFFER_LOAD_DWORDX2_ADDR64
; FP16: %[[VREG64:[0-9]+]]:vreg_64 = GLOBAL_LOAD_DWORDX2_SADDR
; GCN: %[[LO32:[0-9]+]]:sreg_32 = COPY %[[VREG64]].sub0
; GCN: %[[HI32:[0-9]+]]:sreg_32 = COPY %[[VREG64]].sub1
; GCN: %[[SREG_MASK:[0-9]+]]:sreg_32 = S_MOV_B32 -2147483648
; GCN: %[[OR:[0-9]+]]:sreg_32 = S_OR_B32 killed %[[HI32]], killed %[[SREG_MASK]]
; GCN: %[[OR_COPY:[0-9]+]]:sreg_32 = COPY killed %[[OR]]
; GCN: REG_SEQUENCE killed %[[LO32]], %subreg.sub0, killed %[[OR_COPY]], %subreg.sub1


  %in.gep = getelementptr inbounds double, ptr addrspace(1) %in, i64 %idx
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i64 %idx
  %val = load volatile double, ptr addrspace(1) %in.gep
  %fabs = call double @llvm.fabs.f64(double %val)
  %fneg = fneg double %fabs
  store double %fneg, ptr addrspace(1) %out.gep
  ret void
}

declare float @llvm.fabs.f32(float)
declare half @llvm.fabs.f16(half)
declare double @llvm.fabs.f64(double)
declare <2 x half> @llvm.fabs.v2f16(<2 x half>)
declare <2 x float> @llvm.fabs.v2f32(<2 x float>)

declare i32 @llvm.amdgcn.workitem.id.x()
