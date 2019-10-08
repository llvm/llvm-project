; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-fp64-fp16-denormals -fp-contract=on -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-STRICT,VI-FLUSH,VI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-fp64-fp16-denormals -fp-contract=fast -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-CONTRACT,VI-FLUSH,VI %s

; RUN: llc -march=amdgcn -mcpu=fiji -mattr=+fp64-fp16-denormals -fp-contract=on -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-STRICT,GCN-DENORM,GCN-DENORM-STRICT,VI-DENORM-STRICT,VI-DENORM,VI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=+fp64-fp16-denormals -fp-contract=fast -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-CONTRACT,GCN-DENORM,GCN-DENORM-CONTRACT,VI-DENORM-CONTRACT,VI-DENORM,VI %s

; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-fp64-fp16-denormals -fp-contract=on -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-STRICT,GFX10-FLUSH,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-fp64-fp16-denormals -fp-contract=fast -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-CONTRACT,GFX10-FLUSH,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+fp64-fp16-denormals -fp-contract=on -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-STRICT,GCN-DENORM,GCN-DENORM-STRICT,GFX10-DENORM-STRICT,GFX10-DENORM,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+fp64-fp16-denormals -fp-contract=fast -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-CONTRACT,GCN-DENORM,GCN-DENORM-CONTRACT,GFX10-DENORM-CONTRACT,GFX10-DENORM,GFX10 %s

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare half @llvm.fmuladd.f16(half, half, half) #1
declare half @llvm.fabs.f16(half) #1

; GCN-LABEL: {{^}}fmuladd_f16:
; VI-FLUSH: v_mac_f16_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}

; VI-DENORM: v_fma_f16 {{v[0-9]+, v[0-9]+, v[0-9]+}}

; GFX10-FLUSH:  v_mul_f16_e32
; GFX10-FLUSH:  v_add_f16_e32
; GFX10-DENORM: v_fmac_f16_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}

define amdgpu_kernel void @fmuladd_f16(half addrspace(1)* %out, half addrspace(1)* %in1,
                         half addrspace(1)* %in2, half addrspace(1)* %in3) #0 {
  %r0 = load half, half addrspace(1)* %in1
  %r1 = load half, half addrspace(1)* %in2
  %r2 = load half, half addrspace(1)* %in3
  %r3 = tail call half @llvm.fmuladd.f16(half %r0, half %r1, half %r2)
  store half %r3, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_2.0_a_b_f16
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],
; VI-FLUSH: v_mac_f16_e32 [[R2]], 2.0, [[R1]]
; VI-FLUSH: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; VI-DENORM:    v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]
; GFX10-DENORM: v_fmac_f16_e32 [[R2:v[0-9]+]], 2.0, [[R1]]

; GFX10-FLUSH:  v_add_f16_e32 [[MUL2:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH:  v_add_f16_e32 [[RESULT:v[0-9]+]], [[MUL2]], [[R2]]

; VI-DENORM:    flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]
; GFX10-FLUSH:  global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

define amdgpu_kernel void @fmuladd_2.0_a_b_f16(half addrspace(1)* %out, half addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r1 = load volatile half, half addrspace(1)* %gep.0
  %r2 = load volatile half, half addrspace(1)* %gep.1

  %r3 = tail call half @llvm.fmuladd.f16(half 2.0, half %r1, half %r2)
  store half %r3, half addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_a_2.0_b_f16
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],
; VI-FLUSH: v_mac_f16_e32 [[R2]], 2.0, [[R1]]
; VI-FLUSH: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; VI-DENORM:    v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]
; GFX10-DENORM: v_fmac_f16_e32 [[R2]], 2.0, [[R1]]

; GFX10-FLUSH:  v_add_f16_e32 [[MUL2:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH:  v_add_f16_e32 [[RESULT:v[0-9]+]], [[MUL2]], [[R2]]

; VI-DENORM: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]
; GFX10-FLUSH:  global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

define amdgpu_kernel void @fmuladd_a_2.0_b_f16(half addrspace(1)* %out, half addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r1 = load volatile half, half addrspace(1)* %gep.0
  %r2 = load volatile half, half addrspace(1)* %gep.1

  %r3 = tail call half @llvm.fmuladd.f16(half %r1, half 2.0, half %r2)
  store half %r3, half addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fadd_a_a_b_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],
; VI-FLUSH: v_mac_f16_e32 [[R2]], 2.0, [[R1]]
; VI-FLUSH: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; VI-DENORM-CONTRACT:    v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]
; GFX10-DENORM-CONTRACT: v_fmac_f16_e32 [[R2]], 2.0, [[R1]]

; GCN-DENORM-STRICT: v_add_f16_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-STRICT: v_add_f16_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]

; VI-DENORM: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH:           v_add_f16_e32 [[MUL2:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH:           v_add_f16_e32 [[RESULT:v[0-9]+]], [[MUL2]], [[R2]]
; GFX10-FLUSH:           global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-STRICT:   global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-CONTRACT: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

define amdgpu_kernel void @fadd_a_a_b_f16(half addrspace(1)* %out,
                            half addrspace(1)* %in1,
                            half addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r0 = load volatile half, half addrspace(1)* %gep.0
  %r1 = load volatile half, half addrspace(1)* %gep.1

  %add.0 = fadd half %r0, %r0
  %add.1 = fadd half %add.0, %r1
  store half %add.1, half addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fadd_b_a_a_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],
; VI-FLUSH: v_mac_f16_e32 [[R2]], 2.0, [[R1]]
; VI-FLUSH: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; VI-DENORM-CONTRACT:    v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]
; GFX10-DENORM-CONTRACT: v_fmac_f16_e32 [[R2]], 2.0, [[R1]]

; GCN-DENORM-STRICT: v_add_f16_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-STRICT: v_add_f16_e32 [[RESULT:v[0-9]+]],  [[R2]], [[TMP]]

; VI-DENORM: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_add_f16_e32 [[MUL2:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH: v_add_f16_e32 [[RESULT:v[0-9]+]], [[R2]], [[MUL2]]
; GFX10-FLUSH: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-STRICT:   global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-CONTRACT: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

define amdgpu_kernel void @fadd_b_a_a_f16(half addrspace(1)* %out,
                            half addrspace(1)* %in1,
                            half addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r0 = load volatile half, half addrspace(1)* %gep.0
  %r1 = load volatile half, half addrspace(1)* %gep.1

  %add.0 = fadd half %r0, %r0
  %add.1 = fadd half %r1, %add.0
  store half %add.1, half addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_neg_2.0_a_b_f16
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],
; VI-FLUSH:     v_mac_f16_e32 [[R2]], -2.0, [[R1]]
; VI-DENORM:    v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], -2.0, [[R2]]
; GFX10-DENORM: v_fmac_f16_e32 [[R2]], -2.0, [[R1]]
; VI-FLUSH:  flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]
; VI-DENORM: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-FLUSH: v_add_f16_e32 [[MUL2:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[R2]], [[MUL2]]
; GFX10-FLUSH:  global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]
define amdgpu_kernel void @fmuladd_neg_2.0_a_b_f16(half addrspace(1)* %out, half addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r1 = load volatile half, half addrspace(1)* %gep.0
  %r2 = load volatile half, half addrspace(1)* %gep.1

  %r3 = tail call half @llvm.fmuladd.f16(half -2.0, half %r1, half %r2)
  store half %r3, half addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_neg_2.0_neg_a_b_f16
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],
; VI-FLUSH: v_mac_f16_e32 [[R2]], 2.0, [[R1]]
; VI-FLUSH: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; VI-DENORM:    v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]
; VI-DENORM:  flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_add_f16_e32 [[MUL2:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH: v_add_f16_e32 [[RESULT:v[0-9]+]], [[R2]], [[MUL2]]
; GFX10-FLUSH:  global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-DENORM: v_fmac_f16_e32 [[R2]], 2.0, [[R1]]
; GFX10-DENORM: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]
define amdgpu_kernel void @fmuladd_neg_2.0_neg_a_b_f16(half addrspace(1)* %out, half addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r1 = load volatile half, half addrspace(1)* %gep.0
  %r2 = load volatile half, half addrspace(1)* %gep.1

  %r1.fneg = fsub half -0.000000e+00, %r1

  %r3 = tail call half @llvm.fmuladd.f16(half -2.0, half %r1.fneg, half %r2)
  store half %r3, half addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_2.0_neg_a_b_f16
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],
; VI-FLUSH: v_mac_f16_e32 [[R2]], -2.0, [[R1]]
; VI-FLUSH: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; VI-DENORM: v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], -2.0, [[R2]]
; VI-DENORM:  flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_add_f16_e32 [[MUL2:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[R2]], [[MUL2]]
; GFX10-FLUSH: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-DENORM: v_fmac_f16_e32 [[R2]], -2.0, [[R1]]
; GFX10-DENORM: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]
define amdgpu_kernel void @fmuladd_2.0_neg_a_b_f16(half addrspace(1)* %out, half addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r1 = load volatile half, half addrspace(1)* %gep.0
  %r2 = load volatile half, half addrspace(1)* %gep.1

  %r1.fneg = fsub half -0.000000e+00, %r1

  %r3 = tail call half @llvm.fmuladd.f16(half 2.0, half %r1.fneg, half %r2)
  store half %r3, half addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_2.0_a_neg_b_f16
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],
; VI-FLUSH:   v_mad_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, -[[R2]]
; GCN-DENORM: v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, -[[R2]]
; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-FLUSH: v_add_f16_e32 [[MUL2:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[MUL2]], [[R2]]
; GFX10:       global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fmuladd_2.0_a_neg_b_f16(half addrspace(1)* %out, half addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r1 = load volatile half, half addrspace(1)* %gep.0
  %r2 = load volatile half, half addrspace(1)* %gep.1

  %r2.fneg = fsub half -0.000000e+00, %r2

  %r3 = tail call half @llvm.fmuladd.f16(half 2.0, half %r1, half %r2.fneg)
  store half %r3, half addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}mad_sub_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGC:v[0-9]+]]

; VI-FLUSH: v_mad_f16 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -[[REGC]]

; GCN-DENORM-CONTRACT: v_fma_f16 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -[[REGC]]

; GCN-DENORM-STRICT: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[TMP]], [[REGC]]

; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GFX10-FLUSH: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[TMP]], [[REGC]]
; GFX10:       global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @mad_sub_f16(half addrspace(1)* noalias nocapture %out, half addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr half, half addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr half, half addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr half, half addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %gep0, align 2
  %b = load volatile half, half addrspace(1)* %gep1, align 2
  %c = load volatile half, half addrspace(1)* %gep2, align 2
  %mul = fmul half %a, %b
  %sub = fsub half %mul, %c
  store half %sub, half addrspace(1)* %outgep, align 2
  ret void
}

; GCN-LABEL: {{^}}mad_sub_inv_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGC:v[0-9]+]]
; VI-FLUSH: v_mad_f16 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], [[REGC]]

; VI-DENORM-CONTRACT:    v_fma_f16 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], [[REGC]]
; GFX10-DENORM-CONTRACT: v_fmac_f16_e64 [[REGC]], -[[REGA]], [[REGB]]

; GCN-DENORM-STRICT: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[REGC]], [[TMP]]

; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GFX10-FLUSH: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[REGC]], [[TMP]]
; GFX10-FLUSH:  global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-STRICT: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-CONTRACT: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REGC]]
define amdgpu_kernel void @mad_sub_inv_f16(half addrspace(1)* noalias nocapture %out, half addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr half, half addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr half, half addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr half, half addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %gep0, align 2
  %b = load volatile half, half addrspace(1)* %gep1, align 2
  %c = load volatile half, half addrspace(1)* %gep2, align 2
  %mul = fmul half %a, %b
  %sub = fsub half %c, %mul
  store half %sub, half addrspace(1)* %outgep, align 2
  ret void
}

; GCN-LABEL: {{^}}mad_sub_fabs_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGC:v[0-9]+]]
; VI-FLUSH: v_mad_f16 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -|[[REGC]]|

; GCN-DENORM-CONTRACT: v_fma_f16 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -|[[REGC]]|

; GCN-DENORM-STRICT: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_sub_f16_e64 [[RESULT:v[0-9]+]], [[TMP]], |[[REGC]]|

; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GFX10-FLUSH: v_sub_f16_e64 [[RESULT:v[0-9]+]], [[TMP]], |[[REGC]]|
; GFX10:       global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @mad_sub_fabs_f16(half addrspace(1)* noalias nocapture %out, half addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr half, half addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr half, half addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr half, half addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %gep0, align 2
  %b = load volatile half, half addrspace(1)* %gep1, align 2
  %c = load volatile half, half addrspace(1)* %gep2, align 2
  %c.abs = call half @llvm.fabs.f16(half %c) #0
  %mul = fmul half %a, %b
  %sub = fsub half %mul, %c.abs
  store half %sub, half addrspace(1)* %outgep, align 2
  ret void
}

; GCN-LABEL: {{^}}mad_sub_fabs_inv_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGC:v[0-9]+]]

; VI-FLUSH: v_mad_f16 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], |[[REGC]]|

; GCN-DENORM-CONTRACT: v_fma_f16 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], |[[REGC]]|

; GCN-DENORM-STRICT: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_sub_f16_e64 [[RESULT:v[0-9]+]], |[[REGC]]|, [[TMP]]

; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GFX10-FLUSH: v_sub_f16_e64 [[RESULT:v[0-9]+]], |[[REGC]]|, [[TMP]]
; GFX10:       global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @mad_sub_fabs_inv_f16(half addrspace(1)* noalias nocapture %out, half addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr half, half addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr half, half addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr half, half addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %gep0, align 2
  %b = load volatile half, half addrspace(1)* %gep1, align 2
  %c = load volatile half, half addrspace(1)* %gep2, align 2
  %c.abs = call half @llvm.fabs.f16(half %c) #0
  %mul = fmul half %a, %b
  %sub = fsub half %c.abs, %mul
  store half %sub, half addrspace(1)* %outgep, align 2
  ret void
}

; GCN-LABEL: {{^}}neg_neg_mad_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGC:v[0-9]+]]

; VI-FLUSH: v_mac_f16_e32 [[REGC]], [[REGA]], [[REGB]]
; VI-FLUSH: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REGC]]

; VI-DENORM-CONTRACT:    v_fma_f16 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], [[REGC]]
; GFX10-DENORM-CONTRACT: v_fmac_f16_e32 [[REGC]], [[REGA]], [[REGB]]

; GCN-DENORM-STRICT: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_add_f16_e32 [[RESULT:v[0-9]+]], [[REGC]], [[TMP]]
; VI-DENORM: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_mul_f16_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GFX10-FLUSH: v_add_f16_e32 [[RESULT:v[0-9]+]], [[REGC]], [[TMP]]
; GFX10-FLUSH:  global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-STRICT: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-CONTRACT: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REGC]]
define amdgpu_kernel void @neg_neg_mad_f16(half addrspace(1)* noalias nocapture %out, half addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr half, half addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr half, half addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr half, half addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %gep0, align 2
  %b = load volatile half, half addrspace(1)* %gep1, align 2
  %c = load volatile half, half addrspace(1)* %gep2, align 2
  %nega = fsub half -0.000000e+00, %a
  %negb = fsub half -0.000000e+00, %b
  %mul = fmul half %nega, %negb
  %sub = fadd half %mul, %c
  store half %sub, half addrspace(1)* %outgep, align 2
  ret void
}

; GCN-LABEL: {{^}}mad_fabs_sub_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_ushort [[REGC:v[0-9]+]]

; VI-FLUSH: v_mad_f16 [[RESULT:v[0-9]+]], [[REGA]], |[[REGB]]|, -[[REGC]]

; GCN-DENORM-CONTRACT: v_fma_f16 [[RESULT:v[0-9]+]], [[REGA]], |[[REGB]]|, -[[REGC]]

; GCN-DENORM-STRICT: v_mul_f16_e64 [[TMP:v[0-9]+]], [[REGA]], |[[REGB]]|
; GCN-DENORM-STRICT: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[TMP]], [[REGC]]

; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_mul_f16_e64 [[TMP:v[0-9]+]], [[REGA]], |[[REGB]]|
; GFX10-FLUSH: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[TMP]], [[REGC]]
; GFX10:       global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @mad_fabs_sub_f16(half addrspace(1)* noalias nocapture %out, half addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr half, half addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr half, half addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr half, half addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %gep0, align 2
  %b = load volatile half, half addrspace(1)* %gep1, align 2
  %c = load volatile half, half addrspace(1)* %gep2, align 2
  %b.abs = call half @llvm.fabs.f16(half %b) #0
  %mul = fmul half %a, %b.abs
  %sub = fsub half %mul, %c
  store half %sub, half addrspace(1)* %outgep, align 2
  ret void
}

; GCN-LABEL: {{^}}fsub_c_fadd_a_a_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],
; VI-FLUSH: v_mac_f16_e32 [[R2]], -2.0, [[R1]]
; VI-FLUSH: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; VI-DENORM-CONTRACT:    v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], -2.0, [[R2]]
; GFX10-DENORM-CONTRACT: v_fmac_f16_e32 [[R2]], -2.0, [[R1]]

; GCN-DENORM-STRICT: v_add_f16_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-STRICT: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[R2]], [[TMP]]

; VI-DENORM: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_add_f16_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[R2]], [[TMP]]
; GFX10-FLUSH:  global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-STRICT:   global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GFX10-DENORM-CONTRACT: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[R2]]
define amdgpu_kernel void @fsub_c_fadd_a_a_f16(half addrspace(1)* %out, half addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r1 = load volatile half, half addrspace(1)* %gep.0
  %r2 = load volatile half, half addrspace(1)* %gep.1

  %add = fadd half %r1, %r1
  %r3 = fsub half %r2, %add

  store half %r3, half addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fsub_fadd_a_a_c_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_ushort [[R2:v[0-9]+]],

; VI-FLUSH: v_mad_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, -[[R2]]

; GCN-DENORM-CONTRACT: v_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, -[[R2]]

; GCN-DENORM-STRICT: v_add_f16_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-STRICT: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]

; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX10-FLUSH: v_add_f16_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GFX10-FLUSH: v_sub_f16_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]
; GFX10:       global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fsub_fadd_a_a_c_f16(half addrspace(1)* %out, half addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr half, half addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr half, half addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr half, half addrspace(1)* %out, i32 %tid

  %r1 = load volatile half, half addrspace(1)* %gep.0
  %r2 = load volatile half, half addrspace(1)* %gep.1

  %add = fadd half %r1, %r1
  %r3 = fsub half %add, %r2

  store half %r3, half addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
