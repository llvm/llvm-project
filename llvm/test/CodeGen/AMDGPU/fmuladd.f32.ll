; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=tahiti -denormal-fp-math-f32=preserve-sign -mattr=+fast-fmaf -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-FLUSH,GCN-FLUSH-MAD,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=tahiti -denormal-fp-math-f32=ieee -mattr=+fast-fmaf -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-DENORM-STRICT,SI-DENORM,GCN-DENORM-FASTFMA,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=verde -denormal-fp-math-f32=preserve-sign -mattr=-fast-fmaf -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-FLUSH,GCN-FLUSH-MAD,SI-FLUSH,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=verde -denormal-fp-math-f32=ieee -mattr=-fast-fmaf -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-DENORM-STRICT,SI-DENORM,GCN-DENORM-SLOWFMA,SI %s

; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=tahiti -denormal-fp-math-f32=preserve-sign -mattr=+fast-fmaf -fp-contract=fast < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-FLUSH,GCN-FLUSH-MAD,SI-FLUSH,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=tahiti -denormal-fp-math-f32=ieee -mattr=+fast-fmaf -fp-contract=fast < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI-DENORM,GCN-DENORM-FASTFMA,GCN-DENORM-FASTFMA-CONTRACT,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=verde -denormal-fp-math-f32=preserve-sign -mattr=-fast-fmaf -fp-contract=fast < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-FLUSH,GCN-FLUSH-MAD,SI-FLUSH,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=verde -denormal-fp-math-f32=ieee -mattr=-fast-fmaf -fp-contract=fast < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI-DENORM,GCN-DENORM-SLOWFMA,GCN-DENORM-SLOWFMA-CONTRACT,SI %s


; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-FLUSH,GCN-FLUSH-MAD %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=gfx900 -denormal-fp-math-f32=ieee -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-DENORM-STRICT,GCN-DENORM-FASTFMA %s

; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=gfx906 -denormal-fp-math-f32=preserve-sign -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-FLUSH,GCN-FLUSH-FMAC %s

; FIXME: Should probably test this, but sometimes selecting fmac is painful to match.
; XUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=gfx906 -denormal-fp-math-f32=ieee -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-DENORM-STRICT,GCN-DENORM-FASTFMA %s

; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=gfx1030 -denormal-fp-math-f32=preserve-sign -mattr=+mad-mac-f32-insts -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-FLUSH,GCN-FLUSH-FMAC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mcpu=gfx1030 -denormal-fp-math-f32=ieee -mattr=+mad-mac-f32-insts -fp-contract=on < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-DENORM-STRICT %s

; Test all permutations of: fp32 denormals, fast fp contract, fp contract enabled for fmuladd, fmaf fast/slow.

target triple = "amdgcn--"


declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.fmuladd.f32(float, float, float) #1
declare half @llvm.fmuladd.f16(half, half, half) #1
declare float @llvm.fabs.f32(float) #1

; GCN-LABEL: {{^}}fmuladd_f32:
; GCN-FLUSH-MAD: v_mac_f32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}
; GCN-FLUSH-FMAC: v_fmac_f32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}

; GCN-DENORM-FASTFMA: v_fma_f32 {{v[0-9]+, v[0-9]+, v[0-9]+}}

; GCN-DENORM-SLOWFMA: v_mul_f32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}
; GCN-DENORM-SLOWFMA: v_add_f32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}
define amdgpu_kernel void @fmuladd_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                         float addrspace(1)* %in2, float addrspace(1)* %in3) #0 {
  %r0 = load float, float addrspace(1)* %in1
  %r1 = load float, float addrspace(1)* %in2
  %r2 = load float, float addrspace(1)* %in3
  %r3 = tail call float @llvm.fmuladd.f32(float %r0, float %r1, float %r2)
  store float %r3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fmul_fadd_f32:
; GCN-FLUSH: v_mac_f32

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32

; GCN-DENORM-SLOWFMA-CONTRACT: v_mul_f32_e32
; GCN-DENORM-SLOWFMA-CONTRACT: v_add_f32_e32

; GCN-DENORM-STRICT: v_mul_f32_e32
; GCN-DENORM-STRICT: v_add_f32_e32
define amdgpu_kernel void @fmul_fadd_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                           float addrspace(1)* %in2, float addrspace(1)* %in3) #0 {
  %r0 = load volatile float, float addrspace(1)* %in1
  %r1 = load volatile float, float addrspace(1)* %in2
  %r2 = load volatile float, float addrspace(1)* %in3
  %mul = fmul float %r0, %r1
  %add = fadd float %mul, %r2
  store float %add, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fmul_fadd_contract_f32:
; GCN-FLUSH-FMAC: v_fmac_f32_e32

; GCN-DENORM-SLOWFMA-CONTRACT: v_mul_f32_e32
; GCN-DENORM-SLOWFMA-CONTRACT: v_add_f32_e32

; GCN-DENORM-FASTFMA: v_fma_f32
define amdgpu_kernel void @fmul_fadd_contract_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                           float addrspace(1)* %in2, float addrspace(1)* %in3) #0 {
  %r0 = load volatile float, float addrspace(1)* %in1
  %r1 = load volatile float, float addrspace(1)* %in2
  %r2 = load volatile float, float addrspace(1)* %in3
  %mul = fmul float %r0, %r1
  %add = fadd contract float %mul, %r2
  store float %add, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_2.0_a_b_f32
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],

; GCN-FLUSH-MAD: v_mac_f32_e32 [[R2]], 2.0, [[R1]]
; GCN-FLUSH-FMAC: v_fmac_f32_e32 [[R2]], 2.0, [[R1]]
; SI-FLUSH: buffer_store_dword [[R2]]
; VI-FLUSH: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; GCN-DENORM-FASTFMA: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]

; GCN-DENORM-SLOWFMA: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA: v_add_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fmuladd_2.0_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %r3 = tail call float @llvm.fmuladd.f32(float 2.0, float %r1, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_a_2.0_b_f32
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],

; GCN-FLUSH-MAD: v_mac_f32_e32 [[R2]], 2.0, [[R1]]
; GCN-FLUSH-FMAC: v_fmac_f32_e32 [[R2]], 2.0, [[R1]]

; SI-FLUSH: buffer_store_dword [[R2]]
; VI-FLUSH: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; GCN-DENORM-FASTFMA: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]

; GCN-DENORM-SLOWFMA: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA: v_add_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fmuladd_a_2.0_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %r3 = tail call float @llvm.fmuladd.f32(float %r1, float 2.0, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fadd_a_a_b_f32:
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],

; GCN-FLUSH: v_mac_f32_e32 [[R2]], 2.0, [[R1]]

; SI-FLUSH: buffer_store_dword [[R2]]
; VI-FLUSH: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]

; GCN-DENORM-SLOWFMA-CONTRACT: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA-CONTRACT: v_add_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]

; GCN-DENORM-STRICT: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-STRICT: v_add_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fadd_a_a_b_f32(float addrspace(1)* %out,
                            float addrspace(1)* %in1,
                            float addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r0 = load volatile float, float addrspace(1)* %gep.0
  %r1 = load volatile float, float addrspace(1)* %gep.1

  %add.0 = fadd float %r0, %r0
  %add.1 = fadd float %add.0, %r1
  store float %add.1, float addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fadd_b_a_a_f32:
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],

; GCN-FLUSH: v_mac_f32_e32 [[R2]], 2.0, [[R1]]

; SI-FLUSH: buffer_store_dword [[R2]]
; VI-FLUSH: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]

; GCN-DENORM-SLOWFMA-CONTRACT: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA-CONTRACT: v_add_f32_e32 [[RESULT:v[0-9]+]], [[R2]], [[TMP]]

; GCN-DENORM-STRICT: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-STRICT: v_add_f32_e32 [[RESULT:v[0-9]+]], [[R2]], [[TMP]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fadd_b_a_a_f32(float addrspace(1)* %out,
                            float addrspace(1)* %in1,
                            float addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r0 = load volatile float, float addrspace(1)* %gep.0
  %r1 = load volatile float, float addrspace(1)* %gep.1

  %add.0 = fadd float %r0, %r0
  %add.1 = fadd float %r1, %add.0
  store float %add.1, float addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_neg_2.0_a_b_f32
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],
; GCN-FLUSH-MAD: v_mac_f32_e32 [[R2]], -2.0, [[R1]]
; GCN-FLUSH-FMAC: v_fmac_f32_e32 [[R2]], -2.0, [[R1]]

; GCN-DENORM-FASTFMA: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], -2.0, [[R2]]

; GCN-DENORM-SLOWFMA: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[R2]], [[TMP]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fmuladd_neg_2.0_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %r3 = tail call float @llvm.fmuladd.f32(float -2.0, float %r1, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; XXX
; GCN-LABEL: {{^}}fmuladd_neg_2.0_neg_a_b_f32
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],

; GCN-FLUSH-MAD: v_mac_f32_e32 [[R2]], 2.0, [[R1]]
; GCN-FLUSH-FMAC: v_fmac_f32_e32 [[R2]], 2.0, [[R1]]

; SI-FLUSH: buffer_store_dword [[R2]]
; VI-FLUSH: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; GCN-DENORM-FASTFMA: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]

; GCN-DENORM-SLOWFMA: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA: v_add_f32_e32 [[RESULT:v[0-9]+]], [[R2]], [[TMP]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fmuladd_neg_2.0_neg_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %r1.fneg = fneg float %r1

  %r3 = tail call float @llvm.fmuladd.f32(float -2.0, float %r1.fneg, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_2.0_neg_a_b_f32:
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],

; GCN-FLUSH-MAD: v_mac_f32_e32 [[R2]], -2.0, [[R1]]
; GCN-FLUSH-FMAC: v_fmac_f32_e32 [[R2]], -2.0, [[R1]]

; SI-FLUSH: buffer_store_dword [[R2]]
; VI-FLUSH: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; GCN-DENORM-FASTFMA: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], -2.0, [[R2]]

; GCN-DENORM-SLOWFMA: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[R2]], [[TMP]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fmuladd_2.0_neg_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %r1.fneg = fneg float %r1

  %r3 = tail call float @llvm.fmuladd.f32(float 2.0, float %r1.fneg, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_2.0_a_neg_b_f32:
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],
; GCN-FLUSH-MAD: v_mad_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, -[[R2]]
; GCN-FLUSH-FMAC: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, -[[R2]]

; SI-FLUSH: buffer_store_dword [[RESULT]]
; VI-FLUSH: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GCN-DENORM-FASTFMA: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, -[[R2]]

; GCN-DENORM-SLOWFMA: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fmuladd_2.0_a_neg_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %r2.fneg = fneg float %r2

  %r3 = tail call float @llvm.fmuladd.f32(float 2.0, float %r1, float %r2.fneg)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}mad_sub_f32:
; GCN: {{buffer|flat|global}}_load_dword [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGC:v[0-9]+]]
; GCN-FLUSH: v_mad_f32 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -[[REGC]]

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -[[REGC]]

; GCN-DENORM-SLOWFMA-CONTRACT: v_mul_f32_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-SLOWFMA-CONTRACT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[REGC]]

; GCN-DENORM-STRICT: v_mul_f32_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[REGC]]

; SI: buffer_store_dword [[RESULT]]
; VI: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @mad_sub_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float, float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float, float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float, float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %gep0, align 4
  %b = load volatile float, float addrspace(1)* %gep1, align 4
  %c = load volatile float, float addrspace(1)* %gep2, align 4
  %mul = fmul float %a, %b
  %sub = fsub float %mul, %c
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; GCN-LABEL: {{^}}mad_sub_inv_f32:
; GCN: {{buffer|flat|global}}_load_dword [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGC:v[0-9]+]]

; GCN-FLUSH: v_mad_f32 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], [[REGC]]

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], [[REGC]]

; GCN-DENORM-SLOWFMA-CONTRACT: v_mul_f32_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-SLOWFMA-CONTRACT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[REGC]], [[TMP]]

; GCN-DENORM-STRICT: v_mul_f32_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[REGC]], [[TMP]]

; SI: buffer_store_dword [[RESULT]]
; VI: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @mad_sub_inv_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float, float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float, float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float, float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %gep0, align 4
  %b = load volatile float, float addrspace(1)* %gep1, align 4
  %c = load volatile float, float addrspace(1)* %gep2, align 4
  %mul = fmul float %a, %b
  %sub = fsub float %c, %mul
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; GCN-LABEL: {{^}}mad_sub_fabs_f32:
; GCN: {{buffer|flat|global}}_load_dword [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGC:v[0-9]+]]
; GCN-FLUSH: v_mad_f32 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -|[[REGC]]|

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -|[[REGC]]|

; GCN-DENORM-SLOWFMA-CONTRACT: v_mul_f32_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-SLOWFMA-CONTRACT: v_sub_f32_e64 [[RESULT:v[0-9]+]],  [[TMP]], |[[REGC]]|

; GCN-DENORM-STRICT: v_mul_f32_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_sub_f32_e64 [[RESULT:v[0-9]+]],  [[TMP]], |[[REGC]]|

; SI: buffer_store_dword [[RESULT]]
; VI: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @mad_sub_fabs_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float, float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float, float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float, float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %gep0, align 4
  %b = load volatile float, float addrspace(1)* %gep1, align 4
  %c = load volatile float, float addrspace(1)* %gep2, align 4
  %c.abs = call float @llvm.fabs.f32(float %c) #0
  %mul = fmul float %a, %b
  %sub = fsub float %mul, %c.abs
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; GCN-LABEL: {{^}}mad_sub_fabs_inv_f32:
; GCN: {{buffer|flat|global}}_load_dword [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGC:v[0-9]+]]
; GCN-FLUSH-MAD: v_mad_f32 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], |[[REGC]]|
; GCN-FLUSH-FMA: v_fma_f32 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], |[[REGC]]|

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], |[[REGC]]|

; GCN-DENORM-SLOWFMA-CONTRACT: v_mul_f32_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-SLOWFMA-CONTRACT: v_sub_f32_e64 [[RESULT:v[0-9]+]], |[[REGC]]|, [[TMP]]

; GCN-DENORM-STRICT: v_mul_f32_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_sub_f32_e64 [[RESULT:v[0-9]+]], |[[REGC]]|, [[TMP]]

; SI: buffer_store_dword [[RESULT]]
; VI: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @mad_sub_fabs_inv_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float, float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float, float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float, float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %gep0, align 4
  %b = load volatile float, float addrspace(1)* %gep1, align 4
  %c = load volatile float, float addrspace(1)* %gep2, align 4
  %c.abs = call float @llvm.fabs.f32(float %c) #0
  %mul = fmul float %a, %b
  %sub = fsub float %c.abs, %mul
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; GCN-LABEL: {{^}}neg_neg_mad_f32:
; GCN: {{buffer|flat|global}}_load_dword [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGC:v[0-9]+]]

; GCN-FLUSH: v_mac_f32_e32 [[REGC]], [[REGA]], [[REGB]]
; SI-FLUSH: buffer_store_dword [[REGC]]
; VI-FLUSH: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REGC]]

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], [[REGC]]

; GCN-DENORM-SLOWFMA-CONTRACT: v_mul_f32_e32 [[TMP:v[0-9]+]],  [[REGA]], [[REGB]]
; GCN-DENORM-SLOWFMA-CONTRACT: v_add_f32_e32 [[RESULT:v[0-9]+]], [[REGC]], [[TMP]]

; GCN-DENORM-STRICT: v_mul_f32_e32 [[TMP:v[0-9]+]], [[REGA]], [[REGB]]
; GCN-DENORM-STRICT: v_add_f32_e32 [[RESULT:v[0-9]+]], [[REGC]], [[TMP]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @neg_neg_mad_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float, float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float, float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float, float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %gep0, align 4
  %b = load volatile float, float addrspace(1)* %gep1, align 4
  %c = load volatile float, float addrspace(1)* %gep2, align 4
  %nega = fneg float %a
  %negb = fneg float %b
  %mul = fmul float %nega, %negb
  %sub = fadd float %mul, %c
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; GCN-LABEL: {{^}}mad_fabs_sub_f32:
; GCN: {{buffer|flat|global}}_load_dword [[REGA:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGB:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[REGC:v[0-9]+]]
; GCN-FLUSH: v_mad_f32 [[RESULT:v[0-9]+]], [[REGA]], |[[REGB]]|, -[[REGC]]

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], [[REGA]], |[[REGB]]|, -[[REGC]]

; GCN-DENORM-SLOWFMA-CONTRACT: v_mul_f32_e64 [[TMP:v[0-9]+]], [[REGA]], |[[REGB]]|
; GCN-DENORM-SLOWFMA-CONTRACT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[REGC]]

; GCN-DENORM-STRICT: v_mul_f32_e64 [[TMP:v[0-9]+]], [[REGA]], |[[REGB]]|
; GCN-DENORM-STRICT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[REGC]]

; SI: buffer_store_dword [[RESULT]]
; VI: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @mad_fabs_sub_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float, float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float, float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float, float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %gep0, align 4
  %b = load volatile float, float addrspace(1)* %gep1, align 4
  %c = load volatile float, float addrspace(1)* %gep2, align 4
  %b.abs = call float @llvm.fabs.f32(float %b) #0
  %mul = fmul float %a, %b.abs
  %sub = fsub float %mul, %c
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; GCN-LABEL: {{^}}fsub_c_fadd_a_a_f32:
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],
; GCN-FLUSH: v_mac_f32_e32 [[R2]], -2.0, [[R1]]
; SI-FLUSH: buffer_store_dword [[R2]]
; VI-FLUSH: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[R2]]

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], -2.0, [[R2]]

; GCN-DENORM-SLOWFMA-CONTRACT: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA-CONTRACT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[R2]], [[TMP]]

; GCN-DENORM-STRICT: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-STRICT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[R2]], [[TMP]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; VI-DENORM: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fsub_c_fadd_a_a_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %add = fadd float %r1, %r1
  %r3 = fsub float %r2, %add

  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fsub_fadd_a_a_c_f32:
; GCN: {{buffer|flat|global}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat|global}}_load_dword [[R2:v[0-9]+]],
; GCN-FLUSH: v_mad_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, -[[R2]]

; GCN-DENORM-FASTFMA-CONTRACT: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, -[[R2]]

; GCN-DENORM-SLOWFMA-CONTRACT: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-SLOWFMA-CONTRACT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]

; GCN-DENORM-STRICT: v_add_f32_e32 [[TMP:v[0-9]+]], [[R1]], [[R1]]
; GCN-DENORM-STRICT: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[R2]]

; SI: buffer_store_dword [[RESULT]]
; VI: {{global|flat}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fsub_fadd_a_a_c_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %add = fadd float %r1, %r1
  %r3 = fsub float %add, %r2

  store float %r3, float addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
