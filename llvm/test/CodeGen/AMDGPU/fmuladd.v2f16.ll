; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx901 -mattr=-fp64-fp16-denormals -fp-contract=on -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefixes=GCN,GCN-STRICT,GFX9-FLUSH,GFX9 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx901 -mattr=-fp64-fp16-denormals -fp-contract=on -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefixes=GCN,GCN-STRICT,GFX9-FLUSH,GFX9 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx901 -mattr=-fp64-fp16-denormals -fp-contract=fast -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefixes=GCN,GCN-CONTRACT,GFX9-FLUSH,GFX9 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx901 -mattr=-fp64-fp16-denormals -fp-contract=fast -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefixes=GCN,GCN-CONTRACT,GFX9-FLUSH,GFX9 %s

; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx901 -mattr=+fp64-fp16-denormals -fp-contract=on -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefixes=GCN,GCN-STRICT,GFX9-DENORM-STRICT,GFX9-DENORM,GFX9 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx901 -mattr=+fp64-fp16-denormals -fp-contract=on -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefixes=GCN,GCN-STRICT,GFX9-DENORM-STRICT,GFX9-DENORM,GFX9 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx901 -mattr=+fp64-fp16-denormals -fp-contract=fast -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefixes=GCN,GCN-CONTRACT,GFX9-DENORM-CONTRACT,GFX9-DENORM,GFX9 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx901 -mattr=+fp64-fp16-denormals -fp-contract=fast -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefixes=GCN,GCN-CONTRACT,GFX9-DENORM-CONTRACT,GFX9-DENORM,GFX9 %s

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare <2 x half> @llvm.fmuladd.v2f16(<2 x half>, <2 x half>, <2 x half>) #1
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #1

; GCN-LABEL: {{^}}fmuladd_v2f16:
; GFX9-FLUSH: v_pk_mul_f16 {{v[0-9]+, v[0-9]+, v[0-9]+}}
; GFX9-FLUSH: v_pk_add_f16 {{v[0-9]+, v[0-9]+, v[0-9]+}}

; GFX9-DENORM: v_pk_fma_f16 {{v[0-9]+, v[0-9]+, v[0-9]+}}
define amdgpu_kernel void @fmuladd_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in1,
                         <2 x half> addrspace(1)* %in2, <2 x half> addrspace(1)* %in3) #0 {
  %r0 = load <2 x half>, <2 x half> addrspace(1)* %in1
  %r1 = load <2 x half>, <2 x half> addrspace(1)* %in2
  %r2 = load <2 x half>, <2 x half> addrspace(1)* %in3
  %r3 = tail call <2 x half> @llvm.fmuladd.v2f16(<2 x half> %r0, <2 x half> %r1, <2 x half> %r2)
  store <2 x half> %r3, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_2.0_a_b_v2f16:
; GCN: {{buffer|flat}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat}}_load_dword [[R2:v[0-9]+]],
; GFX9-FLUSH: v_pk_add_f16 [[ADD0:v[0-9]+]], [[R1]], [[R1]]
; GFX9-FLUSH: v_pk_add_f16 [[RESULT:v[0-9]+]], [[ADD0]], [[R2]]

; GFX9-FLUSH: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX9-DENORM: v_pk_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]
; GFX9-DENORM: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fmuladd_2.0_a_b_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr <2 x half>, <2 x half> addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid

  %r1 = load volatile <2 x half>, <2 x half> addrspace(1)* %gep.0
  %r2 = load volatile <2 x half>, <2 x half> addrspace(1)* %gep.1

  %r3 = tail call <2 x half> @llvm.fmuladd.v2f16(<2 x half> <half 2.0, half 2.0>, <2 x half> %r1, <2 x half> %r2)
  store <2 x half> %r3, <2 x half> addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fmuladd_a_2.0_b_v2f16:
; GCN: {{buffer|flat}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat}}_load_dword [[R2:v[0-9]+]],
; GFX9-FLUSH: v_pk_add_f16 [[ADD0:v[0-9]+]], [[R1]], [[R1]]
; GFX9-FLUSH: v_pk_add_f16 [[RESULT:v[0-9]+]], [[ADD0]], [[R2]]

; GFX9-FLUSH: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

; GFX9-DENORM: v_pk_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]
; GFX9-DENORM: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fmuladd_a_2.0_b_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr <2 x half>, <2 x half> addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid

  %r1 = load volatile <2 x half>, <2 x half> addrspace(1)* %gep.0
  %r2 = load volatile <2 x half>, <2 x half> addrspace(1)* %gep.1

  %r3 = tail call <2 x half> @llvm.fmuladd.v2f16(<2 x half> %r1, <2 x half> <half 2.0, half 2.0>, <2 x half> %r2)
  store <2 x half> %r3, <2 x half> addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fadd_a_a_b_v2f16:
; GCN: {{buffer|flat}}_load_dword [[R1:v[0-9]+]],
; GCN: {{buffer|flat}}_load_dword [[R2:v[0-9]+]],
; GFX9-FLUSH: v_pk_add_f16 [[ADD0:v[0-9]+]], [[R1]], [[R1]]
; GFX9-FLUSH: v_pk_add_f16 [[RESULT:v[0-9]+]], [[ADD0]], [[R2]]

; GFX9-DENORM-STRICT: v_pk_add_f16 [[ADD0:v[0-9]+]], [[R1]], [[R1]]
; GFX9-DENORM-STRICT: v_pk_add_f16 [[RESULT:v[0-9]+]], [[ADD0]], [[R2]]

; GFX9-DENORM-CONTRACT: v_pk_fma_f16 [[RESULT:v[0-9]+]], [[R1]], 2.0, [[R2]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fadd_a_a_b_v2f16(<2 x half> addrspace(1)* %out,
                            <2 x half> addrspace(1)* %in1,
                            <2 x half> addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr <2 x half>, <2 x half> addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid

  %r0 = load volatile <2 x half>, <2 x half> addrspace(1)* %gep.0
  %r1 = load volatile <2 x half>, <2 x half> addrspace(1)* %gep.1

  %add.0 = fadd <2 x half> %r0, %r0
  %add.1 = fadd <2 x half> %add.0, %r1
  store <2 x half> %add.1, <2 x half> addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
