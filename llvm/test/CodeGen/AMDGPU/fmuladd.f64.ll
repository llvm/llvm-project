; RUN: llc -march=amdgcn -mcpu=tahiti -fp-contract=on -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GCN-STRICTSI %s
; RUN: llc -march=amdgcn -mcpu=verde  -fp-contract=on -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GCN-STRICT,SI %s
; RUN: llc -march=amdgcn -mcpu=tahiti -fp-contract=fast -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GCN-CONTRACT,SI %s
; RUN: llc -march=amdgcn -mcpu=verde  -fp-contract=fast -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GCN-CONTRACT,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global  -fp-contract=on -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GCN-STRICT,VI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global  -fp-contract=fast -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GCN-CONTRACT,VI %s

; GCN-LABEL: {{^}}fmuladd_f64:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fmuladd_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                         double addrspace(1)* %in2, double addrspace(1)* %in3) #0 {
  %r0 = load double, double addrspace(1)* %in1
  %r1 = load double, double addrspace(1)* %in2
  %r2 = load double, double addrspace(1)* %in3
  %r3 = tail call double @llvm.fmuladd.f64(double %r0, double %r1, double %r2)
  store double %r3, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fmul_fadd_f64:
; GCN-CONTRACT: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}

; GCN-STRICT: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GCN-STRICT: v_add_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fmul_fadd_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                           double addrspace(1)* %in2, double addrspace(1)* %in3) #0 {
  %r0 = load double, double addrspace(1)* %in1
  %r1 = load double, double addrspace(1)* %in2
  %r2 = load double, double addrspace(1)* %in3
  %tmp = fmul double %r0, %r1
  %r3 = fadd double %tmp, %r2
  store double %r3, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fadd_a_a_b_f64:
; GCN: {{buffer|flat}}_load_dwordx2 [[R1:v\[[0-9]+:[0-9]+\]]],
; GCN: {{buffer|flat}}_load_dwordx2 [[R2:v\[[0-9]+:[0-9]+\]]],

; GCN-STRICT: v_add_f64 [[TMP:v\[[0-9]+:[0-9]+\]]], [[R1]], [[R1]]
; GCN-STRICT: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[TMP]], [[R2]]

; GCN-CONTRACT: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[R1]], 2.0, [[R2]]

; SI: buffer_store_dwordx2 [[RESULT]]
; VI: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fadd_a_a_b_f64(double addrspace(1)* %out,
                            double addrspace(1)* %in1,
                            double addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %r0 = load volatile double, double addrspace(1)* %gep.0
  %r1 = load volatile double, double addrspace(1)* %gep.1

  %add.0 = fadd double %r0, %r0
  %add.1 = fadd double %add.0, %r1
  store double %add.1, double addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fadd_b_a_a_f64:
; GCN: {{buffer|flat}}_load_dwordx2 [[R1:v\[[0-9]+:[0-9]+\]]],
; GCN: {{buffer|flat}}_load_dwordx2 [[R2:v\[[0-9]+:[0-9]+\]]],

; GCN-STRICT: v_add_f64 [[TMP:v\[[0-9]+:[0-9]+\]]], [[R1]], [[R1]]
; GCN-STRICT: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[R2]], [[TMP]]

; GCN-CONTRACT: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[R1]], 2.0, [[R2]]

; SI: buffer_store_dwordx2 [[RESULT]]
; VI: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @fadd_b_a_a_f64(double addrspace(1)* %out,
                            double addrspace(1)* %in1,
                            double addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %r0 = load volatile double, double addrspace(1)* %gep.0
  %r1 = load volatile double, double addrspace(1)* %gep.1

  %add.0 = fadd double %r0, %r0
  %add.1 = fadd double %r1, %add.0
  store double %add.1, double addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}mad_sub_f64:
; GCN-STRICT: v_mul_f64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
; GCN-STRICT: v_add_f64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, -v{{\[[0-9]+:[0-9]+\]}}

; GCN-CONTRACT: v_fma_f64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, -v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @mad_sub_f64(double addrspace(1)* noalias nocapture %out, double addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr double, double addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr double, double addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr double, double addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %gep0, align 8
  %b = load volatile double, double addrspace(1)* %gep1, align 8
  %c = load volatile double, double addrspace(1)* %gep2, align 8
  %mul = fmul double %a, %b
  %sub = fsub double %mul, %c
  store double %sub, double addrspace(1)* %outgep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_a_a_b_f64_fast_add0:
; GCN-STRICT: v_add_f64
; GCN-STRICT: v_add_f64

; GCN-CONTRACT: v_fma_f64
define amdgpu_kernel void @fadd_a_a_b_f64_fast_add0(double addrspace(1)* %out,
                                      double addrspace(1)* %in1,
                                      double addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %r0 = load volatile double, double addrspace(1)* %gep.0
  %r1 = load volatile double, double addrspace(1)* %gep.1

  %add.0 = fadd fast double %r0, %r0
  %add.1 = fadd double %add.0, %r1
  store double %add.1, double addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fadd_a_a_b_f64_fast_add1:
; GCN-STRICT: v_add_f64
; GCN-STRICT: v_add_f64

; GCN-CONTRACT: v_fma_f64
define amdgpu_kernel void @fadd_a_a_b_f64_fast_add1(double addrspace(1)* %out,
                                      double addrspace(1)* %in1,
                                      double addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %r0 = load volatile double, double addrspace(1)* %gep.0
  %r1 = load volatile double, double addrspace(1)* %gep.1

  %add.0 = fadd double %r0, %r0
  %add.1 = fadd fast double %add.0, %r1
  store double %add.1, double addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fadd_a_a_b_f64_fast:
; GCN: v_fma_f64
define amdgpu_kernel void @fadd_a_a_b_f64_fast(double addrspace(1)* %out,
                                 double addrspace(1)* %in1,
                                double addrspace(1)* %in2) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %r0 = load volatile double, double addrspace(1)* %gep.0
  %r1 = load volatile double, double addrspace(1)* %gep.1

  %add.0 = fadd fast double %r0, %r0
  %add.1 = fadd fast double %add.0, %r1
  store double %add.1, double addrspace(1)* %gep.out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare double @llvm.fmuladd.f64(double, double, double) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
