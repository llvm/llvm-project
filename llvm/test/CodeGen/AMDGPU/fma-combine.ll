; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -fp-contract=fast < %s | FileCheck -check-prefix=SI-FASTFMAF -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs -fp-contract=fast < %s | FileCheck -check-prefix=SI-SLOWFMAF -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare double @llvm.fabs.f64(double) #0
declare double @llvm.fma.f64(double, double, double) #0
declare float @llvm.fma.f32(float, float, float) #0

; (fadd (fmul x, y), z) -> (fma x, y, z)
; FUNC-LABEL: {{^}}combine_to_fma_f64_0:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_f64_0(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fadd double %mul, %c
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fadd (fmul x, y), z) -> (fma x, y, z)
; FUNC-LABEL: {{^}}combine_to_fma_f64_0_2use:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_f64_0_2use(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %fma0 = fadd double %mul, %c
  %fma1 = fadd double %mul, %d
  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fadd x, (fmul y, z)) -> (fma y, z, x)
; FUNC-LABEL: {{^}}combine_to_fma_f64_1:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_f64_1(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fadd double %c, %mul
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_0_f64:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_fsub_0_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fsub double %mul, %c
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_f64_0_2use:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_fsub_f64_0_2use(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %fma0 = fsub double %mul, %c
  %fma1 = fsub double %mul, %d
  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
; FUNC-LABEL: {{^}}combine_to_fma_fsub_1_f64:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], [[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_fsub_1_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fsub double %c, %mul
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
; FUNC-LABEL: {{^}}combine_to_fma_fsub_1_f64_2use:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], [[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], [[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_fsub_1_f64_2use(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %fma0 = fsub double %c, %mul
  %fma1 = fsub double %d, %mul
  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_2_f64:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_fsub_2_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %mul.neg = fsub double -0.0, %mul
  %fma = fsub double %mul.neg, %c

  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_2_f64_2uses_neg:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_fsub_2_f64_2uses_neg(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %mul.neg = fsub double -0.0, %mul
  %fma0 = fsub double %mul.neg, %c
  %fma1 = fsub double %mul.neg, %d

  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_2_f64_2uses_mul:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_fsub_2_f64_2uses_mul(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %mul.neg = fsub double -0.0, %mul
  %fma0 = fsub double %mul.neg, %c
  %fma1 = fsub double %mul, %d

  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; fold (fsub (fma x, y, (fmul u, v)), z) -> (fma x, y (fma u, v, (fneg z)))

; FUNC-LABEL: {{^}}aggressive_combine_to_fma_fsub_0_f64:
; SI-DAG: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[Y:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[Z:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[U:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: buffer_load_dwordx2 [[V:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:32{{$}}
; SI: v_fma_f64 [[FMA0:v\[[0-9]+:[0-9]+\]]], [[U]], [[V]], -[[Z]]
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[X]], [[Y]], [[FMA0]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @aggressive_combine_to_fma_fsub_0_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr double, double addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %x = load volatile double, double addrspace(1)* %gep.0
  %y = load volatile double, double addrspace(1)* %gep.1
  %z = load volatile double, double addrspace(1)* %gep.2
  %u = load volatile double, double addrspace(1)* %gep.3
  %v = load volatile double, double addrspace(1)* %gep.4

  %tmp0 = fmul double %u, %v
  %tmp1 = call double @llvm.fma.f64(double %x, double %y, double %tmp0) #0
  %tmp2 = fsub double %tmp1, %z

  store double %tmp2, double addrspace(1)* %gep.out
  ret void
}

; fold (fsub x, (fma y, z, (fmul u, v)))
;   -> (fma (fneg y), z, (fma (fneg u), v, x))

; FUNC-LABEL: {{^}}aggressive_combine_to_fma_fsub_1_f64:
; SI-DAG: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[Y:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[Z:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[U:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: buffer_load_dwordx2 [[V:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:32{{$}}
; SI: v_fma_f64 [[FMA0:v\[[0-9]+:[0-9]+\]]], -[[U]], [[V]], [[X]]
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[Y]], [[Z]], [[FMA0]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @aggressive_combine_to_fma_fsub_1_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr double, double addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %x = load volatile double, double addrspace(1)* %gep.0
  %y = load volatile double, double addrspace(1)* %gep.1
  %z = load volatile double, double addrspace(1)* %gep.2
  %u = load volatile double, double addrspace(1)* %gep.3
  %v = load volatile double, double addrspace(1)* %gep.4

  %tmp0 = fmul double %u, %v
  %tmp1 = call double @llvm.fma.f64(double %y, double %z, double %tmp0) #0
  %tmp2 = fsub double %x, %tmp1

  store double %tmp2, double addrspace(1)* %gep.out
  ret void
}

;
; Patterns (+ fneg variants): mul(add(1.0,x),y), mul(sub(1.0,x),y), mul(sub(x,1.0),y)
;

; FUNC-LABEL: {{^}}test_f32_mul_add_x_one_y:
; SI: v_mac_f32_e32 [[VY:v[0-9]]], [[VY:v[0-9]]], [[VX:v[0-9]]]
define void @test_f32_mul_add_x_one_y(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load volatile float, float addrspace(1)* %in1
  %y = load volatile float, float addrspace(1)* %in2
  %a = fadd float %x, 1.0
  %m = fmul float %a, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_add_x_one:
; SI: v_mac_f32_e32 [[VY:v[0-9]]], [[VY:v[0-9]]], [[VX:v[0-9]]]
define void @test_f32_mul_y_add_x_one(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load volatile float, float addrspace(1)* %in1
  %y = load volatile float, float addrspace(1)* %in2
  %a = fadd float %x, 1.0
  %m = fmul float %y, %a
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_add_x_negone_y:
; SI: v_mad_f32 [[VX:v[0-9]]], [[VX]], [[VY:v[0-9]]], -[[VY]]
define void @test_f32_mul_add_x_negone_y(float addrspace(1)* %out,
                                           float addrspace(1)* %in1,
                                           float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %a = fadd float %x, -1.0
  %m = fmul float %a, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_add_x_negone:
; SI: v_mad_f32 [[VX:v[0-9]]], [[VX]], [[VY:v[0-9]]], -[[VY]]
define void @test_f32_mul_y_add_x_negone(float addrspace(1)* %out,
                                           float addrspace(1)* %in1,
                                           float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %a = fadd float %x, -1.0
  %m = fmul float %y, %a
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_sub_one_x_y:
; SI: v_mad_f32 [[VX:v[0-9]]], -[[VX]], [[VY:v[0-9]]], [[VY]]
define void @test_f32_mul_sub_one_x_y(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float 1.0, %x
  %m = fmul float %s, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_sub_one_x:
; SI: v_mad_f32 [[VX:v[0-9]]], -[[VX]], [[VY:v[0-9]]], [[VY]]
define void @test_f32_mul_y_sub_one_x(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float 1.0, %x
  %m = fmul float %y, %s
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_sub_negone_x_y:
; SI: v_mad_f32 [[VX:v[0-9]]], -[[VX]], [[VY:v[0-9]]], -[[VY]]
define void @test_f32_mul_sub_negone_x_y(float addrspace(1)* %out,
                                           float addrspace(1)* %in1,
                                           float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float -1.0, %x
  %m = fmul float %s, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_sub_negone_x:
; SI: v_mad_f32 [[VX:v[0-9]]], -[[VX]], [[VY:v[0-9]]], -[[VY]]
define void @test_f32_mul_y_sub_negone_x(float addrspace(1)* %out,
                                         float addrspace(1)* %in1,
                                         float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float -1.0, %x
  %m = fmul float %y, %s
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_sub_x_one_y:
; SI: v_mad_f32 [[VX:v[0-9]]], [[VX]], [[VY:v[0-9]]], -[[VY]]
define void @test_f32_mul_sub_x_one_y(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float %x, 1.0
  %m = fmul float %s, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_sub_x_one:
; SI: v_mad_f32 [[VX:v[0-9]]], [[VX]], [[VY:v[0-9]]], -[[VY]]
define void @test_f32_mul_y_sub_x_one(float addrspace(1)* %out,
                                      float addrspace(1)* %in1,
                                      float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float %x, 1.0
  %m = fmul float %y, %s
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_sub_x_negone_y:
; SI: v_mac_f32_e32 [[VY:v[0-9]]], [[VY]], [[VX:v[0-9]]]
define void @test_f32_mul_sub_x_negone_y(float addrspace(1)* %out,
                                         float addrspace(1)* %in1,
                                         float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float %x, -1.0
  %m = fmul float %s, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_sub_x_negone:
; SI: v_mac_f32_e32 [[VY:v[0-9]]], [[VY]], [[VX:v[0-9]]]
define void @test_f32_mul_y_sub_x_negone(float addrspace(1)* %out,
                                         float addrspace(1)* %in1,
                                         float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float %x, -1.0
  %m = fmul float %y, %s
  store float %m, float addrspace(1)* %out
  ret void
}

;
; Interpolation Patterns: add(mul(x,t),mul(sub(1.0,t),y))
;

; FUNC-LABEL: {{^}}test_f32_interp:
; SI: v_mad_f32 [[VR:v[0-9]]], -[[VT:v[0-9]]], [[VY:v[0-9]]], [[VY]]
; SI: v_mac_f32_e32 [[VR]], [[VT]], [[VX:v[0-9]]]
define void @test_f32_interp(float addrspace(1)* %out,
                             float addrspace(1)* %in1,
                             float addrspace(1)* %in2,
                             float addrspace(1)* %in3) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %t = load float, float addrspace(1)* %in3
  %t1 = fsub float 1.0, %t
  %tx = fmul float %x, %t
  %ty = fmul float %y, %t1
  %r = fadd float %tx, %ty
  store float %r, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f64_interp:
; SI: v_fma_f64 [[VR:v\[[0-9]+:[0-9]+\]]], -[[VT:v\[[0-9]+:[0-9]+\]]], [[VY:v\[[0-9]+:[0-9]+\]]], [[VY]]
; SI: v_fma_f64 [[VR:v\[[0-9]+:[0-9]+\]]], [[VX:v\[[0-9]+:[0-9]+\]]], [[VT]], [[VR]]
define void @test_f64_interp(double addrspace(1)* %out,
                             double addrspace(1)* %in1,
                             double addrspace(1)* %in2,
                             double addrspace(1)* %in3) {
  %x = load double, double addrspace(1)* %in1
  %y = load double, double addrspace(1)* %in2
  %t = load double, double addrspace(1)* %in3
  %t1 = fsub double 1.0, %t
  %tx = fmul double %x, %t
  %ty = fmul double %y, %t1
  %r = fadd double %tx, %ty
  store double %r, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
