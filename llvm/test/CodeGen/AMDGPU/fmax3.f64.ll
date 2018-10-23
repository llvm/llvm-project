; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare double @llvm.maxnum.f64(double, double) nounwind readnone

; SI-LABEL: {{^}}test_fmax3_f64:
; SI: buffer_load_dwordx2 [[REGA:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+:[0-9]+}}], 0{{$}}
; SI: buffer_load_dwordx2 [[REGB:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+:[0-9]+}}], 0 offset:8
; SI: buffer_load_dwordx2 [[REGC:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+:[0-9]+}}], 0 offset:16
; SI: v_max_f64 [[QUIET_A:v\[[0-9]+:[0-9]+\]]], [[REGA]], [[REGA]]
; SI: v_max_f64 [[QUIET_B:v\[[0-9]+:[0-9]+\]]], [[REGB]], [[REGB]]
; SI: v_max_f64 [[MAX0:v\[[0-9]+:[0-9]+\]]], [[QUIET_A]], [[QUIET_B]]
; SI: v_max_f64 [[QUIET_C:v\[[0-9]+:[0-9]+\]]], [[REGC]], [[REGC]]
; SI: v_max_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[MAX0]], [[QUIET_C]]
; SI: buffer_store_dwordx2 [[RESULT]],
; SI: s_endpgm
define amdgpu_kernel void @test_fmax3_f64(double addrspace(1)* %out, double addrspace(1)* %aptr) nounwind {
  %bptr = getelementptr double, double addrspace(1)* %aptr, i32 1
  %cptr = getelementptr double, double addrspace(1)* %aptr, i32 2
  %a = load volatile double, double addrspace(1)* %aptr, align 8
  %b = load volatile double, double addrspace(1)* %bptr, align 8
  %c = load volatile double, double addrspace(1)* %cptr, align 8
  %f0 = call double @llvm.maxnum.f64(double %a, double %b) nounwind readnone
  %f1 = call double @llvm.maxnum.f64(double %f0, double %c) nounwind readnone
  store double %f1, double addrspace(1)* %out, align 8
  ret void
}
