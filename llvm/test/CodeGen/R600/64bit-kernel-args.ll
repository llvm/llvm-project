; RUN: llc < %s -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s --check-prefix=SI

; SI: {{^}}f64_kernel_arg:
; SI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x9
; SI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0xb
; SI: buffer_store_dwordx2
define void @f64_kernel_arg(double addrspace(1)* %out, double  %in) {
entry:
  store double %in, double addrspace(1)* %out
  ret void
}
