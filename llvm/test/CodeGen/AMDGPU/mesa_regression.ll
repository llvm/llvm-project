; RUN: llc -O2 -mtriple amdgcn--amdhsa -mcpu=fiji -amdgpu-scalarize-global-loads=false -verify-machineinstrs  < %s | FileCheck %s

; CHECK-LABEL: %entry
; CHECK: flat_load_dwordx4

define amdgpu_kernel void @store_global(ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture readonly %in) {
entry:
  %tmp = load <16 x double>, ptr addrspace(1) %in
  store <16 x double> %tmp, ptr addrspace(1) %out
  ret void
}
