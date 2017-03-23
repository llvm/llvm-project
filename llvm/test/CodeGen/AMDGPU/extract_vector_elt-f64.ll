; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}extract_vector_elt_v3f64_2:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx2
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @extract_vector_elt_v3f64_2(double addrspace(1)* %out, <3 x double> addrspace(1)* %in) #0 {
  %ld = load volatile <3 x double>, <3 x double> addrspace(1)* %in
  %elt = extractelement <3 x double> %ld, i32 2
  store volatile double %elt, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v3f64:
define amdgpu_kernel void @dyn_extract_vector_elt_v3f64(double addrspace(1)* %out, <3 x double> %foo, i32 %elt) #0 {
  %dynelt = extractelement <3 x double> %foo, i32 %elt
  store volatile double %dynelt, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v4f64:
define amdgpu_kernel void @dyn_extract_vector_elt_v4f64(double addrspace(1)* %out, <4 x double> %foo, i32 %elt) #0 {
  %dynelt = extractelement <4 x double> %foo, i32 %elt
  store volatile double %dynelt, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
