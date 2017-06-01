; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-HSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}constant_load_f64:
; GCN: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}]
; GCN-NOHSA: buffer_store_dwordx2
; GCN-HSA: flat_store_dwordx2
define void @constant_load_f64(double addrspace(1)* %out, double addrspace(2)* %in) #0 {
  %ld = load double, double addrspace(2)* %in
  store double %ld, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
