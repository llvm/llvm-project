; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA --check-prefix=HSA-CI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo | FileCheck --check-prefix=HSA --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=tonga | FileCheck --check-prefix=HSA --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=polaris10 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=polaris11 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s

; HSA: .hsa_code_object_version 2,1
; HSA-CI: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
; HSA-VI801: .hsa_code_object_isa 8,0,1,"AMD","AMDGPU"
; HSA-VI802: .hsa_code_object_isa 8,0,2,"AMD","AMDGPU"
; HSA-VI803: .hsa_code_object_isa 8,0,3,"AMD","AMDGPU"
