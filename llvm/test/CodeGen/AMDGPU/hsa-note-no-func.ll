; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA --check-prefix=HSA-CI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo | FileCheck --check-prefix=HSA --check-prefix=HSA-VI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji | FileCheck --check-prefix=HSA --check-prefix=HSA-FIJI %s

; HSA: .hsa_code_object_version 1,0
; HSA-CI: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
; HSA-VI: .hsa_code_object_isa 8,0,1,"AMD","AMDGPU"
; HSA-FIJI: .hsa_code_object_isa 8,0,3,"AMD","AMDGPU"
