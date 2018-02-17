; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx600 | FileCheck --check-prefix=HSA --check-prefix=HSA-SI600 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx601 | FileCheck --check-prefix=HSA --check-prefix=HSA-SI601 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx700 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx701 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx702 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI702 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx703 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx704 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=bonaire | FileCheck --check-prefix=HSA --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=mullins | FileCheck --check-prefix=HSA --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=hawaii | FileCheck --check-prefix=HSA --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kabini | FileCheck --check-prefix=HSA --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=-flat-for-global | FileCheck --check-prefix=HSA --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=tonga -mattr=-flat-for-global | FileCheck --check-prefix=HSA --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=-flat-for-global | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=polaris10 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=polaris11 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx801 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx802 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx803 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx810 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI810 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx900 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX900 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx902 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX902 %s

; HSA: .hsa_code_object_version 2,1
; HSA-SI600: .hsa_code_object_isa 6,0,0,"AMD","AMDGPU"
; HSA-SI601: .hsa_code_object_isa 6,0,1,"AMD","AMDGPU"
; HSA-CI700: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
; HSA-CI701: .hsa_code_object_isa 7,0,1,"AMD","AMDGPU"
; HSA-CI702: .hsa_code_object_isa 7,0,2,"AMD","AMDGPU"
; HSA-CI703: .hsa_code_object_isa 7,0,3,"AMD","AMDGPU"
; HSA-CI704: .hsa_code_object_isa 7,0,4,"AMD","AMDGPU"
; HSA-VI801: .hsa_code_object_isa 8,0,1,"AMD","AMDGPU"
; HSA-VI802: .hsa_code_object_isa 8,0,2,"AMD","AMDGPU"
; HSA-VI803: .hsa_code_object_isa 8,0,3,"AMD","AMDGPU"
; HSA-VI810: .hsa_code_object_isa 8,1,0,"AMD","AMDGPU"
; HSA-GFX900: .hsa_code_object_isa 9,0,0,"AMD","AMDGPU"
; HSA-GFX902: .hsa_code_object_isa 9,0,2,"AMD","AMDGPU"
