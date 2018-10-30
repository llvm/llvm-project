; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx600 | FileCheck --check-prefix=HSA --check-prefix=HSA-SI600 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx601 | FileCheck --check-prefix=HSA --check-prefix=HSA-SI601 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx700 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx701 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx702 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI702 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx703 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx704 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=bonaire | FileCheck --check-prefix=HSA --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=mullins | FileCheck --check-prefix=HSA --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=hawaii | FileCheck --check-prefix=HSA --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=kabini | FileCheck --check-prefix=HSA --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=kaveri | FileCheck --check-prefix=HSA --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=carrizo -mattr=-flat-for-global | FileCheck --check-prefix=HSA --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=tonga -mattr=-flat-for-global | FileCheck --check-prefix=HSA --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=fiji -mattr=-flat-for-global | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=polaris10 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=polaris11 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx801 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx802 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx803 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx810 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI810 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx900 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX900 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx902 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX902 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx904 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX904 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx906 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX906 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=gfx909 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX909 %s

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
; HSA-GFX904: .hsa_code_object_isa 9,0,4,"AMD","AMDGPU"
; HSA-GFX906: .hsa_code_object_isa 9,0,6,"AMD","AMDGPU"
; HSA-GFX909: .hsa_code_object_isa 9,0,9,"AMD","AMDGPU"
