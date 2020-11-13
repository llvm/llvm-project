; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx600 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-SI600 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx601 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-SI601 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx700 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx701 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx702 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI702 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx703 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx704 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=bonaire --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=mullins --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=hawaii --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kabini --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo --amdhsa-code-object-version=2 -mattr=-flat-for-global | FileCheck --check-prefix=HSA --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=tonga --amdhsa-code-object-version=2 -mattr=-flat-for-global | FileCheck --check-prefix=HSA --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji --amdhsa-code-object-version=2 -mattr=-flat-for-global | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=polaris10 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=polaris11 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx801 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx802 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx803 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx810 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-VI810 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx900 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX900 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx902 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX902 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx904 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX904 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx906 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX906 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx908 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX908 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx909 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX909 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1010 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX1010 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1011 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX1011 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1012 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX1012 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1030 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX1030 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1031 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX1031 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1032 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX1032 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1033 --amdhsa-code-object-version=2 | FileCheck --check-prefix=HSA --check-prefix=HSA-GFX1033 %s

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
; HSA-GFX908: .hsa_code_object_isa 9,0,8,"AMD","AMDGPU"
; HSA-GFX909: .hsa_code_object_isa 9,0,9,"AMD","AMDGPU"
; HSA-GFX1010: .hsa_code_object_isa 10,1,0,"AMD","AMDGPU"
; HSA-GFX1011: .hsa_code_object_isa 10,1,1,"AMD","AMDGPU"
; HSA-GFX1012: .hsa_code_object_isa 10,1,2,"AMD","AMDGPU"
; HSA-GFX1030: .hsa_code_object_isa 10,3,0,"AMD","AMDGPU"
; HSA-GFX1031: .hsa_code_object_isa 10,3,1,"AMD","AMDGPU"
; HSA-GFX1032: .hsa_code_object_isa 10,3,2,"AMD","AMDGPU"
; HSA-GFX1033: .hsa_code_object_isa 10,3,3,"AMD","AMDGPU"
