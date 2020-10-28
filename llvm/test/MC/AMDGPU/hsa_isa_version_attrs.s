// RUN: llvm-mc -arch=amdgcn -mcpu=gfx801 -mattr=-fast-fmaf -show-encoding %s | FileCheck --check-prefix=GFX8 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -mattr=-mad-mix-insts -show-encoding %s | FileCheck --check-prefix=GFX9 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32 -show-encoding %s | FileCheck --check-prefix=GFX10 %s

.hsa_code_object_isa
// GFX8:  .hsa_code_object_isa 8,0,1,"AMD","AMDGPU"
// GFX9:  .hsa_code_object_isa 9,0,0,"AMD","AMDGPU"
// GFX10: .hsa_code_object_isa 10,1,0,"AMD","AMDGPU"
