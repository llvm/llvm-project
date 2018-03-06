// RUN: llvm-mc -arch=amdgcn -mcpu=gfx801 -mattr=-fast-fmaf -show-encoding %s | FileCheck --check-prefix=GFX8 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -mattr=-mad-mix-insts -show-encoding %s | FileCheck --check-prefix=GFX9 %s

.hsa_code_object_isa
// GFX8:  .hsa_code_object_isa 8,0,0,"AMD","AMDGPU"
// GFX9:  .hsa_code_object_isa 9,0,0,"AMD","AMDGPU"
