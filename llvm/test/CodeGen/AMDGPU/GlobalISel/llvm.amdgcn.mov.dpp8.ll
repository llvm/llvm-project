; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-flat-for-global -verify-machineinstrs < %S/../llvm.amdgcn.mov.dpp8.ll | FileCheck -check-prefix=GFX10 %S/../llvm.amdgcn.mov.dpp8.ll
