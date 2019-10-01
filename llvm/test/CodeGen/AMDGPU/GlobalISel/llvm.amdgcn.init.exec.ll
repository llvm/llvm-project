; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %S/../llvm.amdgcn.init.exec.ll | FileCheck -check-prefix=GCN %S/../llvm.amdgcn.init.exec.ll
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %S/../llvm.amdgcn.init.exec.ll | FileCheck -check-prefix=GCN %S/../llvm.amdgcn.init.exec.ll
