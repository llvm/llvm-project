; RUN: llc -march=amdgcn -mcpu=gfx900 -global-isel -verify-machineinstrs < %S/../llvm.amdgcn.init.exec.ll | FileCheck -check-prefix=GCN %S/../llvm.amdgcn.init.exec.ll
; RUN: llc -march=amdgcn -mcpu=gfx1010 -global-isel -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %S/../llvm.amdgcn.init.exec.ll | FileCheck -check-prefix=GCN %S/../llvm.amdgcn.init.exec.ll
