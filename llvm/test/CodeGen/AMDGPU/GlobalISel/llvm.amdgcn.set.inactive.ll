; RUN: llc -global-isel -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %S/../llvm.amdgcn.set.inactive.ll |  FileCheck -check-prefix=GCN %S/../llvm.amdgcn.set.inactive.ll
