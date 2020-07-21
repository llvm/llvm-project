; RUN: llc -global-isel -march=amdgcn -verify-machineinstrs < %S/../llvm.amdgcn.ds.ordered.swap.ll | FileCheck -check-prefixes=GCN,FUNC %S/../llvm.amdgcn.ds.ordered.swap.ll
; RUN: llc -global-isel -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %S/../llvm.amdgcn.ds.ordered.swap.ll | FileCheck -check-prefixes=GCN,FUNC %S/../llvm.amdgcn.ds.ordered.swap.ll
; RUN: llc -global-isel -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %S/../llvm.amdgcn.ds.ordered.swap.ll | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %S/../llvm.amdgcn.ds.ordered.swap.ll
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %S/../llvm.amdgcn.ds.ordered.swap.ll | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %S/../llvm.amdgcn.ds.ordered.swap.ll
