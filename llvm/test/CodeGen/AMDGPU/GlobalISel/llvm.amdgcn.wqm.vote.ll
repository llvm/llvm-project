; Runs original SDAG test with -global-isel
; RUN: llc -global-isel -march=amdgcn -mcpu=tonga -verify-machineinstrs < %S/../llvm.amdgcn.wqm.vote.ll | FileCheck -enable-var-scope -check-prefixes=CHECK,WAVE64  %S/../llvm.amdgcn.wqm.vote.ll
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %S/../llvm.amdgcn.wqm.vote.ll | FileCheck -enable-var-scope -check-prefixes=CHECK,WAVE32 %S/../llvm.amdgcn.wqm.vote.ll
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %S/../llvm.amdgcn.wqm.vote.ll | FileCheck -enable-var-scope -check-prefixes=CHECK,WAVE32 %S/../llvm.amdgcn.wqm.vote.ll
