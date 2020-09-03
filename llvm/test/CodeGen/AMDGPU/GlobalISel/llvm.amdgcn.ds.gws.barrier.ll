; XUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.barrier.ll | FileCheck -enable-var-scope -check-prefixes=GCN,LOOP %S/../llvm.amdgcn.ds.gws.barrier.ll
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=hawaii -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.barrier.ll | FileCheck -enable-var-scope -check-prefixes=GCN,LOOP %S/../llvm.amdgcn.ds.gws.barrier.ll
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=fiji -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.barrier.ll | FileCheck -enable-var-scope -check-prefixes=GCN,LOOP %S/../llvm.amdgcn.ds.gws.barrier.ll
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.barrier.ll | FileCheck -enable-var-scope -check-prefixes=GCN,NOLOOP,NOLOOP-GISEL %S/../llvm.amdgcn.ds.gws.barrier.ll
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 -asm-verbose=0 -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.barrier.ll | FileCheck -enable-var-scope -check-prefixes=GCN,NOLOOP,NOLOOP-GISEL,GFX10 %S/../llvm.amdgcn.ds.gws.barrier.ll

; Make sure the op is emitted bundled with a waitcnt with and without the retry loop, and the bundle is not removed by ExpandPostRAPseudos.
; XUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -stop-after=postrapseudos -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.barrier.ll | FileCheck -enable-var-scope -check-prefix=MIR %S/../llvm.amdgcn.ds.gws.barrier.ll
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -stop-after=postrapseudos -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.barrier.ll | FileCheck -enable-var-scope -check-prefix=MIR %S/../llvm.amdgcn.ds.gws.barrier.ll
