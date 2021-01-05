; RUN: not --crash llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -o - < %S/../llvm.amdgcn.ds.gws.sema.release.all.ll 2>&1 | FileCheck -enable-var-scope -check-prefix=GFX6ERR-GISEL %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=hawaii -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.sema.release.all.ll | FileCheck -enable-var-scope -check-prefixes=GCN,LOOP %S/../llvm.amdgcn.ds.gws.sema.release.all.ll
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=fiji -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.sema.release.all.ll | FileCheck -enable-var-scope -check-prefixes=GCN,LOOP %S/../llvm.amdgcn.ds.gws.sema.release.all.ll
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.sema.release.all.ll | FileCheck -enable-var-scope -check-prefixes=GCN,NOLOOP %S/../llvm.amdgcn.ds.gws.sema.release.all.ll
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 -o - -verify-machineinstrs < %S/../llvm.amdgcn.ds.gws.sema.release.all.ll | FileCheck -enable-var-scope -check-prefixes=GCN,NOLOOP %S/../llvm.amdgcn.ds.gws.sema.release.all.ll

; GFX6ERR-GISEL: LLVM ERROR: cannot select: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.ds.gws.sema.release.all), %{{[0-9]+}}:sgpr(s32) :: (store 4 into custom "GWSResource") (in function: gws_sema_release_all_offset0)

