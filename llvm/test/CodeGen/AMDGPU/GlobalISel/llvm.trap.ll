; Runs original SDAG test with -global-isel
; RUN: llc -global-isel -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -verify-machineinstrs < %S/../trap.ll | FileCheck -check-prefix=GCN -check-prefix=HSA-TRAP -enable-var-scope %S/../trap.ll

; RUN: llc -global-isel -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -mattr=+trap-handler -verify-machineinstrs < %S/../trap.ll | FileCheck -check-prefix=GCN -check-prefix=HSA-TRAP -enable-var-scope %S/../trap.ll
; RUN: llc -global-isel -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -mattr=-trap-handler -verify-machineinstrs < %S/../trap.ll | FileCheck -check-prefix=GCN -check-prefix=NO-HSA-TRAP -enable-var-scope %S/../trap.ll
; RUN: llc -global-isel -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -mattr=-trap-handler -verify-machineinstrs < %S/../trap.ll 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING -enable-var-scope %S/../trap.ll

; enable trap handler feature
; RUN: llc -global-isel -mtriple=amdgcn-unknown-mesa3d -mattr=+trap-handler -verify-machineinstrs < %S/../trap.ll | FileCheck -check-prefix=GCN -check-prefix=NO-MESA-TRAP -check-prefix=TRAP-BIT -check-prefix=MESA-TRAP -enable-var-scope %S/../trap.ll
; RUN: llc -global-isel -mtriple=amdgcn-unknown-mesa3d -mattr=+trap-handler -verify-machineinstrs < %S/../trap.ll 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING -check-prefix=TRAP-BIT -enable-var-scope %S/../trap.ll

; disable trap handler feature
; RUN: llc -global-isel -mtriple=amdgcn-unknown-mesa3d -mattr=-trap-handler -verify-machineinstrs < %S/../trap.ll | FileCheck -check-prefix=GCN -check-prefix=NO-MESA-TRAP -check-prefix=NO-TRAP-BIT -check-prefix=NOMESA-TRAP -enable-var-scope %S/../trap.ll
; RUN: llc -global-isel -mtriple=amdgcn-unknown-mesa3d -mattr=-trap-handler -verify-machineinstrs < %S/../trap.ll 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING -check-prefix=NO-TRAP-BIT -enable-var-scope %S/../trap.ll

; RUN: llc -global-isel -march=amdgcn -verify-machineinstrs < %S/../trap.ll 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING -enable-var-scope %S/../trap.ll
