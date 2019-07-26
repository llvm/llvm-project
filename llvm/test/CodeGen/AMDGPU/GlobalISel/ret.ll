; Runs original SDAG test with -global-isel
; RUN: llc -global-isel -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %S/../ret.ll | FileCheck -check-prefix=GCN %S/../ret.ll
; RUN: llc -global-isel -march=amdgcn -mcpu=tonga -verify-machineinstrs < %S/../ret.ll | FileCheck -check-prefix=GCN %S/../ret.ll
