; Runs original SDAG test with -global-isel
; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=bonaire -verify-machineinstrs < %S/../read_register.ll | FileCheck -enable-var-scope %S/../read_register.ll
