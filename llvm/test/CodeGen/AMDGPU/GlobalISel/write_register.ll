; Runs original SDAG test with -global-isel
; RUN: llc -global-isel -march=amdgcn -mcpu=bonaire -enable-misched=0 -verify-machineinstrs < %S/../write_register.ll | FileCheck -enable-var-scope %S/../write_register.ll
