; RUN: not llc -global-isel -march=amdgcn -mcpu=tonga < %S/../lds-zero-initializer.ll 2>&1 | FileCheck %s

; CHECK: <unknown>:0: error: lds: unsupported initializer for address space
