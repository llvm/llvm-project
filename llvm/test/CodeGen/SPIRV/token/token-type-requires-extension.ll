; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - 2>&1 | FileCheck %s

; CHECK: A token is encountered but SPIR-V without extensions does not support token type

declare token @llvm.myfun()

define dso_local spir_func void @func() {
entry:
  %tok = call token @llvm.myfun()
  ret void
}
