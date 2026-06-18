; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

@.str = private unnamed_addr addrspace(2) constant [12 x i8] c"Hello World\00", align 1

; CHECK-SPIRV: %[[#]] = OpExtInst %[[#]] %[[#]] printf %[[#]]

define dso_local spir_kernel void @BuiltinPrintf() {
entry:
  %call = tail call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) noundef @.str)
  ret void
}

declare noundef i32 @printf(ptr addrspace(2) nocapture noundef readonly, ...)
