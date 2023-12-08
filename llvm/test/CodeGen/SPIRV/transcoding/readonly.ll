; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpDecorate %[[#PARAM:]] FuncParamAttr NoWrite
; CHECK-SPIRV: %[[#PARAM]] = OpFunctionParameter %{{.*}}

define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function(i32 addrspace(1)* readonly %_arg_) local_unnamed_addr {
entry:
  ret void
}
