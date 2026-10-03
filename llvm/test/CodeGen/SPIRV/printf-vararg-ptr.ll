; Check that a pointer constant passed in a variadic argument position does not
; crash when the callee declares fewer formal parameters than the call site.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#ExtImport:]] = OpExtInstImport "OpenCL.std"
; CHECK: OpExtInst %[[#]] %[[#ExtImport]] printf

@.fmt = private unnamed_addr addrspace(2) constant [3 x i8] c"%s\00"
@.arg = private unnamed_addr addrspace(2) constant [2 x i8] c"a\00"

define spir_kernel void @foo() {
entry:
  %r = call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) @.fmt, ptr addrspace(2) @.arg)
  ret void
}

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)
