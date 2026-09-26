; Verify that the OpenCL printf format operand is always a pointer to i8, even
; when the format string is a global whose deduced pointee type is the array
; type. Consumers that type the printf declaration from its first call site
; reject a module whose format strings differ in length otherwise.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#ExtImport:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#CharPtr:]] = OpTypePointer UniformConstant %[[#Char]]

; Both format strings are cast to the same pointer-to-i8 type before the call,
; so both printf calls agree on the type of their first operand.
; CHECK: %[[#Cast1:]] = OpBitcast %[[#CharPtr]] %[[#]]
; CHECK: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#Cast1]]
; CHECK: %[[#Cast2:]] = OpBitcast %[[#CharPtr]] %[[#]]
; CHECK: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#Cast2]]

@.str = private unnamed_addr addrspace(2) constant [10 x i8] c"short %d\0A\00", align 1
@.str.1 = private unnamed_addr addrspace(2) constant [37 x i8] c"a much longer format string here %d\0A\00", align 1

declare spir_func i32 @printf(ptr addrspace(2), ...)

define spir_kernel void @test_printf_format_operand(i32 %n) {
entry:
  %call = call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str, i32 %n)
  %call1 = call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str.1, i32 %n)
  ret void
}
