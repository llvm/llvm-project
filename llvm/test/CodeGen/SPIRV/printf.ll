; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#ExtImport:]] = OpExtInstImport "OpenCL.std"
; CHECK: %[[#Char:]] = OpTypeInt 8 0
; CHECK: %[[#CharPtr:]] = OpTypePointer UniformConstant %[[#Char]]
; CHECK: %[[#GV:]] = OpVariable %[[#]] UniformConstant %[[#]]
; CHECK: OpFunction
; CHECK: %[[#Arg1:]] = OpFunctionParameter
; CHECK: %[[#Arg2:]] = OpFunctionParameter
; CHECK: %[[#CastedGV:]] = OpBitcast %[[#CharPtr]] %[[#GV]]
; CHECK-NEXT: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#CastedGV]] %[[#ArgConst:]]
; CHECK-NEXT: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#CastedGV]] %[[#ArgConst]]
; CHECK-NEXT: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#Arg1]] %[[#ArgConst:]]
; CHECK-NEXT: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#Arg1]] %[[#ArgConst]]
; CHECK-NEXT: %[[#CastedArg2:]] = OpBitcast %[[#CharPtr]] %[[#Arg2]]
; CHECK-NEXT: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#CastedArg2]] %[[#ArgConst]]
; CHECK-NEXT: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#CastedArg2]] %[[#ArgConst]]
; CHECK: OpFunctionEnd

%struct = type { [6 x i8] }

@FmtStr = internal addrspace(2) constant [6 x i8] c"c=%c\0A\00", align 1

define spir_kernel void @foo(ptr addrspace(2) %_arg_fmt1, ptr addrspace(2) byval(%struct) %_arg_fmt2) {
entry:
  %r1 = tail call spir_func i32 (ptr addrspace(2), ...) @_Z6printfPU3AS2Kcz(ptr addrspace(2) @FmtStr, i8 signext 97)
  %r2 = tail call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) @FmtStr, i8 signext 97)
  %r3 = tail call spir_func i32 (ptr addrspace(2), ...) @_Z6printfPU3AS2Kcz(ptr addrspace(2) %_arg_fmt1, i8 signext 97)
  %r4 = tail call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) %_arg_fmt1, i8 signext 97)
  %r5 = tail call spir_func i32 (ptr addrspace(2), ...) @_Z6printfPU3AS2Kcz(ptr addrspace(2) %_arg_fmt2, i8 signext 97)
  %r6 = tail call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) %_arg_fmt2, i8 signext 97)
  ret void
}

declare dso_local spir_func i32 @_Z6printfPU3AS2Kcz(ptr addrspace(2), ...)
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)
