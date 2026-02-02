; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#ExtImport:]] = OpExtInstImport "OpenCL.std"
; CHECK: %[[#Char:]] = OpTypeInt 8 0
; CHECK: %[[#ConstCharPtr:]] = OpTypePointer UniformConstant %[[#Char]]
; CHECK: %[[#VarargStruct:]] = OpTypeStruct %[[#Char]]
; CHECK: %[[#VarargStructPtr:]] = OpTypePointer Function %[[#VarargStruct]]
; CHECK: %[[#CharPtr:]] = OpTypePointer Function %[[#Char]]
; CHECK: %[[#IntConst:]] = OpConstant %[[#Char]] 97
; CHECK: %[[#GV:]] = OpVariable %[[#]] UniformConstant %[[#]]
; CHECK: OpFunction
; CHECK: %[[#Arg:]] = OpFunctionParameter
; CHECK: %[[#VarargBuffer1:]] = OpVariable %[[#VarargStructPtr]] Function
; CHECK: %[[#VarargBuffer2:]] = OpVariable %[[#VarargStructPtr]] Function
; CHECK: %[[#CastedBuffer1:]] = OpBitcast %[[#CharPtr]] %[[#VarargBuffer1]]
; CHECK: %[[#GEP1:]] = OpInBoundsPtrAccessChain %[[#CharPtr]] %[[#VarargBuffer1]]
; CHECK: OpStore %[[#GEP1]] %[[#IntConst]] Aligned 1
; CHECK: %[[#CastedGV:]] = OpBitcast %[[#ConstCharPtr]] %[[#GV]]
; CHECK: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#CastedGV]] %[[#CastedBuffer1:]]
; CHECK: %[[#CastedBuffer2:]] = OpBitcast %[[#CharPtr]] %[[#VarargBuffer2]]
; CHECK: %[[#GEP2:]] = OpInBoundsPtrAccessChain %[[#CharPtr]] %[[#VarargBuffer2]]
; CHECK: OpStore %[[#GEP2]] %[[#IntConst]] Aligned 1
; CHECK: OpExtInst %[[#]] %[[#ExtImport]] printf %[[#Arg]] %[[#CastedBuffer2:]]
; CHECK: OpFunctionEnd

%struct = type { [6 x i8] }

@FmtStr = internal addrspace(2) constant [6 x i8] c"c=%c\0A\00", align 1

define spir_kernel void @foo(ptr addrspace(2) %_arg_fmt) {
entry:
  %r1 = tail call spir_func i32 (ptr addrspace(2), ...) @_Z6printfPU3AS2Kcz(ptr addrspace(2) @FmtStr, i8 signext 97)
  %r2 = tail call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) %_arg_fmt, i8 signext 97)
  ret void
}

declare dso_local spir_func i32 @_Z6printfPU3AS2Kcz(ptr addrspace(2), ...)
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)
