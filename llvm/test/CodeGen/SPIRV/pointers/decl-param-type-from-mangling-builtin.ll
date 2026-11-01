; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=OpFunctionCall
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Pipe builtins are recognized by lookupBuiltin, but mapBuiltinToOpcode does
; not handle their group. They must retain their existing lowering rather
; than becoming external function calls.
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Ptr:]] = OpTypePointer Generic %[[#Int]]
; CHECK: OpFunction
; CHECK: %[[#Data:]] = OpFunctionParameter %[[#Ptr]]
; CHECK: OpReadPipe %[[#Int]] %[[#]] %[[#Data]]

declare spir_func i32 @_Z13__read_pipe_211ocl_pipe_roPiii(target("spirv.Pipe", 0), ptr addrspace(4), i32, i32)

define spir_kernel void @test(target("spirv.Pipe", 0) %pipe, ptr addrspace(4) %data) {
  %r = call spir_func i32 @_Z13__read_pipe_211ocl_pipe_roPiii(target("spirv.Pipe", 0) %pipe, ptr addrspace(4) %data, i32 4, i32 4)
  ret void
}
