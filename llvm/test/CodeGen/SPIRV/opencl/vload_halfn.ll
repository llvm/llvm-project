; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#IMPORT:]] = OpExtInstImport "OpenCL.std"

; CHECK: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK: %[[#V2FLOAT:]] = OpTypeVector %[[#FLOAT]] 2

define void @test(i64 %a, ptr addrspace(1) %b) {
; CHECK: %[[#]] = OpExtInst %[[#V2FLOAT:]] %[[#IMPORT]] vload_halfn %[[#]] %[[#]] 2
  %c = call spir_func <2 x float> @_Z11vload_half2mPU3AS1KDh(i64 %a, ptr addrspace(1) %b)
  ret void
}

declare <2 x float> @_Z11vload_half2mPU3AS1KDh(i64, ptr addrspace(1))
