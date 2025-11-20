; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

@PrivInternal = internal addrspace(10) global i32 456
; CHECK-DAG:  %[[#type:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ptrty:]] = OpTypePointer Private %[[#type]]
; CHECK-DAG: %[[#value:]] = OpConstant %[[#type]] 456
; CHECK-DAG:   %[[#var:]] = OpVariable %[[#ptrty]] Private %[[#value]]

define hidden spir_func void @Foo() {
  %p = addrspacecast ptr addrspace(10) @PrivInternal to ptr
  %v = load i32, ptr %p, align 4
  ret void
; CHECK:      OpLabel
; CHECK-NEXT: OpLoad %[[#type]] %[[#var]] Aligned 4
; CHECK-Next: OpReturn
}

define void @main() #1 {
  ret void
}

attributes #1 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
