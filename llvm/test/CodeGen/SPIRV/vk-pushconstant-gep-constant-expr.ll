; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s

%struct.PushConstants = type { <4 x float>, <4 x float>, i32 }

@push_constants = external hidden addrspace(13) externally_initialized global %struct.PushConstants, align 1
@result = external addrspace(1) global i32, align 4

; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#S_T:]] = OpTypeStruct {{.*}} {{.*}} %[[#INT32]]
; CHECK-DAG: %[[#PTR_PCS_S:]] = OpTypePointer PushConstant %[[#S_T]]
; CHECK-DAG: %[[#PCS:]] = OpVariable %[[#PTR_PCS_S]] PushConstant
; CHECK-DAG: %[[#PTR_I32:]] = OpTypePointer PushConstant %[[#INT32]]
; CHECK-DAG: %[[#INT_2:]] = OpConstant %[[#INT32]] 2

define void @main() #0 {
entry:
  ; This is the case where GEP is embedded in the load as a constant expression.
  ; We want to ensure that it gets lowered correctly to instructions.
  ; CHECK: %[[#GEP:]] = OpInBoundsAccessChain %[[#PTR_I32]] %[[#PCS]] %[[#INT_2]]
  ; CHECK: %[[#VAL:]] = OpLoad %[[#INT32]] %[[#GEP]]
  %0 = load i32, ptr addrspace(13) getelementptr inbounds nuw (%struct.PushConstants, ptr addrspace(13) @push_constants, i32 0, i32 2), align 1
  %1 = add i32 %0, 1
  store i32 %1, ptr addrspace(1) @result, align 4
  ret void
}

attributes #0 = { convergent noinline norecurse optnone "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
