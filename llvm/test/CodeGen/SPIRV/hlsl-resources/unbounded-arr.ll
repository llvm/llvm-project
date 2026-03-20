; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability RuntimeDescriptorArrayEXT
; CHECK-DAG: %[[int32:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[rwbuffer:[0-9]+]] = OpTypeImage %[[int32]] Buffer 2 0 0 2 R32i
; CHECK-DAG: OpTypeRuntimeArray %[[rwbuffer]]

; This IR was emmited from the following HLSL code:
; [[vk::binding(0)]]
; RWBuffer<int> Buf[] : register(u0);
; 
; [numthreads(4,2,1)]
; void main(uint GI : SV_GroupIndex) {
;     Buf[0][0] = 0;
; }

@Buf.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

; Function Attrs: convergent noinline norecurse
define void @main() #0 {
entry:
  %binding = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 0, i32 0, ptr @Buf.str)
  %pointer = call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24)  %binding, i32 0)
  store i32 0, ptr addrspace(11) %pointer, align 4
  ret void
}

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="4,2,1" "hlsl.shader"="compute" }
