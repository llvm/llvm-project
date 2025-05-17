; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-NO-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - -filetype=obj | spirv-val %} 
;
; CHECK-EXTENSION: Capability BitInstructions
; CHECK-EXTENSION: Extension "SPV_KHR_bit_instructions"
; CHECK-NO-EXTENSION-NOT: Capability BitInstructions 
; CHECK-NO-EXTENSION-NOT: Extension "SPV_KHR_bit_instructions"
; CHECK-NO-EXTENSION: Capability Shader 
;
; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_SPIRVFriendly(long4 b, global long4 *res) {
;   *res = bit_reverse(b);
; }
define spir_kernel void @testBitReverse_SPIRVFriendly(<4 x i64> %b, ptr addrspace(1) nocapture align 32 %res) #3 {
entry:
  %call = call <4 x i64> @llvm.bitreverse.v4i64(<4 x i64> %b)
  store <4 x i64> %call, ptr addrspace(1) %res, align 32
  ret void
}

declare <4 x i64> @llvm.bitreverse.v4i64(<4 x i64>) #4


attributes #3 = { nounwind }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git cc61409d353a40f62d3a137f3c7436aa00df779d)"}
