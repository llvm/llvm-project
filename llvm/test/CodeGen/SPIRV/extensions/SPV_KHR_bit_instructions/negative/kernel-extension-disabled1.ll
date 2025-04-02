; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown-opencl %s --spirv-ext=+SPV_KHR_bit_instructions -o - -filetype=obj | spirv-val %} 

; if spir_kernel present in LLVM IR input and SPV_KHR_bit_instructions is NOT enabled, error.
; CHECK-NO-EXTENSION: LLVM ERROR: This entry point lacks mandatory hlsl.shader attribute.
;
; OpenCL equivalent.
; kernel void testBitReverse_SPIRVFriendly(long4 b, global long4 *res) {
;   *res = bit_reverse(b);
; }
define spir_kernel void @testBitReverse_SPIRVFriendly_kernel(<4 x i64> %b, ptr addrspace(1) nocapture align 32 %res) #3 {
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
