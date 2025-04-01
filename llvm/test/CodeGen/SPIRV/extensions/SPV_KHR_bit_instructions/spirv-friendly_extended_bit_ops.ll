; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - -filetype=obj | spirv-val %} 
;
; CHECK-EXTENSION: Capability BitInstructions
; CHECK-EXTENSION: Extension "SPV_KHR_bit_instructions"
; CHECK-NO-EXTENSION: LLVM ERROR: __spirv_BitFieldInsert: the builtin requires the following SPIR-V extension: SPV_KHR_bit_instructions

; Test SPIRV-friendly builtins.
; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_int2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_int2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int2]] %[[#insertinsert_int2]]
; OpenCL equivalent.
; kernel void testInsert_SPIRVFriendly(int2 b, int2 i, global int2 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }
define spir_kernel void @testInsert_SPIRVFriendly(<2 x i32> %b, <2 x i32> %i, ptr addrspace(1) nocapture align 8 %res) #3 {
entry:
  %call = call spir_func <2 x i32> @_Z22__spirv_BitFieldInsertDv2_iS_jj(<2 x i32> %b, <2 x i32> %i, i32 4, i32 2) #3
  store <2 x i32> %call, ptr addrspace(1) %res, align 8
  ret void
}

declare spir_func <2 x i32> @_Z22__spirv_BitFieldInsertDv2_iS_jj(<2 x i32>, <2 x i32>, i32, i32) #3


; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_SPIRVFriendly(short b, ushort bu, global short *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }
define spir_kernel void @testExtractS_SPIRVFriendly(i16 signext %b, i16 zeroext %bu, ptr addrspace(1) nocapture align 2 %res) #3 {
entry:
  %call = call spir_func i16 @_Z24__spirv_BitFieldSExtractsjj(i16 %b, i32 5, i32 4) #3
  %call1 = call spir_func i16 @_Z24__spirv_BitFieldSExtractsjj(i16 %bu, i32 5, i32 4) #3
  %add = add i16 %call1, %call
  store i16 %add, ptr addrspace(1) %res, align 2
  ret void
}

declare spir_func i16 @_Z24__spirv_BitFieldSExtractsjj(i16, i32, i32) #3

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_SPIRVFriendly(char8 b, uchar8 bu, global uchar8 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }
define spir_kernel void @testExtractU_SPIRVFriendly(<8 x i8> %b, <8 x i8> %bu, ptr addrspace(1) nocapture align 8 %res) #3 {
entry:
  %call = call spir_func <8 x i8> @_Z24__spirv_BitFieldUExtractDv8_hjj(<8 x i8> %b, i32 3, i32 4) #3
  %call1 = call spir_func <8 x i8> @_Z24__spirv_BitFieldUExtractDv8_hjj(<8 x i8> %bu, i32 3, i32 4) #3
  %add = add <8 x i8> %call1, %call
  store <8 x i8> %add, ptr addrspace(1) %res, align 8
  ret void
}

declare spir_func <8 x i8> @_Z24__spirv_BitFieldUExtractDv8_hjj(<8 x i8>, i32, i32) #3

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
