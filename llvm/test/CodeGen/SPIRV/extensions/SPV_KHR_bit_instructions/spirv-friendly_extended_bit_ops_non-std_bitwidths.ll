; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown-opencl %s --spirv-ext=+SPV_KHR_bit_instructions,+SPV_INTEL_arbitrary_precision_integers -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown-opencl %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown-opencl %s --spirv-ext=+SPV_KHR_bit_instructions,+SPV_INTEL_arbitrary_precision_integers -o - -filetype=obj | spirv-val %} 
;
; CHECK-EXTENSION: Capability BitInstructions
; CHECK-EXTENSION: Extension "SPV_KHR_bit_instructions"
; CHECK-NO-EXTENSION: LLVM ERROR: __spirv_BitFieldInsert: the builtin requires the following SPIR-V extension: SPV_KHR_bit_instructions

; Test SPIRV-friendly builtins.
; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_int2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_int2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int2]] %[[#insertinsert_int2]]
define spir_kernel void @testInsert_SPIRVFriendly(i4 %b, i4 %i, ptr addrspace(1) nocapture align 8 %res) #3 {
entry:
  %call = call spir_func i4 @_Z22__spirv_BitFieldInsertDv2_iS_jj(i4 %b, i4 %i, i4 4, i4 2) #3
  store i4 %call, ptr addrspace(1) %res, align 8
  ret void
}

declare spir_func i4 @_Z22__spirv_BitFieldInsertDv2_iS_jj(i4, i4, i4, i4) #3


; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
define spir_kernel void @testExtractS_SPIRVFriendly(i2 signext %b, i2 zeroext %bu, ptr addrspace(1) nocapture align 2 %res) #3 {
entry:
  %call = call spir_func i2 @_Z24__spirv_BitFieldSExtractsjj(i2 %b, i32 5, i32 4) #3
  %call1 = call spir_func i2 @_Z24__spirv_BitFieldSExtractsjj(i2 %bu, i32 5, i32 4) #3
  %add = add i2 %call1, %call
  store i2 %add, ptr addrspace(1) %res, align 2
  ret void
}

declare spir_func i2 @_Z24__spirv_BitFieldSExtractsjj(i2, i32, i32) #3

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
define spir_kernel void @testExtractU_SPIRVFriendly(i5 %b, i5 %bu, ptr addrspace(1) nocapture align 8 %res) #3 {
entry:
  %call = call spir_func i5 @_Z24__spirv_BitFieldUExtractDv8_hjj(i5 %b, i32 3, i32 4) #3
  %call1 = call spir_func i5 @_Z24__spirv_BitFieldUExtractDv8_hjj(i5 %bu, i32 3, i32 4) #3
  %add = add i5 %call1, %call
  store i5 %add, ptr addrspace(1) %res, align 8
  ret void
}

declare spir_func i5 @_Z24__spirv_BitFieldUExtractDv8_hjj(i5, i32, i32) #3

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
define spir_kernel void @testBitReverse_SPIRVFriendly(i3 %b, ptr addrspace(1) nocapture align 32 %res) #3 {
entry:
  %call = call i3 @llvm.bitreverse.v4i3(i3 %b)
  store i3 %call, ptr addrspace(1) %res, align 32
  ret void
}

declare i3 @llvm.bitreverse.v4i3(i3) #4



attributes #3 = { nounwind }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git cc61409d353a40f62d3a137f3c7436aa00df779d)"}
