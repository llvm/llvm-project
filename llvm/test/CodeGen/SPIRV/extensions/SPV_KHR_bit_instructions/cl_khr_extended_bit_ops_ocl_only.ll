; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - -filetype=obj | spirv-val %} 
;
; if OpenCL builtin calls present in LLVM IR input and SPV_KHR_bit_instructions is enabled, no error and BitInstructions capability generated.
; CHECK-EXTENSION: Capability BitInstructions
; CHECK-EXTENSION-NOT: Capability Shader
; CHECK-EXTENSION: Extension "SPV_KHR_bit_instructions"
; if OpenCL builtin calls present in LLVM IR input and SPV_KHR_bit_instructions is NOT enabled, error.
; CHECK-NO-EXTENSION: LLVM ERROR: bitfield_insert: the builtin requires the following SPIR-V extension: SPV_KHR_bit_instructions
;
; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_long:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_long:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long]] %[[#insertinsert_long]]
;
; OpenCL equivalent.
; kernel void testInsert_long(long b, long i, global long *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }
define dso_local spir_kernel void @testInsert_long(i64 noundef %b, i64 noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z15bitfield_insertlljj(i64 noundef %b, i64 noundef %i, i32 noundef 4, i32 noundef 2) #2
  store i64 %call, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

declare spir_func i64 @_Z15bitfield_insertlljj(i64 noundef, i64 noundef, i32 noundef, i32 noundef) 

attributes #2 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git cc61409d353a40f62d3a137f3c7436aa00df779d)"}
!7 = !{!8, !8, i64 0}
!8 = !{!"long", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
