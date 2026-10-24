; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED,PHYSICAL --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED,PHYSICAL --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_INTEL_variable_length_array %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED,PHYSICAL --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED,SHADER --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array

; Preserve non-entry allocation placement, including repeated allocations in
; loops. This deliberately emits Function variables outside the entry block,
; which core SPIR-V does not permit, so these cases cannot pass spirv-val.

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FOUR:]] = OpConstant %[[#INT]] 4{{$}}
; SHADER-DAG: %[[#ZERO:]] = OpConstant %[[#INT]] 0{{$}}
; CHECK-DAG: %[[#INT_ARRAY:]] = OpTypeArray %[[#INT]] %[[#FOUR]]
; TYPED-DAG: %[[#INT_PTR:]] = OpTypePointer Function %[[#INT]]
; TYPED-DAG: %[[#INT_ARRAY_PTR:]] = OpTypePointer Function %[[#INT_ARRAY]]
; UNTYPED-DAG: %[[#PTR:]] = OpTypeUntypedPointerKHR Function

; CHECK-LABEL: Begin function constant_count_loop
; CHECK: OpLabel
; CHECK-NOT: {{OpVariable|OpUntypedVariableKHR}}
; CHECK: OpBranch %[[#LOOP:]]
; CHECK: %[[#LOOP]] = OpLabel
; TYPED-NEXT: %[[#ARRAY:]] = OpVariable %[[#INT_ARRAY_PTR]] Function
; UNTYPED-NEXT: %[[#ARRAY:]] = OpUntypedVariableKHR %[[#PTR]] Function %[[#INT_ARRAY]]
; PHYSICAL: %[[#BASE:]] = OpBitcast %[[#INT_PTR]] %[[#ARRAY]]
; SHADER: %[[#BASE:]] = OpInBoundsAccessChain %[[#INT_PTR]] %[[#ARRAY]] %[[#ZERO]]
; TYPED: OpStore %[[#BASE]]
; UNTYPED: OpStore %[[#ARRAY]]
; CHECK: OpBranchConditional
define spir_func void @constant_count_loop(i1 %repeat) {
entry:
  br label %loop

loop:
  %array = alloca i32, i32 4, align 4
  store volatile i32 1, ptr %array, align 4
  br i1 %repeat, label %loop, label %exit

exit:
  ret void
}

; An existing array alloca must also retain its conditional placement.
; CHECK-LABEL: Begin function conditional_array
; CHECK: OpLabel
; CHECK-NOT: {{OpVariable|OpUntypedVariableKHR}}
; CHECK: OpBranchConditional %{{[0-9]+}} %[[#ALLOC:]] %[[#EXIT:]]
; CHECK: %[[#ALLOC]] = OpLabel
; TYPED-NEXT: %[[#ARRAY:]] = OpVariable %[[#INT_ARRAY_PTR]] Function
; UNTYPED-NEXT: %[[#ARRAY:]] = OpUntypedVariableKHR %[[#PTR]] Function %[[#INT_ARRAY]]
; PHYSICAL: %[[#BASE:]] = OpBitcast %[[#INT_PTR]] %[[#ARRAY]]
; SHADER: %[[#BASE:]] = OpInBoundsAccessChain %[[#INT_PTR]] %[[#ARRAY]] %[[#ZERO]]
; TYPED: OpStore %[[#BASE]]
; UNTYPED: OpStore %[[#ARRAY]]
; CHECK: OpBranch %[[#EXIT]]
define spir_func void @conditional_array(i1 %allocate) {
entry:
  br i1 %allocate, label %alloc, label %exit

alloc:
  %array = alloca [4 x i32], align 4
  store volatile i32 1, ptr %array, align 4
  br label %exit

exit:
  ret void
}
