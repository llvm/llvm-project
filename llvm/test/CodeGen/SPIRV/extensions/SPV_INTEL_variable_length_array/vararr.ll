; Modified from: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/test/extensions/INTEL/SPV_INTEL_variable_length_array/basic.ll

; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - -filetype=obj | spirv-val %}

; TODO: currently spirv-val mistakenly rejects Element Type as operand of OpUntypedVariableLengthArrayINTEL. Re-enable spirv-val once it's fixed.
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array,+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array,+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-SPIRV-UNTYPED

; CHECK-ERROR: LLVM ERROR: array allocation: this instruction requires the following SPIR-V extension: SPV_INTEL_variable_length_array

; CHECK-COMMON: Capability VariableLengthArrayINTEL
; CHECK-SPIRV-UNTYPED: Capability UntypedVariableLengthArrayINTEL
; CHECK-COMMON-DAG: Extension "SPV_INTEL_variable_length_array"
; CHECK-SPIRV-UNTYPED-DAG: Extension "SPV_KHR_untyped_pointers"

; CHECK-COMMON-DAG: OpName %[[Len:.*]] "a"
; CHECK-COMMON-DAG: %[[Long:.*]] = OpTypeInt 64 0
; CHECK-COMMON-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[Char:.*]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[CharPtr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[Char]]
; CHECK-SPIRV-DAG: %[[IntPtr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[Int]]
; CHECK-SPIRV-UNTYPED-DAG: %[[CharPtr:.*]] = OpTypeUntypedPointerKHR Function
; CHECK-COMMON: %[[Len]] = OpFunctionParameter %[[Long]]
; CHECK-COMMON: %[[SavedMem1:.*]] = OpSaveMemoryINTEL %[[CharPtr]]
; CHECK-SPIRV: OpVariableLengthArrayINTEL %[[IntPtr]] %[[Len]]
; CHECK-SPIRV-UNTYPED: OpUntypedVariableLengthArrayINTEL %[[CharPtr]] %[[Int]] %[[Len]]
; CHECK-COMMON: OpRestoreMemoryINTEL %[[SavedMem1]]
; CHECK-COMMON: %[[SavedMem2:.*]] = OpSaveMemoryINTEL %[[CharPtr]]
; CHECK-SPIRV: OpVariableLengthArrayINTEL %[[IntPtr]] %[[Len]]
; CHECK-SPIRV-UNTYPED: OpUntypedVariableLengthArrayINTEL %[[CharPtr]] %[[Int]] %[[Len]]
; CHECK-COMMON: OpRestoreMemoryINTEL %[[SavedMem2]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

define dso_local spir_func i32 @foo(i64 %a, i64 %b) {
entry:
  %vector1 = alloca [42 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 168, ptr nonnull %vector1)
  %stack1 = call ptr @llvm.stacksave.p0()
  %vla = alloca i32, i64 %a, align 16
  %arrayidx = getelementptr inbounds i32, ptr %vla, i64 %b
  %elem1 = load i32, ptr %arrayidx, align 4
  call void @llvm.stackrestore.p0(ptr %stack1)
  %stack2 = call ptr @llvm.stacksave.p0()
  %vla2 = alloca i32, i64 %a, align 16
  %arrayidx3 = getelementptr inbounds [42 x i32], ptr %vector1, i64 0, i64 %b
  %elemt = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %elemt, %elem1
  %arrayidx4 = getelementptr inbounds i32, ptr %vla2, i64 %b
  %elem2 = load i32, ptr %arrayidx4, align 4
  %add5 = add nsw i32 %add, %elem2
  call void @llvm.stackrestore.p0(ptr %stack2)
  call void @llvm.lifetime.end.p0(i64 168, ptr nonnull %vector1)
  ret i32 %add5
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare ptr @llvm.stacksave.p0()
declare void @llvm.stackrestore.p0(ptr)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
