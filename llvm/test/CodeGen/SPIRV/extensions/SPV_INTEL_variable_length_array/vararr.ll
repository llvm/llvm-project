; Modified from: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/test/extensions/INTEL/SPV_INTEL_variable_length_array/basic.ll

; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - -filetype=obj | spirv-val %}

; TODO: currently spirv-val mistakenly rejects Element Type as operand of OpUntypedVariableLengthArrayINTEL. Re-enable spirv-val once it's fixed.
; Issue: https://github.com/KhronosGroup/SPIRV-Tools/issues/6921
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array,+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array,+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-SPIRV-UNTYPED

; CHECK-ERROR: LLVM ERROR: array allocation: this instruction requires the following SPIR-V extension: SPV_INTEL_variable_length_array

; CHECK-SPIRV-UNTYPED: Capability UntypedPointersKHR
; CHECK-COMMON: Capability VariableLengthArrayINTEL
; CHECK-SPIRV-UNTYPED: Capability UntypedVariableLengthArrayINTEL
; CHECK-COMMON-DAG: Extension "SPV_INTEL_variable_length_array"
; CHECK-SPIRV-UNTYPED-DAG: Extension "SPV_KHR_untyped_pointers"

; CHECK-COMMON-DAG: OpName %[[LenN:.*]] "n"
; CHECK-COMMON-DAG: OpName %[[SVLA:.*]] "svla"
; CHECK-COMMON-DAG: OpDecorate %[[SVLA]] Alignment 8

; CHECK-COMMON-DAG: OpName %[[Len:.*]] "a"
; CHECK-COMMON-DAG: %[[Long:.*]] = OpTypeInt 64 0
; CHECK-COMMON-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[Char:.*]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[CharPtr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[Char]]
; CHECK-SPIRV-DAG: %[[IntPtr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[Int]]
; CHECK-COMMON-DAG: %[[Float:.*]] = OpTypeFloat 32
; CHECK-COMMON-DAG: %[[Struct:.*]] = OpTypeStruct %[[Int]] %[[Float]]
; CHECK-SPIRV-DAG: %[[StructPtr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[Struct]]
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

%struct.S = type { i32, float }

define dso_local spir_func i32 @foo(i64 %a, i64 %b) {
entry:
  %vector1 = alloca [42 x i32], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %vector1)
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
  call void @llvm.lifetime.end.p0(ptr nonnull %vector1)
  ret i32 %add5
}

; Test an aggregate element type instead of a scalar
; CHECK-COMMON: %[[LenN]] = OpFunctionParameter %[[Long]]
; CHECK-COMMON: %[[SavedMem3:.*]] = OpSaveMemoryINTEL %[[CharPtr]]
; CHECK-SPIRV: %[[SVLA]] = OpVariableLengthArrayINTEL %[[StructPtr]] %[[LenN]]
; CHECK-SPIRV-UNTYPED: %[[SVLA]] = OpUntypedVariableLengthArrayINTEL %[[CharPtr]] %[[Struct]] %[[LenN]]
; CHECK-COMMON: OpRestoreMemoryINTEL %[[SavedMem3]]
define dso_local spir_func i32 @bar(i64 %n, i64 %i) {
entry:
  %stack = call ptr @llvm.stacksave.p0()
  %svla = alloca %struct.S, i64 %n, align 8
  %arrayidx = getelementptr inbounds %struct.S, ptr %svla, i64 %i, i32 0
  %elem = load i32, ptr %arrayidx, align 4
  call void @llvm.stackrestore.p0(ptr %stack)
  ret i32 %elem
}

; A constant count outside the entry block keeps the variable-length lowering:
; each execution allocates a new object.
; CHECK-COMMON: Begin function constant_count_loop
; CHECK-SPIRV: OpVariableLengthArrayINTEL %[[IntPtr]]
; CHECK-SPIRV-UNTYPED: OpUntypedVariableLengthArrayINTEL %[[CharPtr]] %[[Int]]
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

; Lifetime markers on a VLA. The pointer returned by OpVariableLengthArrayINTEL
; only points to the element type, so Size 0 would not cover the allocation.
; With a constant element count the byte size is known: it is kept, and the
; pointer is bitcast to uchar* because a non-zero Size requires a void/uchar
; pointer. With a dynamic count there is no size to keep, so Size 0 is emitted
; directly on the array pointer and no bitcast is inserted (unchanged behavior).
; CHECK-SPIRV: %[[ConstArr:.*]] = OpVariableLengthArrayINTEL %[[IntPtr]]
; CHECK-SPIRV: %[[Cast1:.*]] = OpBitcast %[[CharPtr]] %[[ConstArr]]
; CHECK-SPIRV-NEXT: OpLifetimeStart %[[Cast1]] 16
; CHECK-SPIRV: %[[Cast2:.*]] = OpBitcast %[[CharPtr]] %[[ConstArr]]
; CHECK-SPIRV-NEXT: OpLifetimeStop %[[Cast2]] 16
define dso_local spir_func void @const_count_lifetime() {
entry:
  %vla = alloca i32, i32 4, align 4
  call void @llvm.lifetime.start.p0(ptr %vla)
  store i32 0, ptr %vla, align 4
  call void @llvm.lifetime.end.p0(ptr %vla)
  ret void
}

; CHECK-SPIRV: %[[DynArr:.*]] = OpVariableLengthArrayINTEL %[[IntPtr]]
; CHECK-SPIRV-NEXT: OpLifetimeStart %[[DynArr]] 0
; CHECK-SPIRV: OpLifetimeStop %[[DynArr]] 0
define dso_local spir_func void @dyn_count_lifetime(i64 %n) {
entry:
  %vla = alloca i32, i64 %n, align 4
  call void @llvm.lifetime.start.p0(ptr %vla)
  store i32 0, ptr %vla, align 4
  call void @llvm.lifetime.end.p0(ptr %vla)
  ret void
}

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare ptr @llvm.stacksave.p0()
declare void @llvm.stackrestore.p0(ptr)
declare void @llvm.lifetime.end.p0(ptr nocapture)
