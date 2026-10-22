; Per-kernel implicit ContractionOff, as in SPIRV-LLVM-Translator: a kernel
; gets it iff it or a callee has an uncontracted fadd/fsub, a call to a
; non-intrinsic declaration, an indirect call or inline asm.

; RUN: sed 's/FNPTR/ptr/' %s | llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_inline_assembly --spv-allow-unknown-intrinsics=llvm.readcyclecounter - -o - | FileCheck %s --check-prefixes=CHECK,CO
; RUN: sed 's/FNPTR/ptr/' %s | llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_inline_assembly --spv-allow-unknown-intrinsics=llvm.readcyclecounter --spirv-fp-contract=on - -o - | FileCheck %s --check-prefixes=CHECK,CO
; RUN: %if spirv-tools %{ sed 's/FNPTR/ptr/' %s | llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_inline_assembly --spv-allow-unknown-intrinsics=llvm.readcyclecounter --spirv-fp-contract=on - -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ sed 's/FNPTR/ptr/' %s | llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_inline_assembly --spv-allow-unknown-intrinsics=llvm.readcyclecounter - -o - -filetype=obj | spirv-val %}
; AMD triples get all extensions by default, so ContractionOff becomes FPFastMathDefault.
; RUN: sed 's/FNPTR/ptr addrspace(4)/' %s | llc -O0 -mtriple=spirv64-amd-amdhsa - -o - | FileCheck %s --check-prefixes=CHECK,FC2 --implicit-check-not=ContractionOff
; RUN: %if spirv-tools %{ sed 's/FNPTR/ptr addrspace(4)/' %s | llc -O0 -mtriple=spirv64-amd-amdhsa - -o - -filetype=obj | spirv-val %}

; =fast: no mode.
; RUN: sed 's/FNPTR/ptr/' %s | llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_inline_assembly --spv-allow-unknown-intrinsics=llvm.readcyclecounter --spirv-fp-contract=fast - -o - | FileCheck %s --check-prefixes=CHECK,FAST
; RUN: %if spirv-tools %{ sed 's/FNPTR/ptr/' %s | llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_inline_assembly --spv-allow-unknown-intrinsics=llvm.readcyclecounter --spirv-fp-contract=fast - -o - -filetype=obj | spirv-val %}
; RUN: sed 's/FNPTR/ptr addrspace(4)/' %s | llc -O0 -mtriple=spirv64-amd-amdhsa --spirv-fp-contract=fast - -o - | FileCheck %s --check-prefixes=CHECK,FAST
; RUN: %if spirv-tools %{ sed 's/FNPTR/ptr addrspace(4)/' %s | llc -O0 -mtriple=spirv64-amd-amdhsa --spirv-fp-contract=fast - -o - -filetype=obj | spirv-val %}

; FC2: OpCapability FloatControls2
; FC2: OpExtension "SPV_KHR_float_controls2"
; CHECK-DAG: OpEntryPoint Kernel %[[#K1:]] "kernel_off_1"
; CHECK-DAG: OpEntryPoint Kernel %[[#K2:]] "kernel_off_2"
; CHECK-DAG: OpEntryPoint Kernel %[[#K3:]] "kernel_off_3"
; CHECK-DAG: OpEntryPoint Kernel %[[#K4:]] "kernel_on_4"
; CHECK-DAG: OpEntryPoint Kernel %[[#K5:]] "kernel_on_5"
; CHECK-DAG: OpEntryPoint Kernel %[[#K6:]] "kernel_off_6"
; CHECK-DAG: OpEntryPoint Kernel %[[#K7:]] "kernel_off_7"
; CHECK-DAG: OpEntryPoint Kernel %[[#K8:]] "kernel_off_8"
; CHECK-DAG: OpEntryPoint Kernel %[[#K9:]] "kernel_off_9"
; CHECK-DAG: OpEntryPoint Kernel %[[#K10:]] "kernel_off_10"
; CHECK-DAG: OpEntryPoint Kernel %[[#K11:]] "kernel_on_11"
; CHECK-DAG: OpEntryPoint Kernel %[[#K12:]] "kernel_off_12"
; FAST-NOT: ContractionOff
; FAST-NOT: FPFastMathDefault

; Modes are emitted in module order.
; CO-NOT: ContractionOff
; CO: OpExecutionMode %[[#K1]] ContractionOff
; CO: OpExecutionMode %[[#K2]] ContractionOff
; CO: OpExecutionMode %[[#K3]] ContractionOff
; CO-NOT: ContractionOff
; CO: OpExecutionMode %[[#K6]] ContractionOff
; CO: OpExecutionMode %[[#K7]] ContractionOff
; CO: OpExecutionMode %[[#K8]] ContractionOff
; CO: OpExecutionMode %[[#K9]] ContractionOff
; CO: OpExecutionMode %[[#K10]] ContractionOff
; CO-NOT: ContractionOff
; CO: OpExecutionMode %[[#K12]] ContractionOff
; CO-NOT: ContractionOff

; FC2-NOT: FPFastMathDefault
; FC2: OpExecutionModeId %[[#K1]] FPFastMathDefault %[[#FLOAT:]] %[[#ZERO:]]
; FC2: OpExecutionModeId %[[#K2]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; FC2: OpExecutionModeId %[[#K3]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; FC2-NOT: FPFastMathDefault
; FC2: OpExecutionModeId %[[#K6]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; FC2: OpExecutionModeId %[[#K7]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; FC2: OpExecutionModeId %[[#K8]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; FC2: OpExecutionModeId %[[#K9]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; FC2: OpExecutionModeId %[[#K10]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; FC2-NOT: FPFastMathDefault
; FC2: OpExecutionModeId %[[#K12]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; FC2-NOT: FPFastMathDefault
; FC2-DAG: %[[#INT:]] = OpTypeInt 32 0
; FC2-DAG: %[[#FLOAT]] = OpTypeFloat 32
; FC2-DAG: %[[#ZERO]] = OpConstantNull %[[#INT]]

; Off: calls @func_off.
define float @func_nested_off(float %a, float %b, float %c) {
entry:
  %0 = call float @func_off(float %a, float %b, float %c)
  ret float %0
}

; Off: uncontracted fadd.
define float @func_off(float %a, float %b, float %c) {
entry:
  %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  %mul = fmul float %0, %b
  %add = fadd float %mul, %c
  ret float %add
}

; On: no fadd/fsub.
define float @func_mul(float %a, float %b) {
entry:
  %0 = fmul float %a, %b
  ret float %0
}

; Off for callers: external declaration.
declare float @func_extern(float, float, float)

; Off: uncontracted fadd.
define spir_kernel void @kernel_off_1(float %a, float %b, float %c, ptr addrspace(1) %p) {
entry:
  %mul = fmul float %a, %b
  %add = fadd float %mul, %c
  store float %add, ptr addrspace(1) %p
  ret void
}

; Off: reaches @func_off through @func_nested_off.
define spir_kernel void @kernel_off_2(float %a, float %b, float %c, ptr addrspace(1) %p) {
entry:
  %call = call float @func_nested_off(float %a, float %b, float %c)
  store float %call, ptr addrspace(1) %p
  ret void
}

; Off: calls external declaration.
define spir_kernel void @kernel_off_3(float %a, float %b, float %c, ptr addrspace(1) %p) {
entry:
  %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  %call = call float @func_extern(float %0, float %b, float %c)
  store float %call, ptr addrspace(1) %p
  ret void
}

; On: contracted fadd; callee has no fadd/fsub.
define spir_kernel void @kernel_on_4(float %a, float %b, float %c, ptr addrspace(1) %p) {
entry:
  %mul = call float @func_mul(float %a, float %b)
  %add = fadd contract float %mul, %c
  store float %add, ptr addrspace(1) %p
  ret void
}

; On: unknown intrinsic becomes a call to a @spirv.llvm_* declaration.
define spir_kernel void @kernel_on_5(ptr addrspace(1) %p) {
entry:
  %c = call i64 @llvm.readcyclecounter()
  store i64 %c, ptr addrspace(1) %p
  ret void
}

; Off: indirect call.
; Function pointers are in the program address space, 4 on AMD.
define spir_kernel void @kernel_off_6(float %a, FNPTR %fp, ptr addrspace(1) %p) {
entry:
  %call = call float %fp(float %a)
  store float %call, ptr addrspace(1) %p
  ret void
}

; Off: inline asm.
define spir_kernel void @kernel_off_7(float %a, ptr addrspace(1) %p) {
entry:
  call void asm sideeffect "nop", ""()
  store float %a, ptr addrspace(1) %p
  ret void
}

; Off: uncontracted vector fadd.
define spir_kernel void @kernel_off_8(<2 x float> %a, <2 x float> %b, ptr addrspace(1) %p) {
entry:
  %add = fadd <2 x float> %a, %b
  store <2 x float> %add, ptr addrspace(1) %p
  ret void
}

; Off: uncontracted fsub.
define spir_kernel void @kernel_off_9(float %a, float %b, ptr addrspace(1) %p) {
entry:
  %sub = fsub float %a, %b
  store float %sub, ptr addrspace(1) %p
  ret void
}

; Off: uncontracted vector fsub.
define spir_kernel void @kernel_off_10(<2 x float> %a, <2 x float> %b, ptr addrspace(1) %p) {
entry:
  %sub = fsub <2 x float> %a, %b
  store <2 x float> %sub, ptr addrspace(1) %p
  ret void
}

; On: scalar and vector fadd/fsub all contracted.
define spir_kernel void @kernel_on_11(float %a, <2 x float> %va, <2 x float> %vb, ptr addrspace(1) %p, ptr addrspace(1) %vp) {
entry:
  %sub = fsub contract float %a, %a
  store float %sub, ptr addrspace(1) %p
  %vadd = fadd contract <2 x float> %va, %vb
  %vsub = fsub contract <2 x float> %vadd, %vb
  store <2 x float> %vsub, ptr addrspace(1) %vp
  ret void
}

; Call cycle; @cycle_b disables.
define float @cycle_a(float %a, i32 %n) {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %done, label %rec
rec:
  %n1 = sub i32 %n, 1
  %r = call float @cycle_b(float %a, i32 %n1)
  ret float %r
done:
  ret float %a
}

define float @cycle_b(float %a, i32 %n) {
entry:
  %r = call float @cycle_a(float %a, i32 %n)
  %add = fadd float %r, %a
  ret float %add
}

; Off: reaches @cycle_b.
define spir_kernel void @kernel_off_12(float %a, i32 %n, ptr addrspace(1) %p) {
entry:
  %call = call float @cycle_a(float %a, i32 %n)
  store float %call, ptr addrspace(1) %p
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float)
declare i64 @llvm.readcyclecounter()
