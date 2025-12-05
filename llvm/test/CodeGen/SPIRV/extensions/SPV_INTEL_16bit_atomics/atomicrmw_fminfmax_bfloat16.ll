; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_shader_atomic_float_min_max,+SPV_KHR_bfloat16 %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_shader_atomic_float_min_max,+SPV_INTEL_16bit_atomics,+SPV_KHR_bfloat16 %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: The atomic bfloat16 instruction requires the following SPIR-V extension: SPV_INTEL_16bit_atomics

; CHECK: Capability AtomicBFloat16MinMaxINTEL
; CHECK: Extension "SPV_KHR_bfloat16"
; CHECK: Extension "SPV_EXT_shader_atomic_float_min_max"
; CHECK: Extension "SPV_INTEL_16bit_atomics"
; CHECK-DAG: %[[TyBF16:[0-9]+]] = OpTypeFloat 16 0
; CHECK-DAG: %[[TyBF16Ptr:[0-9]+]] = OpTypePointer {{[a-zA-Z]+}} %[[TyBF16]]
; CHECK-DAG: %[[TyInt32:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[ConstBF16:[0-9]+]] = OpConstant %[[TyBF16]] 16936{{$}}
; CHECK-DAG: %[[Const0:[0-9]+]] = OpConstantNull %[[TyBF16]]
; CHECK-DAG: %[[BF16Ptr:[0-9]+]] = OpVariable %[[TyBF16Ptr]] CrossWorkgroup %[[Const0]]
; CHECK-DAG: %[[ScopeAllSvmDevices:[0-9]+]] = OpConstantNull %[[TyInt32]]
; CHECK-DAG: %[[MemSeqCst:[0-9]+]] = OpConstant %[[TyInt32]] 16{{$}}
; CHECK: OpAtomicFMinEXT %[[TyBF16]] %[[BF16Ptr]] %[[ScopeAllSvmDevices]] %[[MemSeqCst]] %[[ConstBF16]]
; CHECK: OpAtomicFMaxEXT %[[TyBF16]] %[[BF16Ptr]] %[[ScopeAllSvmDevices]] %[[MemSeqCst]] %[[ConstBF16]]

@f = common dso_local local_unnamed_addr addrspace(1) global bfloat 0.000000e+00, align 8

define spir_func void @test1() {
entry:
  %minval = atomicrmw fmin ptr addrspace(1) @f, bfloat 42.0e+00 seq_cst
  %maxval = atomicrmw fmax ptr addrspace(1) @f, bfloat 42.0e+00 seq_cst
  ret void
}
