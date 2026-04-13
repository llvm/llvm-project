; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_NV_shader_atomic_fp16_vector %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: The atomic float16 vector instruction requires the following SPIR-V extension: SPV_NV_shader_atomic_fp16_vector

; CHECK: Capability Float16
; CHECK-DAG: Capability AtomicFloat16VectorNV
; CHECK: Extension "SPV_NV_shader_atomic_fp16_vector"
; CHECK-DAG: %[[TyF16:[0-9]+]] = OpTypeFloat 16
; CHECK-DAG: %[[TyF16Vec2:[0-9]+]] = OpTypeVector %[[TyF16]] 2
; CHECK-DAG: %[[TyF16Vec4:[0-9]+]] = OpTypeVector %[[TyF16]] 4
; CHECK-DAG: %[[TyF16Vec4Ptr:[0-9]+]] = OpTypePointer {{[a-zA-Z]+}} %[[TyF16Vec4]]
; CHECK-DAG: %[[TyF16Vec2Ptr:[0-9]+]] = OpTypePointer {{[a-zA-Z]+}} %[[TyF16Vec2]]
; CHECK-DAG: %[[TyInt32:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[ConstF16:[0-9]+]] = OpConstant %[[TyF16]] 20800{{$}}
; CHECK-DAG: %[[Const0F16Vec2:[0-9]+]] = OpConstantNull %[[TyF16Vec2]]
; CHECK-DAG: %[[f:[0-9]+]] = OpVariable %[[TyF16Vec2Ptr]] CrossWorkgroup %[[Const0F16Vec2]]
; CHECK-DAG: %[[Const0F16Vec4:[0-9]+]] = OpConstantNull %[[TyF16Vec4]]
; CHECK-DAG: %[[g:[0-9]+]] = OpVariable %[[TyF16Vec4Ptr]] CrossWorkgroup %[[Const0F16Vec4]]
; CHECK-DAG: %[[ConstF16Vec2:[0-9]+]] = OpConstantComposite %[[TyF16Vec2]] %[[ConstF16]] %[[ConstF16]]
; CHECK-DAG: %[[ScopeAllSvmDevices:[0-9]+]] = OpConstantNull %[[TyInt32]]
; CHECK-DAG: %[[MemSeqCst:[0-9]+]] = OpConstant %[[TyInt32]] 16{{$}}
; CHECK-DAG: %[[ConstF16Vec4:[0-9]+]] = OpConstantComposite %[[TyF16Vec4]] %[[ConstF16]] %[[ConstF16]] %[[ConstF16]] %[[ConstF16]]

@f = common dso_local local_unnamed_addr addrspace(1) global <2 x half> <half 0.000000e+00, half 0.000000e+00>
@g = common dso_local local_unnamed_addr addrspace(1) global <4 x half> <half 0.000000e+00, half 0.000000e+00, half 0.000000e+00, half 0.000000e+00>

; CHECK-DAG: OpAtomicFAddEXT %[[TyF16Vec2]] %[[f]] %[[ScopeAllSvmDevices]] %[[MemSeqCst]] %[[ConstF16Vec2]]
; CHECK: %[[NegatedConstF16Vec2:[0-9]+]] = OpFNegate %[[TyF16Vec2]] %[[ConstF16Vec2]]
; CHECK: OpAtomicFAddEXT %[[TyF16Vec2]] %[[f]] %[[ScopeAllSvmDevices]] %[[MemSeqCst]] %[[NegatedConstF16Vec2]]
define dso_local spir_func void @test1() local_unnamed_addr {
entry:
  %addval = atomicrmw fadd ptr addrspace(1) @f, <2 x half> <half 42.000000e+00, half 42.000000e+00> seq_cst
  %subval = atomicrmw fsub ptr addrspace(1) @f, <2 x half> <half 42.000000e+00, half 42.000000e+00> seq_cst
  ret void
}

; CHECK-DAG: OpAtomicFAddEXT %[[TyF16Vec4]] %[[g]] %[[ScopeAllSvmDevices]] %[[MemSeqCst]] %[[ConstF16Vec4]]
; CHECK: %[[NegatedConstF16Vec4:[0-9]+]] = OpFNegate %[[TyF16Vec4]] %[[ConstF16Vec4]]
; CHECK: OpAtomicFAddEXT %[[TyF16Vec4]] %[[g]] %[[ScopeAllSvmDevices]] %[[MemSeqCst]] %[[NegatedConstF16Vec4]]
define dso_local spir_func void @test2() local_unnamed_addr {
entry:
  %addval = atomicrmw fadd ptr addrspace(1) @g, <4 x half> <half 42.000000e+00, half 42.000000e+00, half 42.000000e+00, half 42.000000e+00> seq_cst
  %subval = atomicrmw fsub ptr addrspace(1) @g, <4 x half> <half 42.000000e+00, half 42.000000e+00, half 42.000000e+00, half 42.000000e+00> seq_cst
  ret void
}
