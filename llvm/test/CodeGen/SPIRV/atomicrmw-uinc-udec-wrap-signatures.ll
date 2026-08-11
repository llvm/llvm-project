; A module may contain uinc_wrap/udec_wrap atomics with mutually incompatible
; signatures, while SPIR-V resolves an imported function by its linkage name
; alone. Verify that the _p<addrspace>_<type> suffix keeps them apart: each
; distinct (address space, value type) combination gets its own declaration,
; and every call site of a given combination shares that one declaration. The
; value type may be a fixed vector of integers, spelled the LLVM way: v2i32.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Long:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Vec:]] = OpTypeVector %[[#Int]] 2
; CHECK-DAG: %[[#Value:]] = OpConstant %[[#Int]] 42
; CHECK-DAG: %[[#Value64:]] = OpConstant %[[#Long]] 42
; CHECK-DAG: %[[#ValueVec:]] = OpConstantComposite %[[#Vec]] %[[#Value]] %[[#Value]]
; CHECK-DAG: %[[#Scope_CrossDevice:]] = OpConstantNull %[[#Int]]
; The storage class contributes to the memory semantics, so an atomic on a
; Workgroup pointer carries WorkgroupMemory (256) where a CrossWorkgroup one
; carries CrossWorkgroupMemory (512).
; CHECK-DAG: %[[#MemSem_Relaxed_Local:]] = OpConstant %[[#Int]] 256
; CHECK-DAG: %[[#MemSem_Relaxed:]] = OpConstant %[[#Int]] 512

; CHECK-DAG: OpDecorate %[[#UIncFn:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_i32" Import
; CHECK-DAG: OpDecorate %[[#UIncFnLocal:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p3_i32" Import
; CHECK-DAG: OpDecorate %[[#UIncFn64:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_i64" Import
; CHECK-DAG: OpDecorate %[[#UIncFnVec:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_v2i32" Import
; CHECK-DAG: OpDecorate %[[#UDecFn:]] LinkageAttributes "__translate_spirv_atomic_udec_wrap_p1_i32" Import

@ui = common dso_local addrspace(1) global i32 0, align 4
@lui = common dso_local addrspace(3) global i32 0, align 4
@ul = common dso_local addrspace(1) global i64 0, align 8
@uv = common dso_local addrspace(1) global <2 x i32> zeroinitializer, align 8

; Two atomics of the same value type in different address spaces need two
; incompatible signatures, so they must resolve to two distinct declarations.

; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_Relaxed]] %[[#Value]]
; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFnLocal]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_Relaxed_Local]] %[[#Value]]
define dso_local spir_func void @mixed_addrspace() local_unnamed_addr {
entry:
  %g = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic
  %l = atomicrmw uinc_wrap ptr addrspace(3) @lui, i32 42 monotonic
  ret void
}

; Likewise for two atomics in the same address space with different value
; widths: the i64 one must not reuse the i32 declaration, and its call must
; yield an i64 result.

; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_Relaxed]] %[[#Value]]
; CHECK: OpFunctionCall %[[#Long]] %[[#UIncFn64]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_Relaxed]] %[[#Value64]]
define dso_local spir_func void @mixed_width() local_unnamed_addr {
entry:
  %a = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic
  %b = atomicrmw uinc_wrap ptr addrspace(1) @ul, i64 42 monotonic
  ret void
}

; uinc_wrap and udec_wrap on the same (address space, value type) are still two
; separate symbols, and repeated call sites reuse the single declaration.

; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_Relaxed]] %[[#Value]]
; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_Relaxed]] %[[#Value]]
define dso_local spir_func void @shared_declaration() local_unnamed_addr {
entry:
  %a = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic
  %b = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 monotonic
  ret void
}

; A vector operand is carried across just like a scalar one, under its own
; symbol, as long as it fits in the target's maximum atomic size. Anything
; wider is rejected by AtomicExpandPass before this lowering runs; see
; atomicrmw-uinc-udec-wrap-oversized.ll.

; CHECK: OpFunctionCall %[[#Vec]] %[[#UIncFnVec]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_Relaxed]] %[[#ValueVec]]
define dso_local spir_func void @vector_value() local_unnamed_addr {
entry:
  %v = atomicrmw uinc_wrap ptr addrspace(1) @uv, <2 x i32> splat (i32 42) monotonic
  ret void
}
