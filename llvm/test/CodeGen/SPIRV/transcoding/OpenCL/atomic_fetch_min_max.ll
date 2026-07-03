; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; This test checks that the backend correctly translates OpenCL C
;; atomic_fetch_min/max (and their _explicit variants) and the legacy
;; atom_min/max built-in functions into the corresponding SPIR-V
;; OpAtomicSMin/OpAtomicSMax/OpAtomicUMin/OpAtomicUMax instructions,
;; selecting the signed or unsigned variant based on the argument type.
;;
;; atomic_fetch_min/max come from C11/OpenCL 2.0+ atomics and operate
;; on atomic_int / atomic_uint (i32). The legacy atom_min/max builtins
;; come from the cl_khr_*_extended_atomics extensions and exist for
;; both i32 (cl_khr_global_int32_extended_atomics) and i64
;; (cl_khr_int64_extended_atomics). Each path is exercised in its own
;; kernel below.

; CHECK-SPIRV-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#UINT_PTR:]] = OpTypePointer CrossWorkgroup %[[#UINT]]
; CHECK-SPIRV-DAG: %[[#ULONG:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#ULONG_PTR:]] = OpTypePointer CrossWorkgroup %[[#ULONG]]

;; 0x2 Workgroup
; CHECK-SPIRV-DAG: %[[#WORKGROUP_SCOPE:]] = OpConstant %[[#UINT]] 2{{$}}

;;
;; atomic_fetch_min/max + _explicit on i32 (signed): expect OpAtomicSMin / OpAtomicSMax
;;
;; __kernel void test_atomic_fetch_min_max_signed(__global int *p, int val) {
;;   atomic_fetch_min(p, val);
;;   atomic_fetch_max(p, val);
;;   atomic_fetch_min_explicit(p, val, memory_order_relaxed);
;;   atomic_fetch_max_explicit(p, val, memory_order_relaxed);
;; }
; CHECK-SPIRV:     %[[#FETCH_S:]] = OpFunction %[[#]]
; CHECK-SPIRV:     %[[#FS_PTR:]] = OpFunctionParameter %[[#UINT_PTR]]
; CHECK-SPIRV:     %[[#FS_VAL:]] = OpFunctionParameter %[[#UINT]]
; CHECK-SPIRV:     %[[#]] = OpAtomicSMin %[[#UINT]] %[[#FS_PTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#FS_VAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicSMax %[[#UINT]] %[[#FS_PTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#FS_VAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicSMin %[[#UINT]] %[[#FS_PTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#FS_VAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicSMax %[[#UINT]] %[[#FS_PTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#FS_VAL]]

define dso_local spir_kernel void @test_atomic_fetch_min_max_signed(ptr addrspace(1) noundef %p, i32 noundef %val) local_unnamed_addr {
entry:
  %call0 = tail call spir_func i32 @_Z16atomic_fetch_minPU3AS1Vii(ptr addrspace(1) noundef %p, i32 noundef %val)
  %call1 = tail call spir_func i32 @_Z16atomic_fetch_maxPU3AS1Vii(ptr addrspace(1) noundef %p, i32 noundef %val)
  %call2 = tail call spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS1Viii(ptr addrspace(1) noundef %p, i32 noundef %val, i32 noundef 0)
  %call3 = tail call spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS1Viii(ptr addrspace(1) noundef %p, i32 noundef %val, i32 noundef 0)
  ret void
}

;;
;; atomic_fetch_min/max + _explicit on i32 (unsigned): expect OpAtomicUMin / OpAtomicUMax
;;
;; __kernel void test_atomic_fetch_min_max_unsigned(__global unsigned int *p, unsigned int val) {
;;   atomic_fetch_min(p, val);
;;   atomic_fetch_max(p, val);
;;   atomic_fetch_min_explicit(p, val, memory_order_relaxed);
;;   atomic_fetch_max_explicit(p, val, memory_order_relaxed);
;; }
; CHECK-SPIRV:     %[[#FETCH_U:]] = OpFunction %[[#]]
; CHECK-SPIRV:     %[[#FU_PTR:]] = OpFunctionParameter %[[#UINT_PTR]]
; CHECK-SPIRV:     %[[#FU_VAL:]] = OpFunctionParameter %[[#UINT]]
; CHECK-SPIRV:     %[[#]] = OpAtomicUMin %[[#UINT]] %[[#FU_PTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#FU_VAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicUMax %[[#UINT]] %[[#FU_PTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#FU_VAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicUMin %[[#UINT]] %[[#FU_PTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#FU_VAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicUMax %[[#UINT]] %[[#FU_PTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#FU_VAL]]

define dso_local spir_kernel void @test_atomic_fetch_min_max_unsigned(ptr addrspace(1) noundef %p, i32 noundef %val) local_unnamed_addr {
entry:
  %call0 = tail call spir_func i32 @_Z16atomic_fetch_minPU3AS1Vjj(ptr addrspace(1) noundef %p, i32 noundef %val)
  %call1 = tail call spir_func i32 @_Z16atomic_fetch_maxPU3AS1Vjj(ptr addrspace(1) noundef %p, i32 noundef %val)
  %call2 = tail call spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS1Vjji(ptr addrspace(1) noundef %p, i32 noundef %val, i32 noundef 0)
  %call3 = tail call spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS1Vjji(ptr addrspace(1) noundef %p, i32 noundef %val, i32 noundef 0)
  ret void
}

;;
;; Legacy atom_min/max on i32 (cl_khr_global_int32_extended_atomics):
;; expect OpAtomicSMin/OpAtomicSMax for int and OpAtomicUMin/OpAtomicUMax for uint.
;;
;; __kernel void test_atom_min_max_i32(__global int *sp, int sv,
;;                                     __global unsigned int *up, unsigned int uv) {
;;   atom_min(sp, sv);
;;   atom_max(sp, sv);
;;   atom_min(up, uv);
;;   atom_max(up, uv);
;; }
; CHECK-SPIRV:     %[[#ATOM32:]] = OpFunction %[[#]]
; CHECK-SPIRV:     %[[#A32_SPTR:]] = OpFunctionParameter %[[#UINT_PTR]]
; CHECK-SPIRV:     %[[#A32_SVAL:]] = OpFunctionParameter %[[#UINT]]
; CHECK-SPIRV:     %[[#A32_UPTR:]] = OpFunctionParameter %[[#UINT_PTR]]
; CHECK-SPIRV:     %[[#A32_UVAL:]] = OpFunctionParameter %[[#UINT]]
; CHECK-SPIRV:     %[[#]] = OpAtomicSMin %[[#UINT]] %[[#A32_SPTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#A32_SVAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicSMax %[[#UINT]] %[[#A32_SPTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#A32_SVAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicUMin %[[#UINT]] %[[#A32_UPTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#A32_UVAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicUMax %[[#UINT]] %[[#A32_UPTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#A32_UVAL]]

define dso_local spir_kernel void @test_atom_min_max_i32(ptr addrspace(1) noundef %sp, i32 noundef %sv, ptr addrspace(1) noundef %up, i32 noundef %uv) local_unnamed_addr {
entry:
  %call0 = tail call spir_func i32 @_Z8atom_minPU3AS1Vii(ptr addrspace(1) noundef %sp, i32 noundef %sv)
  %call1 = tail call spir_func i32 @_Z8atom_maxPU3AS1Vii(ptr addrspace(1) noundef %sp, i32 noundef %sv)
  %call2 = tail call spir_func i32 @_Z8atom_minPU3AS1Vjj(ptr addrspace(1) noundef %up, i32 noundef %uv)
  %call3 = tail call spir_func i32 @_Z8atom_maxPU3AS1Vjj(ptr addrspace(1) noundef %up, i32 noundef %uv)
  ret void
}

;;
;; Legacy atom_min/max on i64 (cl_khr_int64_extended_atomics):
;; expect OpAtomicSMin/OpAtomicSMax for long and OpAtomicUMin/OpAtomicUMax for ulong.
;;
;; __kernel void test_atom_min_max_i64(__global long *sp, long sv,
;;                                     __global ulong *up, ulong uv) {
;;   atom_min(sp, sv);
;;   atom_max(sp, sv);
;;   atom_min(up, uv);
;;   atom_max(up, uv);
;; }
; CHECK-SPIRV:     %[[#ATOM64:]] = OpFunction %[[#]]
; CHECK-SPIRV:     %[[#A64_SPTR:]] = OpFunctionParameter %[[#ULONG_PTR]]
; CHECK-SPIRV:     %[[#A64_SVAL:]] = OpFunctionParameter %[[#ULONG]]
; CHECK-SPIRV:     %[[#A64_UPTR:]] = OpFunctionParameter %[[#ULONG_PTR]]
; CHECK-SPIRV:     %[[#A64_UVAL:]] = OpFunctionParameter %[[#ULONG]]
; CHECK-SPIRV:     %[[#]] = OpAtomicSMin %[[#ULONG]] %[[#A64_SPTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#A64_SVAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicSMax %[[#ULONG]] %[[#A64_SPTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#A64_SVAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicUMin %[[#ULONG]] %[[#A64_UPTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#A64_UVAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicUMax %[[#ULONG]] %[[#A64_UPTR]] %[[#WORKGROUP_SCOPE]] %[[#]] %[[#A64_UVAL]]

define dso_local spir_kernel void @test_atom_min_max_i64(ptr addrspace(1) noundef %sp, i64 noundef %sv, ptr addrspace(1) noundef %up, i64 noundef %uv) local_unnamed_addr {
entry:
  %call0 = tail call spir_func i64 @_Z8atom_minPU3AS1Vll(ptr addrspace(1) noundef %sp, i64 noundef %sv)
  %call1 = tail call spir_func i64 @_Z8atom_maxPU3AS1Vll(ptr addrspace(1) noundef %sp, i64 noundef %sv)
  %call2 = tail call spir_func i64 @_Z8atom_minPU3AS1Vmm(ptr addrspace(1) noundef %up, i64 noundef %uv)
  %call3 = tail call spir_func i64 @_Z8atom_maxPU3AS1Vmm(ptr addrspace(1) noundef %up, i64 noundef %uv)
  ret void
}

;; atomic_fetch_min/max (i32 signed and unsigned)
declare spir_func i32 @_Z16atomic_fetch_minPU3AS1Vii(ptr addrspace(1) noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z16atomic_fetch_maxPU3AS1Vii(ptr addrspace(1) noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS1Viii(ptr addrspace(1) noundef, i32 noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS1Viii(ptr addrspace(1) noundef, i32 noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z16atomic_fetch_minPU3AS1Vjj(ptr addrspace(1) noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z16atomic_fetch_maxPU3AS1Vjj(ptr addrspace(1) noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS1Vjji(ptr addrspace(1) noundef, i32 noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS1Vjji(ptr addrspace(1) noundef, i32 noundef, i32 noundef) local_unnamed_addr

;; Legacy atom_min/max (i32 signed and unsigned)
declare spir_func i32 @_Z8atom_minPU3AS1Vii(ptr addrspace(1) noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z8atom_maxPU3AS1Vii(ptr addrspace(1) noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z8atom_minPU3AS1Vjj(ptr addrspace(1) noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @_Z8atom_maxPU3AS1Vjj(ptr addrspace(1) noundef, i32 noundef) local_unnamed_addr

;; Legacy atom_min/max (i64 signed and unsigned)
declare spir_func i64 @_Z8atom_minPU3AS1Vll(ptr addrspace(1) noundef, i64 noundef) local_unnamed_addr
declare spir_func i64 @_Z8atom_maxPU3AS1Vll(ptr addrspace(1) noundef, i64 noundef) local_unnamed_addr
declare spir_func i64 @_Z8atom_minPU3AS1Vmm(ptr addrspace(1) noundef, i64 noundef) local_unnamed_addr
declare spir_func i64 @_Z8atom_maxPU3AS1Vmm(ptr addrspace(1) noundef, i64 noundef) local_unnamed_addr

;; References:
;; [1]: https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_C.html#atomic-functions
;; [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpAtomicSMin
;; [3]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpAtomicSMax
