; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; spirv-val incorrectly flags atomic accesses as not allowed under universal validation for DeviceOnlyINTEL/HostOnlyINTEL
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

target triple = "spirv64-amd-amdhsa"

; CHECK: OpName %[[#LOADS:]] "loads"
; CHECK: OpName %[[#OPAQUE_PTR_CAST:]] "spirv.llvm_spv_opaque_ptr_cast_p7_p8"
; CHECK: OpName %[[#STORES:]] "stores"
; CHECK: OpName %[[#ATOMICRMW:]] "atomicrmw"
; CHECK: OpName %[[#CMPXCHG:]] "cmpxchg"

; CHECK: %[[#LOADS]] = OpFunction
; CHECK: %[[#BUF:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_8:]]
;	CHECK: %[[#CAST_AS8_TO_AS7:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7:]] %[[#OPAQUE_PTR_CAST]] %[[#BUF]]
; CHECK: %[[#BC:]] = OpBitcast %[[#FLOATPTR_ADDRSPACE_7:]] %[[#CAST_AS8_TO_AS7]]
; CHECK: %[[#P:]] = OpPtrAccessChain %[[#FLOATPTR_ADDRSPACE_7:]] %[[#BC]]
; CHECK: %[[#SCALAR:]] = OpLoad %[[#FLOAT_TY:]] %[[#P]] Volatile|Aligned 4
; CHECK: %[[#P_AS_VEC2:]] = OpBitcast %[[#VEC2_FLOATPTR_ADDRSPACE_7:]] %[[#P]]
; CHECK: %[[#VEC2:]] = OpLoad %[[#VEC2_FLOAT_TY:]] %[[#P_AS_VEC2]] Volatile|Aligned 8
; CHECK: %[[#P_AS_VEC4:]] = OpBitcast %[[#VEC4_FLOATPTR_ADDRSPACE_7:]] %[[#P]]
; CHECK: %[[#VEC4:]] = OpLoad %[[#VEC4_FLOAT_TY:]] %[[#P_AS_VEC4]] Volatile|Aligned 16
; CHECK: %[[#NONTEMPORAL:]] = OpLoad %[[#FLOAT_TY]] %[[#P]] Volatile|Aligned|Nontemporal 4
; CHECK: %[[#P_AS_INT32:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7:]] %[[#P]]
; CHECK: %[[#ATOMIC:]] = OpAtomicLoad %[[#INT32_TY:]] %[[#P_AS_INT32]]
; CHECK: %[[#ATOMIC_MONOTONIC:]] = OpAtomicLoad %[[#INT32_TY]] %[[#P_AS_INT32]]
; CHECK: %[[#ATOMIC_ACQUIRE:]] = OpAtomicLoad %[[#INT32_TY]] %[[#P_AS_INT32]]
define void @loads(ptr addrspace(8) %buf) {
  %base = addrspacecast ptr addrspace(8) %buf to ptr addrspace(7)
  %p = getelementptr float, ptr addrspace(7) %base, i32 4

  %scalar = load volatile float, ptr addrspace(7) %p, align 4
  %vec2 = load volatile <2 x float>, ptr addrspace(7) %p, align 8
  %vec4 = load volatile <4 x float>, ptr addrspace(7) %p, align 16

  %nontemporal = load volatile float, ptr addrspace(7) %p, !nontemporal !0

  %atomic = load atomic volatile float, ptr addrspace(7) %p syncscope("subgroup") seq_cst, align 4
  %atomic.monotonic = load atomic float, ptr addrspace(7) %p syncscope("subgroup") monotonic, align 4
  %atomic.acquire = load atomic float, ptr addrspace(7) %p acquire, align 4

  ret void
}

; CHECK: %[[#STORES]] = OpFunction
; CHECK: %[[#BUF1:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_8]]
; CHECK: %[[#F:]] = OpFunctionParameter %[[#FLOAT_TY]]
; CHECK: %[[#F4:]] = OpFunctionParameter %[[#VEC4_FLOAT_TY]]
;	CHECK: %[[#CAST_AS8_TO_AS7_1:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST]] %[[#BUF1]]
; CHECK: %[[#BC1:]] = OpBitcast %[[#FLOATPTR_ADDRSPACE_7]] %[[#CAST_AS8_TO_AS7_1]]
; CHECK: %[[#P1:]] = OpPtrAccessChain %[[#FLOATPTR_ADDRSPACE_7]] %[[#BC1]]
; CHECK: OpStore %[[#P1]] %[[#F]] Aligned 4
;	CHECK: %[[#P1_AS_VEC4:]] = OpBitcast %[[#VEC4_FLOATPTR_ADDRSPACE_7]] %[[#P1]]
; CHECK: OpStore %[[#P1_AS_VEC4]] %[[#F4]] Aligned 16
; CHECK: OpStore %[[#P1]] %[[#F]] Aligned|Nontemporal 4
; CHECK: OpStore %[[#P1]] %[[#F]] Volatile|Aligned 4
; CHECK: OpStore %[[#P1]] %[[#F]] Volatile|Aligned|Nontemporal 4
; CHECK: %[[#F_AS_INT32:]] = OpBitcast %[[#INT32_TY]] %[[#F]]
; CHECK: %[[#P1_AS_INT32:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P1]]
; CHECK: OpAtomicStore %[[#P1_AS_INT32]] %[[#]] %[[#]] %[[#F_AS_INT32]]
; CHECK: %[[#F_AS_INT32_1:]] = OpBitcast %[[#INT32_TY]] %[[#F]]
;	OpAtomicStore %[[#P1_AS_INT32_1]] %[[#]] %[[#]] %[[#F_AS_INT32_1]]
; CHECK: %[[#F_AS_INT32_2:]] = OpBitcast %[[#INT32_TY]] %[[#F]]
; OpAtomicStore %[[#P1_AS_INT32_2]] %[[#]] %[[#]] %[[#F_AS_INT32_2]]
define void @stores(ptr addrspace(8) %buf, float %f, <4 x float> %f4) {
  %base = addrspacecast ptr addrspace(8) %buf to ptr addrspace(7)
  %p = getelementptr float, ptr addrspace(7) %base, i32 4

  store float %f, ptr addrspace(7) %p, align 4
  store <4 x float> %f4, ptr addrspace(7) %p, align 16

  store float %f, ptr addrspace(7) %p, !nontemporal !0

  store volatile float %f, ptr addrspace(7) %p
  store volatile float %f, ptr addrspace(7) %p, !nontemporal !0

  store atomic volatile float %f, ptr addrspace(7) %p syncscope("subgroup") seq_cst, align 4
  store atomic float %f, ptr addrspace(7) %p syncscope("subgroup") monotonic, align 4
  store atomic float %f, ptr addrspace(7) %p release, align 4

  ret void
}

; CHECK: %[[#ATOMICRMW]] = OpFunction
; CHECK: %[[#BUF2:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_8]]
; CHECK: %[[#F1:]] = OpFunctionParameter %[[#FLOAT_TY]]
; CHECK: %[[#I:]] = OpFunctionParameter %[[#INT32_TY]]
;	CHECK: %[[#CAST_AS8_TO_AS7_2:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST]] %[[#BUF2]]
; CHECK: %[[#BC2:]] = OpBitcast %[[#FLOATPTR_ADDRSPACE_7]] %[[#CAST_AS8_TO_AS7_2]]
; CHECK: %[[#P2:]] = OpPtrAccessChain %[[#FLOATPTR_ADDRSPACE_7]] %[[#BC2]]
; CHECK: %[[#P2_AS_INT32:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_1:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_2:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_3:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_4:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_5:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_6:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_7:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_8:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_9:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK: %[[#P2_AS_INT32_10:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P2]]
; CHECK:	%78 = OpAtomicExchange %[[#INT32_TY]] %[[#P2_AS_INT32]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%79 = OpAtomicIAdd %[[#INT32_TY]] %[[#P2_AS_INT32_1]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%80 = OpAtomicISub %[[#INT32_TY]] %[[#P2_AS_INT32_2]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%81 = OpAtomicAnd %[[#INT32_TY]] %[[#P2_AS_INT32_3]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%82 = OpAtomicOr %[[#INT32_TY]] %[[#P2_AS_INT32_4]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%83 = OpAtomicXor %[[#INT32_TY]] %[[#P2_AS_INT32_5]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%84 = OpAtomicSMin %[[#INT32_TY]] %[[#P2_AS_INT32_6]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%85 = OpAtomicSMax %[[#INT32_TY]] %[[#P2_AS_INT32_7]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%86 = OpAtomicUMin %[[#INT32_TY]] %[[#P2_AS_INT32_8]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%87 = OpAtomicUMax %[[#INT32_TY]] %[[#P2_AS_INT32_9]] %[[#]] %[[#]] %[[#I]]
; CHECK:	%88 = OpAtomicFAddEXT %[[#FLOAT_TY]] %[[#P2]] %23 %22 %[[#F1]]
; CHECK:	%89 = OpAtomicFMaxEXT %[[#FLOAT_TY]] %[[#P2]] %23 %22 %[[#F1]]
; CHECK:	%90 = OpAtomicFMinEXT %[[#FLOAT_TY]] %[[#P2]] %23 %22 %[[#F1]]
; CHECK:	%91 = OpAtomicIAdd %[[#INT32_TY]] %[[#P2_AS_INT32_10]] %[[#]] %[[#]] %[[#I]]
define void @atomicrmw(ptr addrspace(8) %buf, float %f, i32 %i) {
  %base = addrspacecast ptr addrspace(8) %buf to ptr addrspace(7)
  %p = getelementptr float, ptr addrspace(7) %base, i32 4

  ; Fence insertion is tested by loads and stores
  %xchg = atomicrmw xchg ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4
  %add = atomicrmw add ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4
  %sub = atomicrmw sub ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4
  %and = atomicrmw and ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4
  %or = atomicrmw or ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4
  %xor = atomicrmw xor ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4
  %min = atomicrmw min ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4
  %max = atomicrmw max ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4
  %umin = atomicrmw umin ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4
  %umax = atomicrmw umax ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4

  %fadd = atomicrmw fadd ptr addrspace(7) %p, float %f syncscope("subgroup") seq_cst, align 4
  %fmax = atomicrmw fmax ptr addrspace(7) %p, float %f syncscope("subgroup") seq_cst, align 4
  %fmin = atomicrmw fmin ptr addrspace(7) %p, float %f syncscope("subgroup") seq_cst, align 4

  ; Check a no-return atomic
  atomicrmw add ptr addrspace(7) %p, i32 %i syncscope("subgroup") seq_cst, align 4

  ret void
}

; CHECK: %[[#CMPXCHG]] = OpFunction
; CHECK: %[[#BUF3:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_8]]
; CHECK: %[[#WANTED:]] = OpFunctionParameter %[[#INT32_TY]]
; CHECK: %[[#NEW:]] = OpFunctionParameter %[[#INT32_TY]]
;	CHECK: %[[#CAST_AS8_TO_AS7_3:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST]] %[[#BUF3]]
; CHECK: %[[#BC3:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#CAST_AS8_TO_AS7_3]]
; CHECK: %[[#P3:]] = OpPtrAccessChain %[[#I32PTR_ADDRSPACE_7]] %[[#BC3]]
; CHECK: %[[#P3_AS_STRUCT:]] = OpBitcast %[[#STRUCTPTR_ADDRSPACE_7:]] %[[#P3]]
; CHECK: %[[#P3_AS_INT32:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#P3_AS_STRUCT]]
;	CHECK: %[[#CMPX:]] = OpAtomicCompareExchange %[[#INT32_TY]] %[[#P3_AS_INT32]] %[[#]] %[[#]] %[[#]] %[[#NEW]] %[[#WANTED]]
; CHECK: %[[#SUCCESS:]] = OpIEqual %[[#BOOL_TY:]] %[[#CMPX]] %[[#WANTED]]
define {i32, i1} @cmpxchg(ptr addrspace(8) %buf, i32 %wanted, i32 %new) {
  %base = addrspacecast ptr addrspace(8) %buf to ptr addrspace(7)
  %p = getelementptr i32, ptr addrspace(7) %base, i32 4

  %ret = cmpxchg volatile ptr addrspace(7) %p, i32 %wanted, i32 %new syncscope("subgroup") acq_rel monotonic, align 4
  ret {i32, i1} %ret
}

!0 = ! { i32 1 }
!1 = ! { }
