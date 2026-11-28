; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3 %s -o - -filetype=obj | spirv-val %}

; Optimizations such as VectorCombine may widen the load of a scalar into the
; load of a vector when only the first element is used. SPIR-V cannot
; access a scalar as a vector, so the LegalizePtrCast legalization loads the
; scalar and inserts it into the first element of a vector whose other elements
; are undef

; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#v4f:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#v2f:]] = OpTypeVector %[[#float]] 2
; CHECK-DAG: %[[#v4i:]] = OpTypeVector %[[#int]] 4
; CHECK-DAG: %[[#char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#v4c:]] = OpTypeVector %[[#char]] 4
; CHECK-DAG: %[[#float_pp:]] = OpTypePointer Private %[[#float]]
; CHECK-DAG: %[[#int_pp:]] = OpTypePointer Private %[[#int]]
; CHECK-DAG: %[[#char_pp:]] = OpTypePointer Private %[[#char]]
; CHECK-DAG: %[[#undef_v4f:]] = OpUndef %[[#v4f]]
; CHECK-DAG: %[[#undef_v2f:]] = OpUndef %[[#v2f]]
; CHECK-DAG: %[[#undef_v4i:]] = OpUndef %[[#v4i]]
; CHECK-DAG: %[[#undef_v4c:]] = OpUndef %[[#v4c]]

%S = type { <4 x float>, float, float, i32 }
%F = type { float, float }
%C = type { i32, i8, i8, i8, i8, i32 }

; A float struct member at index 1 loaded as a float4.
define internal spir_func <4 x float> @struct_member_v4f(ptr addrspace(10) %a) {
  %1 = getelementptr inbounds %S, ptr addrspace(10) %a, i32 0, i32 1
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#float_pp]] %[[#]] %[[#]]
  %2 = load <4 x float>, ptr addrspace(10) %1, align 4
; CHECK: %[[#val:]] = OpLoad %[[#float]] %[[#ptr]]
; CHECK: %[[#res:]] = OpCompositeInsert %[[#v4f]] %[[#val]] %[[#undef_v4f]] 0
  ret <4 x float> %2
; CHECK: OpReturnValue %[[#res]]
}

; A float struct member at index 2 loaded as a float2.
define internal spir_func <2 x float> @struct_member_v2f(ptr addrspace(10) %a) {
  %1 = getelementptr inbounds %S, ptr addrspace(10) %a, i32 0, i32 2
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#float_pp]] %[[#]] %[[#]]
  %2 = load <2 x float>, ptr addrspace(10) %1, align 4
; CHECK: %[[#val:]] = OpLoad %[[#float]] %[[#ptr]]
; CHECK: %[[#res:]] = OpCompositeInsert %[[#v2f]] %[[#val]] %[[#undef_v2f]] 0
  ret <2 x float> %2
; CHECK: OpReturnValue %[[#res]]
}

; An integer struct member at index 3 loaded as an int4.
define internal spir_func <4 x i32> @struct_member_v4i(ptr addrspace(10) %a) {
  %1 = getelementptr inbounds %S, ptr addrspace(10) %a, i32 0, i32 3
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#int_pp]] %[[#]] %[[#]]
  %2 = load <4 x i32>, ptr addrspace(10) %1, align 4
; CHECK: %[[#val:]] = OpLoad %[[#int]] %[[#ptr]]
; CHECK: %[[#res:]] = OpCompositeInsert %[[#v4i]] %[[#val]] %[[#undef_v4i]] 0
  ret <4 x i32> %2
; CHECK: OpReturnValue %[[#res]]
}

; An array element loaded as a float4.
define internal spir_func <4 x float> @array_element_v4f(ptr addrspace(10) %a) {
  %1 = getelementptr inbounds [8 x float], ptr addrspace(10) %a, i32 0, i32 5
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#float_pp]] %[[#]] %[[#]]
  %2 = load <4 x float>, ptr addrspace(10) %1, align 4
; CHECK: %[[#val:]] = OpLoad %[[#float]] %[[#ptr]]
; CHECK: %[[#res:]] = OpCompositeInsert %[[#v4f]] %[[#val]] %[[#undef_v4f]] 0
  ret <4 x float> %2
; CHECK: OpReturnValue %[[#res]]
}

; An i8 struct member at index 2 loaded as a <4 x i8>.
define internal spir_func <4 x i8> @struct_member_v4c(ptr addrspace(10) %a) {
  %1 = getelementptr inbounds %C, ptr addrspace(10) %a, i32 0, i32 2
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#char_pp]] %[[#]] %[[#]]
  %2 = load <4 x i8>, ptr addrspace(10) %1, align 1
; CHECK: %[[#val:]] = OpLoad %[[#char]] %[[#ptr]]
; CHECK: %[[#res:]] = OpCompositeInsert %[[#v4c]] %[[#val]] %[[#undef_v4c]] 0
  ret <4 x i8> %2
; CHECK: OpReturnValue %[[#res]]
}

; The pointer is to a struct. The first scalar member is found by
; drilling down through the struct before emulating the widening.
define internal spir_func <4 x float> @struct_first_member_v4f(ptr addrspace(10) %a) {
  %1 = getelementptr inbounds %F, ptr addrspace(10) %a, i32 0
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#]] %[[#]]{{$}}
  %2 = load <4 x float>, ptr addrspace(10) %1, align 4
; CHECK: %[[#gep:]] = OpAccessChain %[[#float_pp]] %[[#ptr]] %[[#]]
; CHECK: %[[#val:]] = OpLoad %[[#float]] %[[#gep]]
; CHECK: %[[#res:]] = OpCompositeInsert %[[#v4f]] %[[#val]] %[[#undef_v4f]] 0
  ret <4 x float> %2
; CHECK: OpReturnValue %[[#res]]
}

; <1 x float> shares its SPIR-V type with float: no vector is built.
define internal spir_func <1 x float> @struct_member_v1f(ptr addrspace(10) %a) {
  %1 = getelementptr inbounds %S, ptr addrspace(10) %a, i32 0, i32 1
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#float_pp]] %[[#]] %[[#]]
  %2 = load <1 x float>, ptr addrspace(10) %1, align 4
; CHECK: %[[#val:]] = OpLoad %[[#float]] %[[#ptr]]
  ret <1 x float> %2
; CHECK: OpReturnValue %[[#val]]
}
