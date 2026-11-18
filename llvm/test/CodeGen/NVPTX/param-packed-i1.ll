; NOTE: Do not autogenerate. The NVPTX check generator skips parameter
; declarations. We need to check their sizes as well as the memory accesses.
; Related: https://github.com/modular/modular/issues/7480
; Check the packed memory layout of Boolean vector arguments.
; Reject extra offset loads in the lane extraction and full-byte tests.
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_50 -O0 -verify-machineinstrs | FileCheck %s \
; RUN:   --implicit-check-not='ld.param{{.*}}[{{(lane[0-9]+_v[0-9]+i1|all_lanes_v8i1)}}_param_0+'
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_50 -O3 -verify-machineinstrs | FileCheck %s \
; RUN:   --implicit-check-not='ld.param{{.*}}[{{(lane[0-9]+_v[0-9]+i1|all_lanes_v8i1)}}_param_0+'
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_50 -O0 -verify-machineinstrs | FileCheck %s \
; RUN:   --implicit-check-not='ld.param{{.*}}[{{(lane[0-9]+_v[0-9]+i1|all_lanes_v8i1)}}_param_0+'
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_50 -O3 -verify-machineinstrs | FileCheck %s \
; RUN:   --implicit-check-not='ld.param{{.*}}[{{(lane[0-9]+_v[0-9]+i1|all_lanes_v8i1)}}_param_0+'

; Read lane 1 of a two-lane Boolean vector stored in one byte.
; CHECK-LABEL: .entry lane1_v2i1(
; CHECK: .b8 lane1_v2i1_param_0[1]
; CHECK: ld.param.b8 [[V2:%[a-z0-9]+]], [lane1_v2i1_param_0];
; CHECK: shr.u16 [[S2:%rs[0-9]+]], [[V2]], 1;
; CHECK: cvt.u32.u16 [[R2:%r[0-9]+]], [[S2]];
; CHECK: and.b32 [[B2:%r[0-9]+]], [[R2]], 1;
; CHECK: st.global.b32 [{{%r[d]?[0-9]+}}], [[B2]];
; CHECK: ret;
define ptx_kernel void @lane1_v2i1(<2 x i1> %value, ptr addrspace(1) %output) {
  %lane = extractelement <2 x i1> %value, i32 1
  %result = zext i1 %lane to i32
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Read lane 7 of an eight-lane Boolean vector stored in one byte.
; CHECK-LABEL: .entry lane7_v8i1(
; CHECK: .b8 lane7_v8i1_param_0[1]
; CHECK: ld.param.b8 [[V8:%[a-z0-9]+]], [lane7_v8i1_param_0];
; CHECK: shr.u16 [[S8:%rs[0-9]+]], [[V8]], 7;
; CHECK: cvt.u32.u16 [[R8:%r[0-9]+]], [[S8]];
; CHECK: st.global.b32 [{{%r[d]?[0-9]+}}], [[R8]];
; CHECK: ret;
define ptx_kernel void @lane7_v8i1(<8 x i1> %value, ptr addrspace(1) %output) {
  %lane = extractelement <8 x i1> %value, i32 7
  %result = zext i1 %lane to i32
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Read lane 8 of a nine-lane Boolean vector, crossing into the second byte.
; CHECK-LABEL: .entry lane8_v9i1(
; CHECK: .b8 lane8_v9i1_param_0[2]
; CHECK: ld.param.b16 [[V9:%rs[0-9]+]], [lane8_v9i1_param_0];
; CHECK: shr.u16 [[S9:%rs[0-9]+]], [[V9]], 8;
; CHECK: cvt.u32.u16 [[R9:%r[0-9]+]], [[S9]];
; CHECK: st.global.b32 [{{%r[d]?[0-9]+}}], [[R9]];
; CHECK: ret;
define ptx_kernel void @lane8_v9i1(<9 x i1> %value, ptr addrspace(1) %output) {
  %lane = extractelement <9 x i1> %value, i32 8
  %result = zext i1 %lane to i32
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Read lane 15 of a 16-lane Boolean vector stored in two bytes.
; CHECK-LABEL: .entry lane15_v16i1(
; CHECK: .b8 lane15_v16i1_param_0[2]
; CHECK: ld.param.b16 [[V16:%rs[0-9]+]], [lane15_v16i1_param_0];
; CHECK: shr.u16 [[S16:%rs[0-9]+]], [[V16]], 15;
; CHECK: cvt.u32.u16 [[R16:%r[0-9]+]], [[S16]];
; CHECK: st.global.b32 [{{%r[d]?[0-9]+}}], [[R16]];
; CHECK: ret;
define ptx_kernel void @lane15_v16i1(<16 x i1> %value, ptr addrspace(1) %output) {
  %lane = extractelement <16 x i1> %value, i32 15
  %result = zext i1 %lane to i32
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Read lane 31 of a 32-lane Boolean vector stored in four bytes.
; CHECK-LABEL: .entry lane31_v32i1(
; CHECK: .b8 lane31_v32i1_param_0[4]
; CHECK: ld.param.b32 [[V32:%r[0-9]+]], [lane31_v32i1_param_0];
; CHECK: shr.u32 [[S32:%r[0-9]+]], [[V32]], 31;
; CHECK: st.global.b32 [{{%r[d]?[0-9]+}}], [[S32]];
; CHECK: ret;
define ptx_kernel void @lane31_v32i1(<32 x i1> %value, ptr addrspace(1) %output) {
  %lane = extractelement <32 x i1> %value, i32 31
  %result = zext i1 %lane to i32
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Read lane 63 of a 64-lane Boolean vector stored in eight bytes.
; CHECK-LABEL: .entry lane63_v64i1(
; CHECK: .b8 lane63_v64i1_param_0[8]
; CHECK: ld.param.b64 [[V64:%rd[0-9]+]], [lane63_v64i1_param_0];
; CHECK: shr.u64 [[S64:%rd[0-9]+]], [[V64]], 63;
; CHECK: st.global.b32 [{{%r[d]?[0-9]+}}], [[S64]];
; CHECK: ret;
define ptx_kernel void @lane63_v64i1(<64 x i1> %value, ptr addrspace(1) %output) {
  %lane = extractelement <64 x i1> %value, i32 63
  %result = zext i1 %lane to i32
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Read all eight Boolean lanes and store their bits as one byte.
; CHECK-LABEL: .entry all_lanes_v8i1(
; CHECK: .b8 all_lanes_v8i1_param_0[1]
; CHECK: ld.param.b8 [[BYTE:%rs[0-9]+]], [all_lanes_v8i1_param_0];
; CHECK: st.global.b8 [{{%r[d]?[0-9]+}}], [[BYTE]];
; CHECK: ret;
define ptx_kernel void @all_lanes_v8i1(<8 x i1> %value, ptr addrspace(1) %output) {
  %byte = bitcast <8 x i1> %value to i8
  store i8 %byte, ptr addrspace(1) %output, align 1
  ret void
}

; Load scalar and vector fields from their separate positions in a struct.
; CHECK-LABEL: .entry mixed_scalar_vector(
; CHECK: .b8 mixed_scalar_vector_param_0[8]
; CHECK-DAG: ld.param.b8 {{%[a-z0-9]+}}, [mixed_scalar_vector_param_0];
; CHECK-DAG: ld.param.b8 {{%[a-z0-9]+}}, [mixed_scalar_vector_param_0+1];
; CHECK-DAG: ld.param.b32 {{%[a-z0-9]+}}, [mixed_scalar_vector_param_0+4];
; CHECK: ret;
define ptx_kernel void @mixed_scalar_vector({ i1, <8 x i1>, i32 } %value, ptr addrspace(1) %output) {
  %scalar = extractvalue { i1, <8 x i1>, i32 } %value, 0
  %vector = extractvalue { i1, <8 x i1>, i32 } %value, 1
  %tail = extractvalue { i1, <8 x i1>, i32 } %value, 2
  %byte = bitcast <8 x i1> %vector to i8
  %v = zext i8 %byte to i32
  %s = select i1 %scalar, i32 1000, i32 0
  %sum = add i32 %v, %s
  %result = add i32 %sum, %tail
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Pass and return nine Boolean lanes using two bytes.
; FIXME: Keep return loads and stores within their two-byte storage.
; CHECK-LABEL: roundtrip_v9i1(
; CHECK: .b8 roundtrip_v9i1_param_0[2]
; CHECK: ld.param.b16 [[ARG:%rs[0-9]+]], [roundtrip_v9i1_param_0];
; CHECK: .b8 param0[2];
; CHECK: .b8 retval0[2];
; CHECK: st.param.b16 [param0], [[ARG]];
; CHECK: call.uni (retval0), callee_v9i1,
; CHECK: ld.param.b8 {{.*}}, [retval0+8];
; CHECK: st.param.b8 [func_retval0+8], {{.*}};
; CHECK: ret;
define <9 x i1> @roundtrip_v9i1(<9 x i1> %value) {
  %result = call <9 x i1> @callee_v9i1(<9 x i1> %value)
  ret <9 x i1> %result
}

declare <9 x i1> @callee_v9i1(<9 x i1>)

; A one-lane Boolean vector already uses the correct one-byte access.
; Keep this behavior unchanged.
; CHECK-LABEL: .entry single_lane_vector(
; CHECK: .b8 single_lane_vector_param_0[1]
; CHECK: ld.param.b8 {{.*}}, [single_lane_vector_param_0];
; CHECK: st.global.b32
; CHECK: ret;
define ptx_kernel void @single_lane_vector(<1 x i1> %value, ptr addrspace(1) %output) {
  %lane = extractelement <1 x i1> %value, i32 0
  %result = zext i1 %lane to i32
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Load two Boolean vectors from a nested array, respecting field padding.
; CHECK-LABEL: .entry nested_vector_array(
; CHECK: .b8 nested_vector_array_param_0[12]
; CHECK-DAG: ld.param.b8 {{%[a-z0-9]+}}, [nested_vector_array_param_0];
; CHECK-DAG: ld.param.b16 {{%[a-z0-9]+}}, [nested_vector_array_param_0+2];
; CHECK-DAG: ld.param.b16 {{%[a-z0-9]+}}, [nested_vector_array_param_0+4];
; CHECK-DAG: ld.param.b32 {{%[a-z0-9]+}}, [nested_vector_array_param_0+8];
; CHECK: st.global.b32
; CHECK: ret;
define ptx_kernel void @nested_vector_array({ i8, [2 x <16 x i1>], i32 } %value, ptr addrspace(1) %output) {
  %head = extractvalue { i8, [2 x <16 x i1>], i32 } %value, 0
  %first = extractvalue { i8, [2 x <16 x i1>], i32 } %value, 1, 0
  %second = extractvalue { i8, [2 x <16 x i1>], i32 } %value, 1, 1
  %tail = extractvalue { i8, [2 x <16 x i1>], i32 } %value, 2
  %first_bits = bitcast <16 x i1> %first to i16
  %second_bits = bitcast <16 x i1> %second to i16
  %a = zext i16 %first_bits to i32
  %b = zext i16 %second_bits to i32
  %c = zext i8 %head to i32
  %sum0 = add i32 %a, %b
  %sum1 = add i32 %sum0, %c
  %result = add i32 %sum1, %tail
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Use a returned Boolean vector as an integer without crashing the compiler.
; FIXME: Keep return loads within their two-byte storage.
; CHECK-LABEL: .entry returned_bits_v16i1(
; CHECK: .b8 returned_bits_v16i1_param_0[2]
; CHECK: .b8 param0[2];
; CHECK: .b8 retval0[2];
; CHECK-DAG: st.param.b8 [param0], {{%rs[0-9]+}};
; CHECK-DAG: st.param.b8 [param0+1], {{%rs[0-9]+}};
; CHECK: call.uni (retval0), callee_v16i1,
; CHECK: ld.param.v2.b8 {{.*}}, [retval0+14];
; CHECK: st.global.b64
; CHECK: st.global.b64
; CHECK: ret;
define ptx_kernel void @returned_bits_v16i1(<16 x i1> %value, ptr addrspace(1) %output) {
  %original = bitcast <16 x i1> %value to i16
  %result = call <16 x i1> @callee_v16i1(<16 x i1> %value)
  %bits = bitcast <16 x i1> %result to i16
  %first = zext i16 %original to i64
  %second = zext i16 %bits to i64
  store i64 %first, ptr addrspace(1) %output, align 8
  %next = getelementptr i64, ptr addrspace(1) %output, i64 1
  store i64 %second, ptr addrspace(1) %next, align 8
  ret void
}

declare <16 x i1> @callee_v16i1(<16 x i1>)
