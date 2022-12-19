; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; This test just checks that the compiler doesn't crash.

; FUNC-LABEL: {{^}}v32i8_to_v8i32:
define amdgpu_ps float @v32i8_to_v8i32(ptr addrspace(4) inreg) #0 {
entry:
  %1 = load <32 x i8>, ptr addrspace(4) %0
  %2 = bitcast <32 x i8> %1 to <8 x i32>
  %3 = extractelement <8 x i32> %2, i32 1
  %4 = icmp ne i32 %3, 0
  %5 = select i1 %4, float 0.0, float 1.0
  ret float %5
}

; FUNC-LABEL: {{^}}i8ptr_v16i8ptr:
; SI: s_endpgm
define amdgpu_kernel void @i8ptr_v16i8ptr(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %0 = load <16 x i8>, ptr addrspace(1) %in
  store <16 x i8> %0, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @f32_to_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load float, ptr addrspace(1) %in, align 4
  %fadd32 = fadd float %load, 1.0
  %bc = bitcast float %fadd32 to <2 x i16>
  %add.bitcast = add <2 x i16> %bc, <i16 2, i16 2>
  store <2 x i16> %add.bitcast, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v2i16_to_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x i16>, ptr addrspace(1) %in, align 4
  %add.v2i16 = add <2 x i16> %load, <i16 2, i16 2>
  %bc = bitcast <2 x i16> %add.v2i16 to float
  %fadd.bitcast = fadd float %bc, 1.0
  store float %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @f32_to_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load float, ptr addrspace(1) %in, align 4
  %fadd32 = fadd float %load, 1.0
  %bc = bitcast float %fadd32 to <2 x half>
  %add.bitcast = fadd <2 x half> %bc, <half 2.0, half 2.0>
  store <2 x half> %add.bitcast, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v2f16_to_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x half>, ptr addrspace(1) %in, align 4
  %add.v2f16 = fadd <2 x half> %load, <half 2.0, half 2.0>
  %bc = bitcast <2 x half> %add.v2f16 to float
  %fadd.bitcast = fadd float %bc, 1.0
  store float %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v4i8_to_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x i8>, ptr addrspace(1) %in, align 4
  %bc = bitcast <4 x i8> %load to i32
  store i32 %bc, ptr addrspace(1) %out, align 4
  ret void
}

define amdgpu_kernel void @i32_to_v4i8(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load i32, ptr addrspace(1) %in, align 4
  %bc = bitcast i32 %load to <4 x i8>
  store <4 x i8> %bc, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v2i32_to_f64:
; SI: s_endpgm
define amdgpu_kernel void @bitcast_v2i32_to_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load <2 x i32>, ptr addrspace(1) %in, align 8
  %add = add <2 x i32> %val, <i32 4, i32 9>
  %bc = bitcast <2 x i32> %add to double
  %fadd.bc = fadd double %bc, 1.0
  store double %fadd.bc, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}bitcast_f64_to_v2i32:
; SI: s_endpgm
define amdgpu_kernel void @bitcast_f64_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load double, ptr addrspace(1) %in, align 8
  %add = fadd double %val, 4.0
  %bc = bitcast double %add to <2 x i32>
  store <2 x i32> %bc, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v2i64_to_v2f64:
define amdgpu_kernel void @bitcast_v2i64_to_v2f64(i32 %cond, ptr addrspace(1) %out, <2 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x i64> %value to <2 x double>
  br label %end

end:
  %phi = phi <2 x double> [zeroinitializer, %entry], [%cast, %if]
  store <2 x double> %phi, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v2f64_to_v2i64:
define amdgpu_kernel void @bitcast_v2f64_to_v2i64(i32 %cond, ptr addrspace(1) %out, <2 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x double> %value to <2 x i64>
  br label %end

end:
  %phi = phi <2 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <2 x i64> %phi, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v4i16_to_f64:
define amdgpu_kernel void @v4i16_to_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in, align 4
  %add.v4i16 = add <4 x i16> %load, <i16 4, i16 4, i16 4, i16 4>
  %bc = bitcast <4 x i16> %add.v4i16 to double
  %fadd.bitcast = fadd double %bc, 1.0
  store double %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v4f16_to_f64:
define amdgpu_kernel void @v4f16_to_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x half>, ptr addrspace(1) %in, align 4
  %add.v4half = fadd <4 x half> %load, <half 4.0, half 4.0, half 4.0, half 4.0>
  %bc = bitcast <4 x half> %add.v4half to double
  %fadd.bitcast = fadd double %bc, 1.0
  store double %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f64_to_v4f16:
define amdgpu_kernel void @f64_to_v4f16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load double, ptr addrspace(1) %in, align 4
  %fadd32 = fadd double %load, 1.0
  %bc = bitcast double %fadd32 to <4 x half>
  %add.bitcast = fadd <4 x half> %bc, <half 2.0, half 2.0, half 2.0, half 2.0>
  store <4 x half> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f64_to_v4i16:
define amdgpu_kernel void @f64_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load double, ptr addrspace(1) %in, align 4
  %fadd32 = fadd double %load, 1.0
  %bc = bitcast double %fadd32 to <4 x i16>
  %add.bitcast = add <4 x i16> %bc, <i16 2, i16 2, i16 2, i16 2>
  store <4 x i16> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v4i16_to_i64:
define amdgpu_kernel void @v4i16_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in, align 4
  %add.v4i16 = add <4 x i16> %load, <i16 4, i16 4, i16 4, i16 4>
  %bc = bitcast <4 x i16> %add.v4i16 to i64
  %add.bitcast = add i64 %bc, 1
  store i64 %add.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v4f16_to_i64:
define amdgpu_kernel void @v4f16_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x half>, ptr addrspace(1) %in, align 4
  %add.v4half = fadd <4 x half> %load, <half 4.0, half 4.0, half 4.0, half 4.0>
  %bc = bitcast <4 x half> %add.v4half to i64
  %add.bitcast = add i64 %bc, 1
  store i64 %add.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}bitcast_i64_to_v4i16:
define amdgpu_kernel void @bitcast_i64_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in, align 8
  %add = add i64 %val, 4
  %bc = bitcast i64 %add to <4 x i16>
  %add.v4i16 = add <4 x i16> %bc, <i16 1, i16 2, i16 3, i16 4>
  store <4 x i16> %add.v4i16, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}bitcast_i64_to_v4f16:
define amdgpu_kernel void @bitcast_i64_to_v4f16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in, align 8
  %add = add i64 %val, 4
  %bc = bitcast i64 %add to <4 x half>
  %add.v4i16 = fadd <4 x half> %bc, <half 1.0, half 2.0, half 4.0, half 8.0>
  store <4 x half> %add.v4i16, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v4i16_to_v2f32:
define amdgpu_kernel void @v4i16_to_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in, align 4
  %add.v4i16 = add <4 x i16> %load, <i16 4, i16 4, i16 4, i16 4>
  %bc = bitcast <4 x i16> %add.v4i16 to <2 x float>
  %fadd.bitcast = fadd <2 x float> %bc, <float 1.0, float 1.0>
  store <2 x float> %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v4f16_to_v2f32:
define amdgpu_kernel void @v4f16_to_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x half>, ptr addrspace(1) %in, align 4
  %add.v4half = fadd <4 x half> %load, <half 4.0, half 4.0, half 4.0, half 4.0>
  %bc = bitcast <4 x half> %add.v4half to <2 x float>
  %fadd.bitcast = fadd <2 x float> %bc, <float 1.0, float 1.0>
  store <2 x float> %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v2f32_to_v4i16:
define amdgpu_kernel void @v2f32_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x float>, ptr addrspace(1) %in, align 4
  %add.v2f32 = fadd <2 x float> %load, <float 2.0, float 4.0>
  %bc = bitcast <2 x float> %add.v2f32 to <4 x i16>
  %add.bitcast = add <4 x i16> %bc, <i16 1, i16 2, i16 3, i16 4>
  store <4 x i16> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v2f32_to_v4f16:
define amdgpu_kernel void @v2f32_to_v4f16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x float>, ptr addrspace(1) %in, align 4
  %add.v2f32 = fadd <2 x float> %load, <float 2.0, float 4.0>
  %bc = bitcast <2 x float> %add.v2f32 to <4 x half>
  %add.bitcast = fadd <4 x half> %bc, <half 1.0, half 2.0, half 4.0, half 8.0>
  store <4 x half> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v4i16_to_v2i32:
define amdgpu_kernel void @v4i16_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in, align 4
  %add.v4i16 = add <4 x i16> %load, <i16 4, i16 4, i16 4, i16 4>
  %bc = bitcast <4 x i16> %add.v4i16 to <2 x i32>
  %add.bitcast = add <2 x i32> %bc, <i32 1, i32 1>
  store <2 x i32> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v4f16_to_v2i32:
define amdgpu_kernel void @v4f16_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x half>, ptr addrspace(1) %in, align 4
  %add.v4half = fadd <4 x half> %load, <half 4.0, half 4.0, half 4.0, half 4.0>
  %bc = bitcast <4 x half> %add.v4half to <2 x i32>
  %add.bitcast = add <2 x i32> %bc, <i32 1, i32 1>
  store <2 x i32> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v2i32_to_v4i16:
define amdgpu_kernel void @v2i32_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x i32>, ptr addrspace(1) %in, align 4
  %add.v2i32 = add <2 x i32> %load, <i32 2, i32 4>
  %bc = bitcast <2 x i32> %add.v2i32 to <4 x i16>
  %add.bitcast = add <4 x i16> %bc, <i16 1, i16 2, i16 3, i16 4>
  store <4 x i16> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v2i32_to_v4f16:
define amdgpu_kernel void @v2i32_to_v4f16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x i32>, ptr addrspace(1) %in, align 4
  %add.v2i32 = add <2 x i32> %load, <i32 2, i32 4>
  %bc = bitcast <2 x i32> %add.v2i32 to <4 x half>
  %add.bitcast = fadd <4 x half> %bc, <half 1.0, half 2.0, half 4.0, half 8.0>
  store <4 x half> %add.bitcast, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.s.buffer.load.v4f32(<4 x i32>, i32, i32 immarg)

; FUNC-LABEL: {{^}}bitcast_v4f32_to_v2i64:
; GCN: s_buffer_load_dwordx4
define <2 x i64> @bitcast_v4f32_to_v2i64(<2 x i64> %arg) {
  %val = call <4 x float> @llvm.amdgcn.s.buffer.load.v4f32(<4 x i32> undef, i32 0, i32 0)
  %cast = bitcast <4 x float> %val to <2 x i64>
  %div = udiv <2 x i64> %cast, %arg
  ret <2 x i64> %div
}

declare half @llvm.canonicalize.f16(half)

; FUNC-LABEL: {{^}}bitcast_f32_to_v1i32:
define amdgpu_kernel void @bitcast_f32_to_v1i32(ptr addrspace(1) %out) {
  %f16 = call arcp afn half @llvm.canonicalize.f16(half 0xH03F0)
  %f32 = fpext half %f16 to float
  %v = bitcast float %f32 to <1 x i32>
  %v1 = extractelement <1 x i32> %v, i32 0
  store i32 %v1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v4i64_to_v16i16:
define amdgpu_kernel void @bitcast_v4i64_to_v16i16(i32 %cond, ptr addrspace(1) %out, <4 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <4 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <4 x i64> %phi_value to <16 x i16>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <16 x i16> [zeroinitializer, %entry], [%cast, %if]
  store <16 x i16> %phi_cast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v4f64_to_v16f16:
define amdgpu_kernel void @bitcast_v4f64_to_v16f16(i32 %cond, ptr addrspace(1) %out, <4 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <4 x double> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <4 x double> %phi_value to <16 x half>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <16 x half> [zeroinitializer, %entry], [%cast, %if]
  store <16 x half> %phi_cast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v16i16_to_v4i64:
define amdgpu_kernel void @bitcast_v16i16_to_v4i64(i32 %cond, ptr addrspace(1) %out, <16 x i16> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <16 x i16> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <16 x i16> %phi_value to <4 x i64>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <4 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <4 x i64> %phi_cast, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v16f16_to_v4f64:
define amdgpu_kernel void @bitcast_v16f16_to_v4f64(i32 %cond, ptr addrspace(1) %out, <16 x half> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <16 x half> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <16 x half> %phi_value to <4 x double>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <4 x double> [zeroinitializer, %entry], [%cast, %if]
  store <4 x double> %phi_cast, ptr addrspace(1) %out
  ret void
}
