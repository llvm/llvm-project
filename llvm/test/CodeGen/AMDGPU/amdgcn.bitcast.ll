; RUN: llc -march=amdgcn -amdgpu-codegenprepare-break-large-phis-threshold=4096 < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -amdgpu-codegenprepare-break-large-phis-threshold=4096 < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-codegenprepare-break-large-phis-threshold=4096 < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -amdgpu-codegenprepare-break-large-phis-threshold=4096 < %s | FileCheck %s

; This test just checks that the compiler doesn't crash.

; CHECK-LABEL: {{^}}v32i8_to_v8i32:
define amdgpu_ps float @v32i8_to_v8i32(ptr addrspace(4) inreg) #0 {
entry:
  %1 = load <32 x i8>, ptr addrspace(4) %0
  %2 = bitcast <32 x i8> %1 to <8 x i32>
  %3 = extractelement <8 x i32> %2, i32 1
  %4 = icmp ne i32 %3, 0
  %5 = select i1 %4, float 0.0, float 1.0
  ret float %5
}

; CHECK-LABEL: {{^}}i8ptr_v16i8ptr:
; CHECK: s_endpgm
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

; CHECK-LABEL: {{^}}bitcast_v2i32_to_f64:
; CHECK: s_endpgm
define amdgpu_kernel void @bitcast_v2i32_to_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load <2 x i32>, ptr addrspace(1) %in, align 8
  %add = add <2 x i32> %val, <i32 4, i32 9>
  %bc = bitcast <2 x i32> %add to double
  %fadd.bc = fadd double %bc, 1.0
  store double %fadd.bc, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK-LABEL: {{^}}bitcast_f64_to_v2i32:
; CHECK: s_endpgm
define amdgpu_kernel void @bitcast_f64_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load double, ptr addrspace(1) %in, align 8
  %add = fadd double %val, 4.0
  %bc = bitcast double %add to <2 x i32>
  store <2 x i32> %bc, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v2i64_to_v2f64:
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

; CHECK-LABEL: {{^}}bitcast_v2f64_to_v2i64:
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

; CHECK-LABEL: {{^}}v4i16_to_f64:
define amdgpu_kernel void @v4i16_to_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in, align 4
  %add.v4i16 = add <4 x i16> %load, <i16 4, i16 4, i16 4, i16 4>
  %bc = bitcast <4 x i16> %add.v4i16 to double
  %fadd.bitcast = fadd double %bc, 1.0
  store double %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v4f16_to_f64:
define amdgpu_kernel void @v4f16_to_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x half>, ptr addrspace(1) %in, align 4
  %add.v4half = fadd <4 x half> %load, <half 4.0, half 4.0, half 4.0, half 4.0>
  %bc = bitcast <4 x half> %add.v4half to double
  %fadd.bitcast = fadd double %bc, 1.0
  store double %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}f64_to_v4f16:
define amdgpu_kernel void @f64_to_v4f16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load double, ptr addrspace(1) %in, align 4
  %fadd32 = fadd double %load, 1.0
  %bc = bitcast double %fadd32 to <4 x half>
  %add.bitcast = fadd <4 x half> %bc, <half 2.0, half 2.0, half 2.0, half 2.0>
  store <4 x half> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}f64_to_v4i16:
define amdgpu_kernel void @f64_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load double, ptr addrspace(1) %in, align 4
  %fadd32 = fadd double %load, 1.0
  %bc = bitcast double %fadd32 to <4 x i16>
  %add.bitcast = add <4 x i16> %bc, <i16 2, i16 2, i16 2, i16 2>
  store <4 x i16> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v4i16_to_i64:
define amdgpu_kernel void @v4i16_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in, align 4
  %add.v4i16 = add <4 x i16> %load, <i16 4, i16 4, i16 4, i16 4>
  %bc = bitcast <4 x i16> %add.v4i16 to i64
  %add.bitcast = add i64 %bc, 1
  store i64 %add.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v4f16_to_i64:
define amdgpu_kernel void @v4f16_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x half>, ptr addrspace(1) %in, align 4
  %add.v4half = fadd <4 x half> %load, <half 4.0, half 4.0, half 4.0, half 4.0>
  %bc = bitcast <4 x half> %add.v4half to i64
  %add.bitcast = add i64 %bc, 1
  store i64 %add.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_i64_to_v4i16:
define amdgpu_kernel void @bitcast_i64_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in, align 8
  %add = add i64 %val, 4
  %bc = bitcast i64 %add to <4 x i16>
  %add.v4i16 = add <4 x i16> %bc, <i16 1, i16 2, i16 3, i16 4>
  store <4 x i16> %add.v4i16, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK-LABEL: {{^}}bitcast_i64_to_v4f16:
define amdgpu_kernel void @bitcast_i64_to_v4f16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in, align 8
  %add = add i64 %val, 4
  %bc = bitcast i64 %add to <4 x half>
  %add.v4i16 = fadd <4 x half> %bc, <half 1.0, half 2.0, half 4.0, half 8.0>
  store <4 x half> %add.v4i16, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK-LABEL: {{^}}v4i16_to_v2f32:
define amdgpu_kernel void @v4i16_to_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in, align 4
  %add.v4i16 = add <4 x i16> %load, <i16 4, i16 4, i16 4, i16 4>
  %bc = bitcast <4 x i16> %add.v4i16 to <2 x float>
  %fadd.bitcast = fadd <2 x float> %bc, <float 1.0, float 1.0>
  store <2 x float> %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v4f16_to_v2f32:
define amdgpu_kernel void @v4f16_to_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x half>, ptr addrspace(1) %in, align 4
  %add.v4half = fadd <4 x half> %load, <half 4.0, half 4.0, half 4.0, half 4.0>
  %bc = bitcast <4 x half> %add.v4half to <2 x float>
  %fadd.bitcast = fadd <2 x float> %bc, <float 1.0, float 1.0>
  store <2 x float> %fadd.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v2f32_to_v4i16:
define amdgpu_kernel void @v2f32_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x float>, ptr addrspace(1) %in, align 4
  %add.v2f32 = fadd <2 x float> %load, <float 2.0, float 4.0>
  %bc = bitcast <2 x float> %add.v2f32 to <4 x i16>
  %add.bitcast = add <4 x i16> %bc, <i16 1, i16 2, i16 3, i16 4>
  store <4 x i16> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v2f32_to_v4f16:
define amdgpu_kernel void @v2f32_to_v4f16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x float>, ptr addrspace(1) %in, align 4
  %add.v2f32 = fadd <2 x float> %load, <float 2.0, float 4.0>
  %bc = bitcast <2 x float> %add.v2f32 to <4 x half>
  %add.bitcast = fadd <4 x half> %bc, <half 1.0, half 2.0, half 4.0, half 8.0>
  store <4 x half> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v4i16_to_v2i32:
define amdgpu_kernel void @v4i16_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in, align 4
  %add.v4i16 = add <4 x i16> %load, <i16 4, i16 4, i16 4, i16 4>
  %bc = bitcast <4 x i16> %add.v4i16 to <2 x i32>
  %add.bitcast = add <2 x i32> %bc, <i32 1, i32 1>
  store <2 x i32> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v4f16_to_v2i32:
define amdgpu_kernel void @v4f16_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <4 x half>, ptr addrspace(1) %in, align 4
  %add.v4half = fadd <4 x half> %load, <half 4.0, half 4.0, half 4.0, half 4.0>
  %bc = bitcast <4 x half> %add.v4half to <2 x i32>
  %add.bitcast = add <2 x i32> %bc, <i32 1, i32 1>
  store <2 x i32> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v2i32_to_v4i16:
define amdgpu_kernel void @v2i32_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x i32>, ptr addrspace(1) %in, align 4
  %add.v2i32 = add <2 x i32> %load, <i32 2, i32 4>
  %bc = bitcast <2 x i32> %add.v2i32 to <4 x i16>
  %add.bitcast = add <4 x i16> %bc, <i16 1, i16 2, i16 3, i16 4>
  store <4 x i16> %add.bitcast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v2i32_to_v4f16:
define amdgpu_kernel void @v2i32_to_v4f16(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %load = load <2 x i32>, ptr addrspace(1) %in, align 4
  %add.v2i32 = add <2 x i32> %load, <i32 2, i32 4>
  %bc = bitcast <2 x i32> %add.v2i32 to <4 x half>
  %add.bitcast = fadd <4 x half> %bc, <half 1.0, half 2.0, half 4.0, half 8.0>
  store <4 x half> %add.bitcast, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.s.buffer.load.v4f32(<4 x i32>, i32, i32 immarg)

; CHECK-LABEL: {{^}}bitcast_v4f32_to_v2i64:
; CHECK: s_buffer_load_{{dwordx4|b128}}
define <2 x i64> @bitcast_v4f32_to_v2i64(<2 x i64> %arg) {
  %val = call <4 x float> @llvm.amdgcn.s.buffer.load.v4f32(<4 x i32> undef, i32 0, i32 0)
  %cast = bitcast <4 x float> %val to <2 x i64>
  %div = udiv <2 x i64> %cast, %arg
  ret <2 x i64> %div
}

declare half @llvm.canonicalize.f16(half)

; CHECK-LABEL: {{^}}bitcast_f32_to_v1i32:
define amdgpu_kernel void @bitcast_f32_to_v1i32(ptr addrspace(1) %out) {
  %f16 = call arcp afn half @llvm.canonicalize.f16(half 0xH03F0)
  %f32 = fpext half %f16 to float
  %v = bitcast float %f32 to <1 x i32>
  %v1 = extractelement <1 x i32> %v, i32 0
  store i32 %v1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v4i64_to_v16i16:
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

; CHECK-LABEL: {{^}}bitcast_v4f64_to_v16f16:
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

; CHECK-LABEL: {{^}}bitcast_v16i16_to_v4i64:
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

; CHECK-LABEL: {{^}}bitcast_v16f16_to_v4f64:
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

; CHECK-LABEL: {{^}}bitcast_v20f16_to_v5f64:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v20f16_to_v5f64(i32 %cond, ptr addrspace(1) %out, <20 x half> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <20 x half> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <20 x half> %phi_value to <5 x double>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <5 x double> [zeroinitializer, %entry], [%cast, %if]
  store <5 x double> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v10f32_to_v5f64:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v10f32_to_v5f64(i32 %cond, ptr addrspace(1) %out, <10 x float> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <10 x float> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <10 x float> %phi_value to <5 x double>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <5 x double> [zeroinitializer, %entry], [%cast, %if]
  store <5 x double> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v10i32_to_v5f64:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v10i32_to_v5f64(i32 %cond, ptr addrspace(1) %out, <10 x i32> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <10 x i32> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <10 x i32> %phi_value to <5 x double>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <5 x double> [zeroinitializer, %entry], [%cast, %if]
  store <5 x double> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v10f32_to_v5i64:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v10f32_to_v5i64(i32 %cond, ptr addrspace(1) %out, <10 x float> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <10 x float> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <10 x float> %phi_value to <5 x i64>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <5 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <5 x i64> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v10i32_to_v5i64:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v10i32_to_v5i64(i32 %cond, ptr addrspace(1) %out, <10 x i32> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <10 x i32> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <10 x i32> %phi_value to <5 x i64>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <5 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <5 x i64> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v40i8_to_v5f64:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v40i8_to_v5f64(i32 %cond, ptr addrspace(1) %out, <40 x i8> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <40 x i8> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <40 x i8> %phi_value to <5 x double>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <5 x double> [zeroinitializer, %entry], [%cast, %if]
  store <5 x double> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v40i8_to_v5i64:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v40i8_to_v5i64(i32 %cond, ptr addrspace(1) %out, <40 x i8> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <40 x i8> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <40 x i8> %phi_value to <5 x i64>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <5 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <5 x i64> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v5f64_to_v10f32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v5f64_to_v10f32(i32 %cond, ptr addrspace(1) %out, <5 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <5 x double> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <5 x double> %phi_value to <10 x float>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <10 x float> [zeroinitializer, %entry], [%cast, %if]
  store <10 x float> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v5f64_to_v10i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v5f64_to_v10i32(i32 %cond, ptr addrspace(1) %out, <5 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <5 x double> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <5 x double> %phi_value to <10 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <10 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <10 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v5i64_to_v10f32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v5i64_to_v10f32(i32 %cond, ptr addrspace(1) %out, <5 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <5 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <5 x i64> %phi_value to <10 x float>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <10 x float> [zeroinitializer, %entry], [%cast, %if]
  store <10 x float> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v5i64_to_v10i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v5i64_to_v10i32(i32 %cond, ptr addrspace(1) %out, <5 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <5 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <5 x i64> %phi_value to <10 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <10 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <10 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v6f64_to_v12i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v6f64_to_v12i32(i32 %cond, ptr addrspace(1) %out, <6 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <6 x double> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <6 x double> %phi_value to <12 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <12 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <12 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v6f64_to_v12f32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v6f64_to_v12f32(i32 %cond, ptr addrspace(1) %out, <6 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <6 x double> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <6 x double> %phi_value to <12 x float>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <12 x float> [zeroinitializer, %entry], [%cast, %if]
  store <12 x float> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v12i32_to_v6i64:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v12i32_to_v6i64(i32 %cond, ptr addrspace(1) %out, <12 x i32> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <12 x i32> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <12 x i32> %phi_value to <6 x i64>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <6 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <6 x i64> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v12i32_to_v6f64:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v12i32_to_v6f64(i32 %cond, ptr addrspace(1) %out, <12 x i32> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <12 x i32> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <12 x i32> %phi_value to <6 x double>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <6 x double> [zeroinitializer, %entry], [%cast, %if]
  store <6 x double> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v6i64_to_v12i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v6i64_to_v12i32(i32 %cond, ptr addrspace(1) %out, <6 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <6 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <6 x i64> %phi_value to <12 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <12 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <12 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v7i64_to_v14i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v7i64_to_v14i32(i32 %cond, ptr addrspace(1) %out, <7 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <7 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <7 x i64> %phi_value to <14 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <14 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <14 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v7f64_to_v14i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v7f64_to_v14i32(i32 %cond, ptr addrspace(1) %out, <7 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <7 x double> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <7 x double> %phi_value to <14 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <14 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <14 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v9i64_to_v18i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v9i64_to_v18i32(i32 %cond, ptr addrspace(1) %out, <9 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <9 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <9 x i64> %phi_value to <18 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <18 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <18 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v10i64_to_v20i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v10i64_to_v20i32(i32 %cond, ptr addrspace(1) %out, <10 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <10 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <10 x i64> %phi_value to <20 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <20 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <20 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v11i64_to_v20i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v11i64_to_v20i32(i32 %cond, ptr addrspace(1) %out, <11 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <11 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <11 x i64> %phi_value to <22 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <22 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <22 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v12i64_to_v22i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v12i64_to_v22i32(i32 %cond, ptr addrspace(1) %out, <12 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <12 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <12 x i64> %phi_value to <24 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <24 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <24 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v13i64_to_v24i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v13i64_to_v24i32(i32 %cond, ptr addrspace(1) %out, <13 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <13 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <13 x i64> %phi_value to <26 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <26 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <26 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v14i64_to_v26i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v14i64_to_v26i32(i32 %cond, ptr addrspace(1) %out, <14 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <14 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <14 x i64> %phi_value to <28 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <28 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <28 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}bitcast_v15i64_to_v26i32:
; CHECK: ScratchSize: 0
define amdgpu_kernel void @bitcast_v15i64_to_v26i32(i32 %cond, ptr addrspace(1) %out, <15 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %phi_value = phi <15 x i64> [zeroinitializer, %entry], [%value, %if]
  %cast = bitcast <15 x i64> %phi_value to <30 x i32>
  %cmp1 = icmp eq i32 %cond, 1
  br i1 %cmp1, label %if, label %end

end:
  %phi_cast = phi <30 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <30 x i32> %phi_cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2bf16_to_i32:
define void @v_bitcast_v2bf16_to_i32(i32 %cond, ptr addrspace(1) %out, <2 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x bfloat> %value to i32
  br label %end

end:
  %phi = phi i32 [0, %entry], [%cast, %if]
  store i32 %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2bf16_to_v2i16:
define void @v_bitcast_v2bf16_to_v2i16(i32 %cond, ptr addrspace(1) %out, <2 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x bfloat> %value to <2 x i16>
  br label %end

end:
  %phi = phi <2 x i16> [zeroinitializer, %entry], [%cast, %if]
  store <2 x i16> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2bf16_to_v2f16:
define void @v_bitcast_v2bf16_to_v2f16(i32 %cond, ptr addrspace(1) %out, <2 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x bfloat> %value to <2 x half>
  br label %end

end:
  %phi = phi <2 x half> [zeroinitializer, %entry], [%cast, %if]
  store <2 x half> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2bf16_to_v4i8:
define void @v_bitcast_v2bf16_to_v4i8(i32 %cond, ptr addrspace(1) %out, <2 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x bfloat> %value to <4 x i8>
  br label %end

end:
  %phi = phi <4 x i8> [zeroinitializer, %entry], [%cast, %if]
  store <4 x i8> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v3bf16_to_v3i16:
define void @v_bitcast_v3bf16_to_v3i16(i32 %cond, ptr addrspace(1) %out, <3 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <3 x bfloat> %value to <3 x i16>
  br label %end

end:
  %phi = phi <3 x i16> [zeroinitializer, %entry], [%cast, %if]
  store <3 x i16> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v3bf16_to_v3f16:
define void @v_bitcast_v3bf16_to_v3f16(i32 %cond, ptr addrspace(1) %out, <3 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <3 x bfloat> %value to <3 x half>
  br label %end

end:
  %phi = phi <3 x half> [zeroinitializer, %entry], [%cast, %if]
  store <3 x half> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_i32_to_v2bf16:
define void @v_bitcast_i32_to_v2bf16(i32 %cond, ptr addrspace(1) %out, i32 %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast i32 %value to <2 x bfloat>
  br label %end

end:
  %phi = phi <2 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <2 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2i16_to_v2bf16:
define void @v_bitcast_v2i16_to_v2bf16(i32 %cond, ptr addrspace(1) %out, <2 x i16> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x i16> %value to <2 x bfloat>
  br label %end

end:
  %phi = phi <2 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <2 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2f16_to_v2bf16:
define void @v_bitcast_v2f16_to_v2bf16(i32 %cond, ptr addrspace(1) %out, <2 x half> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x half> %value to <2 x bfloat>
  br label %end

end:
  %phi = phi <2 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <2 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4i8_to_v2bf16:
define void @v_bitcast_v4i8_to_v2bf16(i32 %cond, ptr addrspace(1) %out, <4 x i8> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x i8> %value to <2 x bfloat>
  br label %end

end:
  %phi = phi <2 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <2 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v3i16_to_v3bf16:
define void @v_bitcast_v3i16_to_v3bf16(i32 %cond, ptr addrspace(1) %out, <3 x i16> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <3 x i16> %value to <3 x bfloat>
  br label %end

end:
  %phi = phi <3 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <3 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4bf16_to_v4f16:
define void @v_bitcast_v4bf16_to_v4f16(i32 %cond, ptr addrspace(1) %out, <4 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x bfloat> %value to <4 x half>
  br label %end

end:
  %phi = phi <4 x half> [zeroinitializer, %entry], [%cast, %if]
  store <4 x half> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4bf16_to_v4i16:
define void @v_bitcast_v4bf16_to_v4i16(i32 %cond, ptr addrspace(1) %out, <4 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x bfloat> %value to <4 x i16>
  br label %end

end:
  %phi = phi <4 x i16> [zeroinitializer, %entry], [%cast, %if]
  store <4 x i16> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4bf16_to_v2i32:
define void @v_bitcast_v4bf16_to_v2i32(i32 %cond, ptr addrspace(1) %out, <4 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x bfloat> %value to <2 x i32>
  br label %end

end:
  %phi = phi <2 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <2 x i32> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4bf16_to_v2f32:
define void @v_bitcast_v4bf16_to_v2f32(i32 %cond, ptr addrspace(1) %out, <4 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x bfloat> %value to <2 x float>
  br label %end

end:
  %phi = phi <2 x float> [zeroinitializer, %entry], [%cast, %if]
  store <2 x float> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4bf16_to_f64:
define void @v_bitcast_v4bf16_to_f64(i32 %cond, ptr addrspace(1) %out, <4 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x bfloat> %value to double
  br label %end

end:
  %phi = phi double [0.0, %entry], [%cast, %if]
  store double %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4bf16_to_i64:
define void @v_bitcast_v4bf16_to_i64(i32 %cond, ptr addrspace(1) %out, <4 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x bfloat> %value to i64
  br label %end

end:
  %phi = phi i64 [0, %entry], [%cast, %if]
  store i64 %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4bf16_to_v8i8:
define void @v_bitcast_v4bf16_to_v8i8(i32 %cond, ptr addrspace(1) %out, <4 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x bfloat> %value to <8 x i8>
  br label %end

end:
  %phi = phi <8 x i8> [zeroinitializer, %entry], [%cast, %if]
  store <8 x i8> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_i64_to_v4bf16:
define void @v_bitcast_i64_to_v4bf16(i32 %cond, ptr addrspace(1) %out, i64 %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast i64 %value to <4 x bfloat>
  br label %end

end:
  %phi = phi <4 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <4 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2f32_to_v4bf16:
define void @v_bitcast_v2f32_to_v4bf16(i32 %cond, ptr addrspace(1) %out, <2 x float> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x float> %value to <4 x bfloat>
  br label %end

end:
  %phi = phi <4 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <4 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2i32_to_v4bf16:
define void @v_bitcast_v2i32_to_v4bf16(i32 %cond, ptr addrspace(1) %out, <2 x i32> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x i32> %value to <4 x bfloat>
  br label %end

end:
  %phi = phi <4 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <4 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4i16_to_v4bf16:
define void @v_bitcast_v4i16_to_v4bf16(i32 %cond, ptr addrspace(1) %out, <4 x i16> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x i16> %value to <4 x bfloat>
  br label %end

end:
  %phi = phi <4 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <4 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4f16_to_v4bf16:
define void @v_bitcast_v4f16_to_v4bf16(i32 %cond, ptr addrspace(1) %out, <4 x half> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x half> %value to <4 x bfloat>
  br label %end

end:
  %phi = phi <4 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <4 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v6bf16_to_v6i16:
define void @v_bitcast_v6bf16_to_v6i16(i32 %cond, ptr addrspace(1) %out, <6 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <6 x bfloat> %value to <6 x i16>
  br label %end

end:
  %phi = phi <6 x i16> [zeroinitializer, %entry], [%cast, %if]
  store <6 x i16> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v6bf16_to_v6f16:
define void @v_bitcast_v6bf16_to_v6f16(i32 %cond, ptr addrspace(1) %out, <6 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <6 x bfloat> %value to <6 x half>
  br label %end

end:
  %phi = phi <6 x half> [zeroinitializer, %entry], [%cast, %if]
  store <6 x half> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v6bf16_to_v12i8:
define void @v_bitcast_v6bf16_to_v12i8(i32 %cond, ptr addrspace(1) %out, <6 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <6 x bfloat> %value to <12 x i8>
  br label %end

end:
  %phi = phi <12 x i8> [zeroinitializer, %entry], [%cast, %if]
  store <12 x i8> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v6f16_to_v6bf16:
define void @v_bitcast_v6f16_to_v6bf16(i32 %cond, ptr addrspace(1) %out, <6 x half> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <6 x half> %value to <6 x bfloat>
  br label %end

end:
  %phi = phi <6 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <6 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v6i16_to_v6bf16:
define void @v_bitcast_v6i16_to_v6bf16(i32 %cond, ptr addrspace(1) %out, <6 x i16> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <6 x i16> %value to <6 x bfloat>
  br label %end

end:
  %phi = phi <6 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <6 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v12i8_to_v6bf16:
define void @v_bitcast_v12i8_to_v6bf16(i32 %cond, ptr addrspace(1) %out, <12 x i8> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <12 x i8> %value to <6 x bfloat>
  br label %end

end:
  %phi = phi <6 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <6 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8bf16_to_v2f64:
define void @v_bitcast_v8bf16_to_v2f64(i32 %cond, ptr addrspace(1) %out, <8 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x bfloat> %value to <2 x double>
  br label %end

end:
  %phi = phi <2 x double> [zeroinitializer, %entry], [%cast, %if]
  store <2 x double> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8bf16_to_v2i64:
define void @v_bitcast_v8bf16_to_v2i64(i32 %cond, ptr addrspace(1) %out, <8 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x bfloat> %value to <2 x i64>
  br label %end

end:
  %phi = phi <2 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <2 x i64> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8bf16_to_v4f32:
define void @v_bitcast_v8bf16_to_v4f32(i32 %cond, ptr addrspace(1) %out, <8 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x bfloat> %value to <4 x float>
  br label %end

end:
  %phi = phi <4 x float> [zeroinitializer, %entry], [%cast, %if]
  store <4 x float> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8bf16_to_v4i32:
define void @v_bitcast_v8bf16_to_v4i32(i32 %cond, ptr addrspace(1) %out, <8 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x bfloat> %value to <4 x i32>
  br label %end

end:
  %phi = phi <4 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <4 x i32> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8bf16_to_v8f16:
define void @v_bitcast_v8bf16_to_v8f16(i32 %cond, ptr addrspace(1) %out, <8 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x bfloat> %value to <8 x half>
  br label %end

end:
  %phi = phi <8 x half> [zeroinitializer, %entry], [%cast, %if]
  store <8 x half> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8bf16_to_v8i16:
define void @v_bitcast_v8bf16_to_v8i16(i32 %cond, ptr addrspace(1) %out, <8 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x bfloat> %value to <8 x i16>
  br label %end

end:
  %phi = phi <8 x i16> [zeroinitializer, %entry], [%cast, %if]
  store <8 x i16> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8f16_to_v8bf16:
define void @v_bitcast_v8f16_to_v8bf16(i32 %cond, ptr addrspace(1) %out, <8 x half> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x half> %value to <8 x bfloat>
  br label %end

end:
  %phi = phi <8 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <8 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8i16_to_v8bf16:
define void @v_bitcast_v8i16_to_v8bf16(i32 %cond, ptr addrspace(1) %out, <8 x i16> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x i16> %value to <8 x bfloat>
  br label %end

end:
  %phi = phi <8 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <8 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16i8_to_v8bf16:
define void @v_bitcast_v16i8_to_v8bf16(i32 %cond, ptr addrspace(1) %out, <16 x i8> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x i8> %value to <8 x bfloat>
  br label %end

end:
  %phi = phi <8 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <8 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2i64_to_v8bf16:
define void @v_bitcast_v2i64_to_v8bf16(i32 %cond, ptr addrspace(1) %out, <2 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x i64> %value to <8 x bfloat>
  br label %end

end:
  %phi = phi <8 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <8 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v2f64_to_v8bf16:
define void @v_bitcast_v2f64_to_v8bf16(i32 %cond, ptr addrspace(1) %out, <2 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x double> %value to <8 x bfloat>
  br label %end

end:
  %phi = phi <8 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <8 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4i32_to_v8bf16:
define void @v_bitcast_v4i32_to_v8bf16(i32 %cond, ptr addrspace(1) %out, <4 x i32> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x i32> %value to <8 x bfloat>
  br label %end

end:
  %phi = phi <8 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <8 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4f32_to_v8bf16:
define void @v_bitcast_v4f32_to_v8bf16(i32 %cond, ptr addrspace(1) %out, <4 x float> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x float> %value to <8 x bfloat>
  br label %end

end:
  %phi = phi <8 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <8 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16bf16_to_v16i16:
define void @v_bitcast_v16bf16_to_v16i16(i32 %cond, ptr addrspace(1) %out, <16 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x bfloat> %value to <16 x i16>
  br label %end

end:
  %phi = phi <16 x i16> [zeroinitializer, %entry], [%cast, %if]
  store <16 x i16> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16bf16_to_v16f16:
define void @v_bitcast_v16bf16_to_v16f16(i32 %cond, ptr addrspace(1) %out, <16 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x bfloat> %value to <16 x half>
  br label %end

end:
  %phi = phi <16 x half> [zeroinitializer, %entry], [%cast, %if]
  store <16 x half> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16bf16_to_v8i32:
define void @v_bitcast_v16bf16_to_v8i32(i32 %cond, ptr addrspace(1) %out, <16 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x bfloat> %value to <8 x i32>
  br label %end

end:
  %phi = phi <8 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <8 x i32> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16bf16_to_v8f32:
define void @v_bitcast_v16bf16_to_v8f32(i32 %cond, ptr addrspace(1) %out, <16 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x bfloat> %value to <8 x float>
  br label %end

end:
  %phi = phi <8 x float> [zeroinitializer, %entry], [%cast, %if]
  store <8 x float> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16bf16_to_v4f64:
define void @v_bitcast_v16bf16_to_v4f64(i32 %cond, ptr addrspace(1) %out, <16 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x bfloat> %value to <4 x double>
  br label %end

end:
  %phi = phi <4 x double> [zeroinitializer, %entry], [%cast, %if]
  store <4 x double> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16bf16_to_v4i64:
define void @v_bitcast_v16bf16_to_v4i64(i32 %cond, ptr addrspace(1) %out, <16 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x bfloat> %value to <4 x i64>
  br label %end

end:
  %phi = phi <4 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <4 x i64> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16bf16_to_v32i8:
define void @v_bitcast_v16bf16_to_v32i8(i32 %cond, ptr addrspace(1) %out, <16 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x bfloat> %value to <32 x i8>
  br label %end

end:
  %phi = phi <32 x i8> [zeroinitializer, %entry], [%cast, %if]
  store <32 x i8> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8f32_to_v16bf16:
define void @v_bitcast_v8f32_to_v16bf16(i32 %cond, ptr addrspace(1) %out, <8 x float> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x float> %value to <16 x bfloat>
  br label %end

end:
  %phi = phi <16 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <16 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8i32_to_v16bf16:
define void @v_bitcast_v8i32_to_v16bf16(i32 %cond, ptr addrspace(1) %out, <8 x i32> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x i32> %value to <16 x bfloat>
  br label %end

end:
  %phi = phi <16 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <16 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4i64_to_v16bf16:
define void @v_bitcast_v4i64_to_v16bf16(i32 %cond, ptr addrspace(1) %out, <4 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x i64> %value to <16 x bfloat>
  br label %end

end:
  %phi = phi <16 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <16 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v4f64_to_v16bf16:
define void @v_bitcast_v4f64_to_v16bf16(i32 %cond, ptr addrspace(1) %out, <4 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <4 x double> %value to <16 x bfloat>
  br label %end

end:
  %phi = phi <16 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <16 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32i8_to_v16bf16:
define void @v_bitcast_v32i8_to_v16bf16(i32 %cond, ptr addrspace(1) %out, <32 x i8> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x i8> %value to <16 x bfloat>
  br label %end

end:
  %phi = phi <16 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <16 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32bf16_to_v8i64:
define void @v_bitcast_v32bf16_to_v8i64(i32 %cond, ptr addrspace(1) %out, <32 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x bfloat> %value to <8 x i64>
  br label %end

end:
  %phi = phi <8 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <8 x i64> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32bf16_to_v8f64:
define void @v_bitcast_v32bf16_to_v8f64(i32 %cond, ptr addrspace(1) %out, <32 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x bfloat> %value to <8 x double>
  br label %end

end:
  %phi = phi <8 x double> [zeroinitializer, %entry], [%cast, %if]
  store <8 x double> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32bf16_to_v16i32:
define void @v_bitcast_v32bf16_to_v16i32(i32 %cond, ptr addrspace(1) %out, <32 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x bfloat> %value to <16 x i32>
  br label %end

end:
  %phi = phi <16 x i32> [zeroinitializer, %entry], [%cast, %if]
  store <16 x i32> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32bf16_to_v16f32:
define void @v_bitcast_v32bf16_to_v16f32(i32 %cond, ptr addrspace(1) %out, <32 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x bfloat> %value to <16 x float>
  br label %end

end:
  %phi = phi <16 x float> [zeroinitializer, %entry], [%cast, %if]
  store <16 x float> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32bf16_to_v32f16:
define void @v_bitcast_v32bf16_to_v32f16(i32 %cond, ptr addrspace(1) %out, <32 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x bfloat> %value to <32 x half>
  br label %end

end:
  %phi = phi <32 x half> [zeroinitializer, %entry], [%cast, %if]
  store <32 x half> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32bf16_to_v32i16:
define void @v_bitcast_v32bf16_to_v32i16(i32 %cond, ptr addrspace(1) %out, <32 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x bfloat> %value to <32 x i16>
  br label %end

end:
  %phi = phi <32 x i16> [zeroinitializer, %entry], [%cast, %if]
  store <32 x i16> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32bf16_to_v64i8:
define void @v_bitcast_v32bf16_to_v64i8(i32 %cond, ptr addrspace(1) %out, <32 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x bfloat> %value to <64 x i8>
  br label %end

end:
  %phi = phi <64 x i8> [zeroinitializer, %entry], [%cast, %if]
  store <64 x i8> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v64i8_to_v32bf16:
define void @v_bitcast_v64i8_to_v32bf16(i32 %cond, ptr addrspace(1) %out, <64 x i8> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <64 x i8> %value to <32 x bfloat>
  br label %end

end:
  %phi = phi <32 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <32 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32i16_to_v32bf16:
define void @v_bitcast_v32i16_to_v32bf16(i32 %cond, ptr addrspace(1) %out, <32 x i16> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x i16> %value to <32 x bfloat>
  br label %end

end:
  %phi = phi <32 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <32 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32f16_to_v32bf16:
define void @v_bitcast_v32f16_to_v32bf16(i32 %cond, ptr addrspace(1) %out, <32 x half> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x half> %value to <32 x bfloat>
  br label %end

end:
  %phi = phi <32 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <32 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16i32_to_v32bf16:
define void @v_bitcast_v16i32_to_v32bf16(i32 %cond, ptr addrspace(1) %out, <16 x i32> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x i32> %value to <32 x bfloat>
  br label %end

end:
  %phi = phi <32 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <32 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v16f32_to_v32bf16:
define void @v_bitcast_v16f32_to_v32bf16(i32 %cond, ptr addrspace(1) %out, <16 x float> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <16 x float> %value to <32 x bfloat>
  br label %end

end:
  %phi = phi <32 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <32 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8f64_to_v32bf16:
define void @v_bitcast_v8f64_to_v32bf16(i32 %cond, ptr addrspace(1) %out, <8 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x double> %value to <32 x bfloat>
  br label %end

end:
  %phi = phi <32 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <32 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v8i64_to_v32bf16:
define void @v_bitcast_v8i64_to_v32bf16(i32 %cond, ptr addrspace(1) %out, <8 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <8 x i64> %value to <32 x bfloat>
  br label %end

end:
  %phi = phi <32 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <32 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}








; CHECK-LABEL: {{^}}v_bitcast_v32f32_to_v64bf16:
define void @v_bitcast_v32f32_to_v64bf16(i32 %cond, ptr addrspace(1) %out, <32 x float> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x float> %value to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <64 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v32i32_to_v64bf16:
define void @v_bitcast_v32i32_to_v64bf16(i32 %cond, ptr addrspace(1) %out, <32 x i32> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <32 x i32> %value to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <64 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v64i16_to_v64bf16:
define void @v_bitcast_v64i16_to_v64bf16(i32 %cond, ptr addrspace(1) %out, <64 x i16> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <64 x i16> %value to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <64 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v64f16_to_v64bf16:
define void @v_bitcast_v64f16_to_v64bf16(i32 %cond, ptr addrspace(1) %out, <64 x half> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <64 x half> %value to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <64 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v128i8_to_v64bf16:
define void @v_bitcast_v128i8_to_v64bf16(i32 %cond, ptr addrspace(1) %out, <128 x i8> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <128 x i8> %value to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [zeroinitializer, %entry], [%cast, %if]
  store <64 x bfloat> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v64bf16_to_v64i16:
define void @v_bitcast_v64bf16_to_v64i16(i32 %cond, ptr addrspace(1) %out, <64 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <64 x bfloat> %value to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [zeroinitializer, %entry], [%cast, %if]
  store <64 x i16> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v64bf16_to_v64f16:
define void @v_bitcast_v64bf16_to_v64f16(i32 %cond, ptr addrspace(1) %out, <64 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <64 x bfloat> %value to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [zeroinitializer, %entry], [%cast, %if]
  store <64 x half> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v64bf16_to_v128i8:
define void @v_bitcast_v64bf16_to_v128i8(i32 %cond, ptr addrspace(1) %out, <64 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <64 x bfloat> %value to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [zeroinitializer, %entry], [%cast, %if]
  store <128 x i8> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v64bf16_to_v16i64:
define void @v_bitcast_v64bf16_to_v16i64(i32 %cond, ptr addrspace(1) %out, <64 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <64 x bfloat> %value to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <16 x i64> %phi, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_bitcast_v64bf16_to_v16f64:
define void @v_bitcast_v64bf16_to_v16f64(i32 %cond, ptr addrspace(1) %out, <64 x bfloat> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <64 x bfloat> %value to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [zeroinitializer, %entry], [%cast, %if]
  store <16 x double> %phi, ptr addrspace(1) %out
  ret void
}
