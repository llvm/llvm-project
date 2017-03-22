; RUN: opt -mtriple=amdgcn-amd-amdhsa -load-store-vectorizer -S -o - %s | FileCheck %s
; Copy of test/CodeGen/AMDGPU/merge-stores.ll with some additions

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

; TODO: Vector element tests
; TODO: Non-zero base offset for load and store combinations
; TODO: Same base addrspacecasted


; CHECK-LABEL: @merge_global_store_2_constants_i8(
; CHECK: store <2 x i8> <i8 -56, i8 123>, <2 x i8> addrspace(1)* %{{[0-9]+}}, align 2
define amdgpu_kernel void @merge_global_store_2_constants_i8(i8 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i32 1

  store i8 123, i8 addrspace(1)* %out.gep.1
  store i8 456, i8 addrspace(1)* %out, align 2
  ret void
}

; CHECK-LABEL: @merge_global_store_2_constants_i8_natural_align
; CHECK: store <2 x i8>
define amdgpu_kernel void @merge_global_store_2_constants_i8_natural_align(i8 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i32 1

  store i8 123, i8 addrspace(1)* %out.gep.1
  store i8 456, i8 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_2_constants_i16
; CHECK: store <2 x i16> <i16 456, i16 123>, <2 x i16> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_2_constants_i16(i16 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i16, i16 addrspace(1)* %out, i32 1

  store i16 123, i16 addrspace(1)* %out.gep.1
  store i16 456, i16 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: @merge_global_store_2_constants_0_i16
; CHECK: store <2 x i16> zeroinitializer, <2 x i16> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_2_constants_0_i16(i16 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i16, i16 addrspace(1)* %out, i32 1

  store i16 0, i16 addrspace(1)* %out.gep.1
  store i16 0, i16 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: @merge_global_store_2_constants_i16_natural_align
; CHECK: store <2 x i16>
define amdgpu_kernel void @merge_global_store_2_constants_i16_natural_align(i16 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i16, i16 addrspace(1)* %out, i32 1

  store i16 123, i16 addrspace(1)* %out.gep.1
  store i16 456, i16 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_2_constants_half_natural_align
; CHECK: store <2 x half>
define amdgpu_kernel void @merge_global_store_2_constants_half_natural_align(half addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr half, half addrspace(1)* %out, i32 1

  store half 2.0, half addrspace(1)* %out.gep.1
  store half 1.0, half addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_2_constants_i32
; CHECK: store <2 x i32> <i32 456, i32 123>, <2 x i32> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_2_constants_i32(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_2_constants_i32_f32
; CHECK: store <2 x i32> <i32 456, i32 1065353216>, <2 x i32> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_2_constants_i32_f32(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.1.bc = bitcast i32 addrspace(1)* %out.gep.1 to float addrspace(1)*
  store float 1.0, float addrspace(1)* %out.gep.1.bc
  store i32 456, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_2_constants_f32_i32
; CHECK  store <2 x float> <float 4.000000e+00, float 0x370EC00000000000>, <2 x float> addrspace(1)* %{{[0-9]+$}}
define amdgpu_kernel void @merge_global_store_2_constants_f32_i32(float addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.1.bc = bitcast float addrspace(1)* %out.gep.1 to i32 addrspace(1)*
  store i32 123, i32 addrspace(1)* %out.gep.1.bc
  store float 4.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_4_constants_i32
; CHECK: store <4 x i32> <i32 1234, i32 123, i32 456, i32 333>, <4 x i32> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_4_constants_i32(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out.gep.2
  store i32 333, i32 addrspace(1)* %out.gep.3
  store i32 1234, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_4_constants_f32_order
; CHECK: store <4 x float> <float 8.000000e+00, float 1.000000e+00, float 2.000000e+00, float 4.000000e+00>, <4 x float> addrspace(1)* %{{[0-9]+}}
define amdgpu_kernel void @merge_global_store_4_constants_f32_order(float addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr float, float addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr float, float addrspace(1)* %out, i32 3

  store float 8.0, float addrspace(1)* %out
  store float 1.0, float addrspace(1)* %out.gep.1
  store float 2.0, float addrspace(1)* %out.gep.2
  store float 4.0, float addrspace(1)* %out.gep.3
  ret void
}

; First store is out of order.
; CHECK-LABEL: @merge_global_store_4_constants_f32
; CHECK: store <4 x float> <float 8.000000e+00, float 1.000000e+00, float 2.000000e+00, float 4.000000e+00>, <4 x float> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_4_constants_f32(float addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr float, float addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr float, float addrspace(1)* %out, i32 3

  store float 1.0, float addrspace(1)* %out.gep.1
  store float 2.0, float addrspace(1)* %out.gep.2
  store float 4.0, float addrspace(1)* %out.gep.3
  store float 8.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_4_constants_mixed_i32_f32
; CHECK: store <4 x i32> <i32 1090519040, i32 11, i32 1073741824, i32 17>, <4 x i32> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_4_constants_mixed_i32_f32(float addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr float, float addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr float, float addrspace(1)* %out, i32 3

  %out.gep.1.bc = bitcast float addrspace(1)* %out.gep.1 to i32 addrspace(1)*
  %out.gep.3.bc = bitcast float addrspace(1)* %out.gep.3 to i32 addrspace(1)*

  store i32 11, i32 addrspace(1)* %out.gep.1.bc
  store float 2.0, float addrspace(1)* %out.gep.2
  store i32 17, i32 addrspace(1)* %out.gep.3.bc
  store float 8.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_3_constants_i32
; CHECK: store <3 x i32> <i32 1234, i32 123, i32 456>, <3 x i32> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_3_constants_i32(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out.gep.2
  store i32 1234, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_2_constants_i64
; CHECK: store <2 x i64> <i64 456, i64 123>, <2 x i64> addrspace(1)* %{{[0-9]+}}, align 8
define amdgpu_kernel void @merge_global_store_2_constants_i64(i64 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i64, i64 addrspace(1)* %out, i64 1

  store i64 123, i64 addrspace(1)* %out.gep.1
  store i64 456, i64 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_4_constants_i64
; CHECK: store <2 x i64> <i64 456, i64 333>, <2 x i64> addrspace(1)* %{{[0-9]+}}, align 8
; CHECK: store <2 x i64> <i64 1234, i64 123>, <2 x i64> addrspace(1)* %{{[0-9]+}}, align 8
define amdgpu_kernel void @merge_global_store_4_constants_i64(i64 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i64, i64 addrspace(1)* %out, i64 1
  %out.gep.2 = getelementptr i64, i64 addrspace(1)* %out, i64 2
  %out.gep.3 = getelementptr i64, i64 addrspace(1)* %out, i64 3

  store i64 123, i64 addrspace(1)* %out.gep.1
  store i64 456, i64 addrspace(1)* %out.gep.2
  store i64 333, i64 addrspace(1)* %out.gep.3
  store i64 1234, i64 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_2_adjacent_loads_i32
; CHECK: [[LOAD:%[^ ]+]] = load <2 x i32>
; CHECK: [[ELT0:%[^ ]+]] = extractelement <2 x i32> [[LOAD]], i32 0
; CHECK: [[ELT1:%[^ ]+]] = extractelement <2 x i32> [[LOAD]], i32 1
; CHECK: [[INSERT0:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[ELT0]], i32 0
; CHECK: [[INSERT1:%[^ ]+]] = insertelement <2 x i32> [[INSERT0]], i32 [[ELT1]], i32 1
; CHECK: store <2 x i32> [[INSERT1]]
define amdgpu_kernel void @merge_global_store_2_adjacent_loads_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1

  %lo = load i32, i32 addrspace(1)* %in
  %hi = load i32, i32 addrspace(1)* %in.gep.1

  store i32 %lo, i32 addrspace(1)* %out
  store i32 %hi, i32 addrspace(1)* %out.gep.1
  ret void
}

; CHECK-LABEL: @merge_global_store_2_adjacent_loads_i32_nonzero_base
; CHECK: extractelement
; CHECK: extractelement
; CHECK: insertelement
; CHECK: insertelement
; CHECK: store <2 x i32>
define amdgpu_kernel void @merge_global_store_2_adjacent_loads_i32_nonzero_base(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %in.gep.0 = getelementptr i32, i32 addrspace(1)* %in, i32 2
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 3

  %out.gep.0 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %lo = load i32, i32 addrspace(1)* %in.gep.0
  %hi = load i32, i32 addrspace(1)* %in.gep.1

  store i32 %lo, i32 addrspace(1)* %out.gep.0
  store i32 %hi, i32 addrspace(1)* %out.gep.1
  ret void
}

; CHECK-LABEL: @merge_global_store_2_adjacent_loads_shuffle_i32
; CHECK: [[LOAD:%[^ ]+]] = load <2 x i32>
; CHECK: [[ELT0:%[^ ]+]] = extractelement <2 x i32> [[LOAD]], i32 0
; CHECK: [[ELT1:%[^ ]+]] = extractelement <2 x i32> [[LOAD]], i32 1
; CHECK: [[INSERT0:%[^ ]+]] = insertelement <2 x i32> undef, i32 [[ELT1]], i32 0
; CHECK: [[INSERT1:%[^ ]+]] = insertelement <2 x i32> [[INSERT0]], i32 [[ELT0]], i32 1
; CHECK: store <2 x i32> [[INSERT1]]
define amdgpu_kernel void @merge_global_store_2_adjacent_loads_shuffle_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1

  %lo = load i32, i32 addrspace(1)* %in
  %hi = load i32, i32 addrspace(1)* %in.gep.1

  store i32 %hi, i32 addrspace(1)* %out
  store i32 %lo, i32 addrspace(1)* %out.gep.1
  ret void
}

; CHECK-LABEL: @merge_global_store_4_adjacent_loads_i32
; CHECK: load <4 x i32>
; CHECK: store <4 x i32>
define amdgpu_kernel void @merge_global_store_4_adjacent_loads_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 2
  %in.gep.3 = getelementptr i32, i32 addrspace(1)* %in, i32 3

  %x = load i32, i32 addrspace(1)* %in
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2
  %w = load i32, i32 addrspace(1)* %in.gep.3

  store i32 %x, i32 addrspace(1)* %out
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %z, i32 addrspace(1)* %out.gep.2
  store i32 %w, i32 addrspace(1)* %out.gep.3
  ret void
}

; CHECK-LABEL: @merge_global_store_3_adjacent_loads_i32
; CHECK: load <3 x i32>
; CHECK: store <3 x i32>
define amdgpu_kernel void @merge_global_store_3_adjacent_loads_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 2

  %x = load i32, i32 addrspace(1)* %in
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2

  store i32 %x, i32 addrspace(1)* %out
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %z, i32 addrspace(1)* %out.gep.2
  ret void
}

; CHECK-LABEL: @merge_global_store_4_adjacent_loads_f32
; CHECK: load <4 x float>
; CHECK: store <4 x float>
define amdgpu_kernel void @merge_global_store_4_adjacent_loads_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr float, float addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr float, float addrspace(1)* %out, i32 3
  %in.gep.1 = getelementptr float, float addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr float, float addrspace(1)* %in, i32 2
  %in.gep.3 = getelementptr float, float addrspace(1)* %in, i32 3

  %x = load float, float addrspace(1)* %in
  %y = load float, float addrspace(1)* %in.gep.1
  %z = load float, float addrspace(1)* %in.gep.2
  %w = load float, float addrspace(1)* %in.gep.3

  store float %x, float addrspace(1)* %out
  store float %y, float addrspace(1)* %out.gep.1
  store float %z, float addrspace(1)* %out.gep.2
  store float %w, float addrspace(1)* %out.gep.3
  ret void
}

; CHECK-LABEL: @merge_global_store_4_adjacent_loads_i32_nonzero_base
; CHECK: load <4 x i32>
; CHECK: store <4 x i32>
define amdgpu_kernel void @merge_global_store_4_adjacent_loads_i32_nonzero_base(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %in.gep.0 = getelementptr i32, i32 addrspace(1)* %in, i32 11
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 12
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 13
  %in.gep.3 = getelementptr i32, i32 addrspace(1)* %in, i32 14
  %out.gep.0 = getelementptr i32, i32 addrspace(1)* %out, i32 7
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 8
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 9
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 10

  %x = load i32, i32 addrspace(1)* %in.gep.0
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2
  %w = load i32, i32 addrspace(1)* %in.gep.3

  store i32 %x, i32 addrspace(1)* %out.gep.0
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %z, i32 addrspace(1)* %out.gep.2
  store i32 %w, i32 addrspace(1)* %out.gep.3
  ret void
}

; CHECK-LABEL: @merge_global_store_4_adjacent_loads_inverse_i32
; CHECK: load <4 x i32>
; CHECK: store <4 x i32>
define amdgpu_kernel void @merge_global_store_4_adjacent_loads_inverse_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 2
  %in.gep.3 = getelementptr i32, i32 addrspace(1)* %in, i32 3

  %x = load i32, i32 addrspace(1)* %in
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2
  %w = load i32, i32 addrspace(1)* %in.gep.3

  ; Make sure the barrier doesn't stop this
  tail call void @llvm.amdgcn.s.barrier() #1

  store i32 %w, i32 addrspace(1)* %out.gep.3
  store i32 %z, i32 addrspace(1)* %out.gep.2
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %x, i32 addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @merge_global_store_4_adjacent_loads_shuffle_i32
; CHECK: load <4 x i32>
; CHECK: store <4 x i32>
define amdgpu_kernel void @merge_global_store_4_adjacent_loads_shuffle_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 2
  %in.gep.3 = getelementptr i32, i32 addrspace(1)* %in, i32 3

  %x = load i32, i32 addrspace(1)* %in
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2
  %w = load i32, i32 addrspace(1)* %in.gep.3

  ; Make sure the barrier doesn't stop this
  tail call void @llvm.amdgcn.s.barrier() #1

  store i32 %w, i32 addrspace(1)* %out
  store i32 %z, i32 addrspace(1)* %out.gep.1
  store i32 %y, i32 addrspace(1)* %out.gep.2
  store i32 %x, i32 addrspace(1)* %out.gep.3

  ret void
}

; CHECK-LABEL: @merge_global_store_4_adjacent_loads_i8
; CHECK: load <4 x i8>
; CHECK: extractelement <4 x i8>
; CHECK: extractelement <4 x i8>
; CHECK: extractelement <4 x i8>
; CHECK: extractelement <4 x i8>
; CHECK: insertelement <4 x i8>
; CHECK: insertelement <4 x i8>
; CHECK: insertelement <4 x i8>
; CHECK: insertelement <4 x i8>
; CHECK: store <4 x i8>
define amdgpu_kernel void @merge_global_store_4_adjacent_loads_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i8 1
  %out.gep.2 = getelementptr i8, i8 addrspace(1)* %out, i8 2
  %out.gep.3 = getelementptr i8, i8 addrspace(1)* %out, i8 3
  %in.gep.1 = getelementptr i8, i8 addrspace(1)* %in, i8 1
  %in.gep.2 = getelementptr i8, i8 addrspace(1)* %in, i8 2
  %in.gep.3 = getelementptr i8, i8 addrspace(1)* %in, i8 3

  %x = load i8, i8 addrspace(1)* %in, align 4
  %y = load i8, i8 addrspace(1)* %in.gep.1
  %z = load i8, i8 addrspace(1)* %in.gep.2
  %w = load i8, i8 addrspace(1)* %in.gep.3

  store i8 %x, i8 addrspace(1)* %out, align 4
  store i8 %y, i8 addrspace(1)* %out.gep.1
  store i8 %z, i8 addrspace(1)* %out.gep.2
  store i8 %w, i8 addrspace(1)* %out.gep.3
  ret void
}

; CHECK-LABEL: @merge_global_store_4_adjacent_loads_i8_natural_align
; CHECK: load <4 x i8>
; CHECK: store <4 x i8>
define amdgpu_kernel void @merge_global_store_4_adjacent_loads_i8_natural_align(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i8 1
  %out.gep.2 = getelementptr i8, i8 addrspace(1)* %out, i8 2
  %out.gep.3 = getelementptr i8, i8 addrspace(1)* %out, i8 3
  %in.gep.1 = getelementptr i8, i8 addrspace(1)* %in, i8 1
  %in.gep.2 = getelementptr i8, i8 addrspace(1)* %in, i8 2
  %in.gep.3 = getelementptr i8, i8 addrspace(1)* %in, i8 3

  %x = load i8, i8 addrspace(1)* %in
  %y = load i8, i8 addrspace(1)* %in.gep.1
  %z = load i8, i8 addrspace(1)* %in.gep.2
  %w = load i8, i8 addrspace(1)* %in.gep.3

  store i8 %x, i8 addrspace(1)* %out
  store i8 %y, i8 addrspace(1)* %out.gep.1
  store i8 %z, i8 addrspace(1)* %out.gep.2
  store i8 %w, i8 addrspace(1)* %out.gep.3
  ret void
}

; CHECK-LABEL: @merge_global_store_4_vector_elts_loads_v4i32
; CHECK: load <4 x i32>
; CHECK: store <4 x i32>
define amdgpu_kernel void @merge_global_store_4_vector_elts_loads_v4i32(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %vec = load <4 x i32>, <4 x i32> addrspace(1)* %in

  %x = extractelement <4 x i32> %vec, i32 0
  %y = extractelement <4 x i32> %vec, i32 1
  %z = extractelement <4 x i32> %vec, i32 2
  %w = extractelement <4 x i32> %vec, i32 3

  store i32 %x, i32 addrspace(1)* %out
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %z, i32 addrspace(1)* %out.gep.2
  store i32 %w, i32 addrspace(1)* %out.gep.3
  ret void
}

; CHECK-LABEL: @merge_local_store_2_constants_i8
; CHECK: store <2 x i8> <i8 -56, i8 123>, <2 x i8> addrspace(3)* %{{[0-9]+}}, align 2
define amdgpu_kernel void @merge_local_store_2_constants_i8(i8 addrspace(3)* %out) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(3)* %out, i32 1

  store i8 123, i8 addrspace(3)* %out.gep.1
  store i8 456, i8 addrspace(3)* %out, align 2
  ret void
}

; CHECK-LABEL: @merge_local_store_2_constants_i32
; CHECK: store <2 x i32> <i32 456, i32 123>, <2 x i32> addrspace(3)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_local_store_2_constants_i32(i32 addrspace(3)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(3)* %out, i32 1

  store i32 123, i32 addrspace(3)* %out.gep.1
  store i32 456, i32 addrspace(3)* %out
  ret void
}

; CHECK-LABEL: @merge_local_store_2_constants_i32_align_2
; CHECK: store i32
; CHECK: store i32
define amdgpu_kernel void @merge_local_store_2_constants_i32_align_2(i32 addrspace(3)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(3)* %out, i32 1

  store i32 123, i32 addrspace(3)* %out.gep.1, align 2
  store i32 456, i32 addrspace(3)* %out, align 2
  ret void
}

; CHECK-LABEL: @merge_local_store_4_constants_i32
; CHECK: store <2 x i32> <i32 456, i32 333>, <2 x i32> addrspace(3)*
; CHECK: store <2 x i32> <i32 1234, i32 123>, <2 x i32> addrspace(3)*
define amdgpu_kernel void @merge_local_store_4_constants_i32(i32 addrspace(3)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(3)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(3)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(3)* %out, i32 3

  store i32 123, i32 addrspace(3)* %out.gep.1
  store i32 456, i32 addrspace(3)* %out.gep.2
  store i32 333, i32 addrspace(3)* %out.gep.3
  store i32 1234, i32 addrspace(3)* %out
  ret void
}

; CHECK-LABEL: @merge_global_store_5_constants_i32
; CHECK: store <4 x i32> <i32 9, i32 12, i32 16, i32 -12>, <4 x i32> addrspace(1)* %{{[0-9]+}}, align 4
; CHECK: store i32
define amdgpu_kernel void @merge_global_store_5_constants_i32(i32 addrspace(1)* %out) {
  store i32 9, i32 addrspace(1)* %out, align 4
  %idx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 1
  store i32 12, i32 addrspace(1)* %idx1, align 4
  %idx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 2
  store i32 16, i32 addrspace(1)* %idx2, align 4
  %idx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 3
  store i32 -12, i32 addrspace(1)* %idx3, align 4
  %idx4 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 4
  store i32 11, i32 addrspace(1)* %idx4, align 4
  ret void
}

; CHECK-LABEL: @merge_global_store_6_constants_i32
; CHECK: store <4 x i32> <i32 13, i32 15, i32 62, i32 63>, <4 x i32> addrspace(1)* %{{[0-9]+}}, align 4
; CHECK: store <2 x i32> <i32 11, i32 123>, <2 x i32> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_6_constants_i32(i32 addrspace(1)* %out) {
  store i32 13, i32 addrspace(1)* %out, align 4
  %idx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 1
  store i32 15, i32 addrspace(1)* %idx1, align 4
  %idx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 2
  store i32 62, i32 addrspace(1)* %idx2, align 4
  %idx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 3
  store i32 63, i32 addrspace(1)* %idx3, align 4
  %idx4 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 4
  store i32 11, i32 addrspace(1)* %idx4, align 4
  %idx5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 5
  store i32 123, i32 addrspace(1)* %idx5, align 4
  ret void
}

; CHECK-LABEL: @merge_global_store_7_constants_i32
; CHECK: store <4 x i32> <i32 34, i32 999, i32 65, i32 33>, <4 x i32> addrspace(1)* %{{[0-9]+}}, align 4
; CHECK: store <3 x i32> <i32 98, i32 91, i32 212>, <3 x i32> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_7_constants_i32(i32 addrspace(1)* %out) {
  store i32 34, i32 addrspace(1)* %out, align 4
  %idx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 1
  store i32 999, i32 addrspace(1)* %idx1, align 4
  %idx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 2
  store i32 65, i32 addrspace(1)* %idx2, align 4
  %idx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 3
  store i32 33, i32 addrspace(1)* %idx3, align 4
  %idx4 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 4
  store i32 98, i32 addrspace(1)* %idx4, align 4
  %idx5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 5
  store i32 91, i32 addrspace(1)* %idx5, align 4
  %idx6 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 6
  store i32 212, i32 addrspace(1)* %idx6, align 4
  ret void
}

; CHECK-LABEL: @merge_global_store_8_constants_i32
; CHECK: store <4 x i32> <i32 34, i32 999, i32 65, i32 33>, <4 x i32> addrspace(1)* %{{[0-9]+}}, align 4
; CHECK: store <4 x i32> <i32 98, i32 91, i32 212, i32 999>, <4 x i32> addrspace(1)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @merge_global_store_8_constants_i32(i32 addrspace(1)* %out) {
  store i32 34, i32 addrspace(1)* %out, align 4
  %idx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 1
  store i32 999, i32 addrspace(1)* %idx1, align 4
  %idx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 2
  store i32 65, i32 addrspace(1)* %idx2, align 4
  %idx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 3
  store i32 33, i32 addrspace(1)* %idx3, align 4
  %idx4 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 4
  store i32 98, i32 addrspace(1)* %idx4, align 4
  %idx5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 5
  store i32 91, i32 addrspace(1)* %idx5, align 4
  %idx6 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 6
  store i32 212, i32 addrspace(1)* %idx6, align 4
  %idx7 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 7
  store i32 999, i32 addrspace(1)* %idx7, align 4
  ret void
}

; CHECK-LABEL: @copy_v3i32_align4
; CHECK: %vec = load <3 x i32>, <3 x i32> addrspace(1)* %in, align 4
; CHECK: store <3 x i32> %vec, <3 x i32> addrspace(1)* %out
define amdgpu_kernel void @copy_v3i32_align4(<3 x i32> addrspace(1)* noalias %out, <3 x i32> addrspace(1)* noalias %in) #0 {
  %vec = load <3 x i32>, <3 x i32> addrspace(1)* %in, align 4
  store <3 x i32> %vec, <3 x i32> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @copy_v3i64_align4
; CHECK: %vec = load <3 x i64>, <3 x i64> addrspace(1)* %in, align 4
; CHECK: store <3 x i64> %vec, <3 x i64> addrspace(1)* %out
define amdgpu_kernel void @copy_v3i64_align4(<3 x i64> addrspace(1)* noalias %out, <3 x i64> addrspace(1)* noalias %in) #0 {
  %vec = load <3 x i64>, <3 x i64> addrspace(1)* %in, align 4
  store <3 x i64> %vec, <3 x i64> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @copy_v3f32_align4
; CHECK: %vec = load <3 x float>, <3 x float> addrspace(1)* %in, align 4
; CHECK: store <3 x float>
define amdgpu_kernel void @copy_v3f32_align4(<3 x float> addrspace(1)* noalias %out, <3 x float> addrspace(1)* noalias %in) #0 {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %in, align 4
  %fadd = fadd <3 x float> %vec, <float 1.0, float 2.0, float 4.0>
  store <3 x float> %fadd, <3 x float> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @copy_v3f64_align4
; CHECK: %vec = load <3 x double>, <3 x double> addrspace(1)* %in, align 4
; CHECK: store <3 x double> %fadd, <3 x double> addrspace(1)* %out
define amdgpu_kernel void @copy_v3f64_align4(<3 x double> addrspace(1)* noalias %out, <3 x double> addrspace(1)* noalias %in) #0 {
  %vec = load <3 x double>, <3 x double> addrspace(1)* %in, align 4
  %fadd = fadd <3 x double> %vec, <double 1.0, double 2.0, double 4.0>
  store <3 x double> %fadd, <3 x double> addrspace(1)* %out
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
