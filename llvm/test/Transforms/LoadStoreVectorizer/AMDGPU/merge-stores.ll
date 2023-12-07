; RUN: opt -mtriple=amdgcn-amd-amdhsa --mcpu=hawaii -passes=load-store-vectorizer -S -o - %s | FileCheck %s
; Copy of test/CodeGen/AMDGPU/merge-stores.ll with some additions

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; TODO: Vector element tests
; TODO: Non-zero base offset for load and store combinations
; TODO: Same base addrspacecasted


define amdgpu_kernel void @merge_global_store_2_constants_i8(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_i8(
; CHECK-NEXT:    store <2 x i8> <i8 -56, i8 123>, ptr addrspace(1) [[OUT:%.*]], align 2
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i8, ptr addrspace(1) %out, i32 1

  store i8 123, ptr addrspace(1) %out.gep.1
  store i8 456, ptr addrspace(1) %out, align 2
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_i8_natural_align(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_i8_natural_align(
; CHECK-NEXT:    store <2 x i8> <i8 -56, i8 123>, ptr addrspace(1) [[OUT:%.*]], align 1
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i8, ptr addrspace(1) %out, i32 1

  store i8 123, ptr addrspace(1) %out.gep.1
  store i8 456, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_i16(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_i16(
; CHECK-NEXT:    store <2 x i16> <i16 456, i16 123>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i16, ptr addrspace(1) %out, i32 1

  store i16 123, ptr addrspace(1) %out.gep.1
  store i16 456, ptr addrspace(1) %out, align 4
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_0_i16(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_0_i16(
; CHECK-NEXT:    store <2 x i16> zeroinitializer, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i16, ptr addrspace(1) %out, i32 1

  store i16 0, ptr addrspace(1) %out.gep.1
  store i16 0, ptr addrspace(1) %out, align 4
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_i16_natural_align(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_i16_natural_align(
; CHECK-NEXT:    store <2 x i16> <i16 456, i16 123>, ptr addrspace(1) [[OUT:%.*]], align 2
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i16, ptr addrspace(1) %out, i32 1

  store i16 123, ptr addrspace(1) %out.gep.1
  store i16 456, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_i16_align_1(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_i16_align_1(
; CHECK-NEXT:    store <2 x i16> <i16 456, i16 123>, ptr addrspace(1) [[OUT:%.*]], align 1
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i16, ptr addrspace(1) %out, i32 1

  store i16 123, ptr addrspace(1) %out.gep.1, align 1
  store i16 456, ptr addrspace(1) %out, align 1
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_half_natural_align(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_half_natural_align(
; CHECK-NEXT:    store <2 x half> <half 0xH3C00, half 0xH4000>, ptr addrspace(1) [[OUT:%.*]], align 2
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr half, ptr addrspace(1) %out, i32 1

  store half 2.0, ptr addrspace(1) %out.gep.1
  store half 1.0, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_half_align_1(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_half_align_1(
; CHECK-NEXT:    store <2 x half> <half 0xH3C00, half 0xH4000>, ptr addrspace(1) [[OUT:%.*]], align 1
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr half, ptr addrspace(1) %out, i32 1

  store half 2.0, ptr addrspace(1) %out.gep.1, align 1
  store half 1.0, ptr addrspace(1) %out, align 1
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_i32(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_i32(
; CHECK-NEXT:    store <2 x i32> <i32 456, i32 123>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1

  store i32 123, ptr addrspace(1) %out.gep.1
  store i32 456, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_i32_f32(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_i32_f32(
; CHECK-NEXT:    store <2 x i32> <i32 456, i32 1065353216>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store float 1.0, ptr addrspace(1) %out.gep.1
  store i32 456, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_f32_i32(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_f32_i32(
; CHECK-NEXT:    store <2 x i32> <i32 1082130432, i32 123>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr float, ptr addrspace(1) %out, i32 1
  store i32 123, ptr addrspace(1) %out.gep.1
  store float 4.0, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_4_constants_i32(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_4_constants_i32(
; CHECK-NEXT:    store <4 x i32> <i32 1234, i32 123, i32 456, i32 333>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr i32, ptr addrspace(1) %out, i32 3

  store i32 123, ptr addrspace(1) %out.gep.1
  store i32 456, ptr addrspace(1) %out.gep.2
  store i32 333, ptr addrspace(1) %out.gep.3
  store i32 1234, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_4_constants_f32_order(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_4_constants_f32_order(
; CHECK-NEXT:    store <4 x float> <float 8.000000e+00, float 1.000000e+00, float 2.000000e+00, float 4.000000e+00>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr float, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr float, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr float, ptr addrspace(1) %out, i32 3

  store float 8.0, ptr addrspace(1) %out
  store float 1.0, ptr addrspace(1) %out.gep.1
  store float 2.0, ptr addrspace(1) %out.gep.2
  store float 4.0, ptr addrspace(1) %out.gep.3
  ret void
}

; First store is out of order.
define amdgpu_kernel void @merge_global_store_4_constants_f32(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_4_constants_f32(
; CHECK-NEXT:    store <4 x float> <float 8.000000e+00, float 1.000000e+00, float 2.000000e+00, float 4.000000e+00>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr float, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr float, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr float, ptr addrspace(1) %out, i32 3

  store float 1.0, ptr addrspace(1) %out.gep.1
  store float 2.0, ptr addrspace(1) %out.gep.2
  store float 4.0, ptr addrspace(1) %out.gep.3
  store float 8.0, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_4_constants_mixed_i32_f32(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_4_constants_mixed_i32_f32(
; CHECK-NEXT:    store <4 x i32> <i32 1090519040, i32 11, i32 1073741824, i32 17>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr float, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr float, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr float, ptr addrspace(1) %out, i32 3


  store i32 11, ptr addrspace(1) %out.gep.1
  store float 2.0, ptr addrspace(1) %out.gep.2
  store i32 17, ptr addrspace(1) %out.gep.3
  store float 8.0, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_3_constants_i32(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_3_constants_i32(
; CHECK-NEXT:    store <3 x i32> <i32 1234, i32 123, i32 456>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr i32, ptr addrspace(1) %out, i32 2

  store i32 123, ptr addrspace(1) %out.gep.1
  store i32 456, ptr addrspace(1) %out.gep.2
  store i32 1234, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_2_constants_i64(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_2_constants_i64(
; CHECK-NEXT:    store <2 x i64> <i64 456, i64 123>, ptr addrspace(1) [[OUT:%.*]], align 8
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i64, ptr addrspace(1) %out, i64 1

  store i64 123, ptr addrspace(1) %out.gep.1
  store i64 456, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_4_constants_i64(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @merge_global_store_4_constants_i64(
; CHECK-NEXT:    [[OUT_GEP_2:%.*]] = getelementptr i64, ptr addrspace(1) [[OUT:%.*]], i64 2
; CHECK-NEXT:    store <2 x i64> <i64 456, i64 333>, ptr addrspace(1) [[OUT_GEP_2]], align 8
; CHECK-NEXT:    store <2 x i64> <i64 1234, i64 123>, ptr addrspace(1) [[OUT]], align 8
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i64, ptr addrspace(1) %out, i64 1
  %out.gep.2 = getelementptr i64, ptr addrspace(1) %out, i64 2
  %out.gep.3 = getelementptr i64, ptr addrspace(1) %out, i64 3

  store i64 123, ptr addrspace(1) %out.gep.1
  store i64 456, ptr addrspace(1) %out.gep.2
  store i64 333, ptr addrspace(1) %out.gep.3
  store i64 1234, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_2_adjacent_loads_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_2_adjacent_loads_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i32>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[LO1:%.*]] = extractelement <2 x i32> [[TMP1]], i32 0
; CHECK-NEXT:    [[HI2:%.*]] = extractelement <2 x i32> [[TMP1]], i32 1
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <2 x i32> poison, i32 [[LO1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <2 x i32> [[TMP2]], i32 [[HI2]], i32 1
; CHECK-NEXT:    store <2 x i32> [[TMP3]], ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %in.gep.1 = getelementptr i32, ptr addrspace(1) %in, i32 1

  %lo = load i32, ptr addrspace(1) %in
  %hi = load i32, ptr addrspace(1) %in.gep.1

  store i32 %lo, ptr addrspace(1) %out
  store i32 %hi, ptr addrspace(1) %out.gep.1
  ret void
}

define amdgpu_kernel void @merge_global_store_2_adjacent_loads_i32_nonzero_base(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_2_adjacent_loads_i32_nonzero_base(
; CHECK-NEXT:    [[IN_GEP_0:%.*]] = getelementptr i32, ptr addrspace(1) [[IN:%.*]], i32 2
; CHECK-NEXT:    [[OUT_GEP_0:%.*]] = getelementptr i32, ptr addrspace(1) [[OUT:%.*]], i32 2
; CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i32>, ptr addrspace(1) [[IN_GEP_0]], align 4
; CHECK-NEXT:    [[LO1:%.*]] = extractelement <2 x i32> [[TMP1]], i32 0
; CHECK-NEXT:    [[HI2:%.*]] = extractelement <2 x i32> [[TMP1]], i32 1
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <2 x i32> poison, i32 [[LO1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <2 x i32> [[TMP2]], i32 [[HI2]], i32 1
; CHECK-NEXT:    store <2 x i32> [[TMP3]], ptr addrspace(1) [[OUT_GEP_0]], align 4
; CHECK-NEXT:    ret void
;
  %in.gep.0 = getelementptr i32, ptr addrspace(1) %in, i32 2
  %in.gep.1 = getelementptr i32, ptr addrspace(1) %in, i32 3

  %out.gep.0 = getelementptr i32, ptr addrspace(1) %out, i32 2
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 3
  %lo = load i32, ptr addrspace(1) %in.gep.0
  %hi = load i32, ptr addrspace(1) %in.gep.1

  store i32 %lo, ptr addrspace(1) %out.gep.0
  store i32 %hi, ptr addrspace(1) %out.gep.1
  ret void
}

define amdgpu_kernel void @merge_global_store_2_adjacent_loads_shuffle_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_2_adjacent_loads_shuffle_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i32>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[LO1:%.*]] = extractelement <2 x i32> [[TMP1]], i32 0
; CHECK-NEXT:    [[HI2:%.*]] = extractelement <2 x i32> [[TMP1]], i32 1
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <2 x i32> poison, i32 [[HI2]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <2 x i32> [[TMP2]], i32 [[LO1]], i32 1
; CHECK-NEXT:    store <2 x i32> [[TMP3]], ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %in.gep.1 = getelementptr i32, ptr addrspace(1) %in, i32 1

  %lo = load i32, ptr addrspace(1) %in
  %hi = load i32, ptr addrspace(1) %in.gep.1

  store i32 %hi, ptr addrspace(1) %out
  store i32 %lo, ptr addrspace(1) %out.gep.1
  ret void
}

define amdgpu_kernel void @merge_global_store_4_adjacent_loads_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_4_adjacent_loads_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[X1:%.*]] = extractelement <4 x i32> [[TMP1]], i32 0
; CHECK-NEXT:    [[Y2:%.*]] = extractelement <4 x i32> [[TMP1]], i32 1
; CHECK-NEXT:    [[Z3:%.*]] = extractelement <4 x i32> [[TMP1]], i32 2
; CHECK-NEXT:    [[W4:%.*]] = extractelement <4 x i32> [[TMP1]], i32 3
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i32> poison, i32 [[X1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x i32> [[TMP2]], i32 [[Y2]], i32 1
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i32> [[TMP3]], i32 [[Z3]], i32 2
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x i32> [[TMP4]], i32 [[W4]], i32 3
; CHECK-NEXT:    store <4 x i32> [[TMP5]], ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr i32, ptr addrspace(1) %out, i32 3
  %in.gep.1 = getelementptr i32, ptr addrspace(1) %in, i32 1
  %in.gep.2 = getelementptr i32, ptr addrspace(1) %in, i32 2
  %in.gep.3 = getelementptr i32, ptr addrspace(1) %in, i32 3

  %x = load i32, ptr addrspace(1) %in
  %y = load i32, ptr addrspace(1) %in.gep.1
  %z = load i32, ptr addrspace(1) %in.gep.2
  %w = load i32, ptr addrspace(1) %in.gep.3

  store i32 %x, ptr addrspace(1) %out
  store i32 %y, ptr addrspace(1) %out.gep.1
  store i32 %z, ptr addrspace(1) %out.gep.2
  store i32 %w, ptr addrspace(1) %out.gep.3
  ret void
}

define amdgpu_kernel void @merge_global_store_3_adjacent_loads_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_3_adjacent_loads_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = load <3 x i32>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[X1:%.*]] = extractelement <3 x i32> [[TMP1]], i32 0
; CHECK-NEXT:    [[Y2:%.*]] = extractelement <3 x i32> [[TMP1]], i32 1
; CHECK-NEXT:    [[Z3:%.*]] = extractelement <3 x i32> [[TMP1]], i32 2
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <3 x i32> poison, i32 [[X1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <3 x i32> [[TMP2]], i32 [[Y2]], i32 1
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <3 x i32> [[TMP3]], i32 [[Z3]], i32 2
; CHECK-NEXT:    store <3 x i32> [[TMP4]], ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  %in.gep.1 = getelementptr i32, ptr addrspace(1) %in, i32 1
  %in.gep.2 = getelementptr i32, ptr addrspace(1) %in, i32 2

  %x = load i32, ptr addrspace(1) %in
  %y = load i32, ptr addrspace(1) %in.gep.1
  %z = load i32, ptr addrspace(1) %in.gep.2

  store i32 %x, ptr addrspace(1) %out
  store i32 %y, ptr addrspace(1) %out.gep.1
  store i32 %z, ptr addrspace(1) %out.gep.2
  ret void
}

define amdgpu_kernel void @merge_global_store_4_adjacent_loads_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_4_adjacent_loads_f32(
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[X1:%.*]] = extractelement <4 x float> [[TMP1]], i32 0
; CHECK-NEXT:    [[Y2:%.*]] = extractelement <4 x float> [[TMP1]], i32 1
; CHECK-NEXT:    [[Z3:%.*]] = extractelement <4 x float> [[TMP1]], i32 2
; CHECK-NEXT:    [[W4:%.*]] = extractelement <4 x float> [[TMP1]], i32 3
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x float> poison, float [[X1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x float> [[TMP2]], float [[Y2]], i32 1
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x float> [[TMP3]], float [[Z3]], i32 2
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x float> [[TMP4]], float [[W4]], i32 3
; CHECK-NEXT:    store <4 x float> [[TMP5]], ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr float, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr float, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr float, ptr addrspace(1) %out, i32 3
  %in.gep.1 = getelementptr float, ptr addrspace(1) %in, i32 1
  %in.gep.2 = getelementptr float, ptr addrspace(1) %in, i32 2
  %in.gep.3 = getelementptr float, ptr addrspace(1) %in, i32 3

  %x = load float, ptr addrspace(1) %in
  %y = load float, ptr addrspace(1) %in.gep.1
  %z = load float, ptr addrspace(1) %in.gep.2
  %w = load float, ptr addrspace(1) %in.gep.3

  store float %x, ptr addrspace(1) %out
  store float %y, ptr addrspace(1) %out.gep.1
  store float %z, ptr addrspace(1) %out.gep.2
  store float %w, ptr addrspace(1) %out.gep.3
  ret void
}

define amdgpu_kernel void @merge_global_store_4_adjacent_loads_i32_nonzero_base(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_4_adjacent_loads_i32_nonzero_base(
; CHECK-NEXT:    [[IN_GEP_0:%.*]] = getelementptr i32, ptr addrspace(1) [[IN:%.*]], i32 11
; CHECK-NEXT:    [[OUT_GEP_0:%.*]] = getelementptr i32, ptr addrspace(1) [[OUT:%.*]], i32 7
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr addrspace(1) [[IN_GEP_0]], align 4
; CHECK-NEXT:    [[X1:%.*]] = extractelement <4 x i32> [[TMP1]], i32 0
; CHECK-NEXT:    [[Y2:%.*]] = extractelement <4 x i32> [[TMP1]], i32 1
; CHECK-NEXT:    [[Z3:%.*]] = extractelement <4 x i32> [[TMP1]], i32 2
; CHECK-NEXT:    [[W4:%.*]] = extractelement <4 x i32> [[TMP1]], i32 3
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i32> poison, i32 [[X1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x i32> [[TMP2]], i32 [[Y2]], i32 1
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i32> [[TMP3]], i32 [[Z3]], i32 2
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x i32> [[TMP4]], i32 [[W4]], i32 3
; CHECK-NEXT:    store <4 x i32> [[TMP5]], ptr addrspace(1) [[OUT_GEP_0]], align 4
; CHECK-NEXT:    ret void
;
  %in.gep.0 = getelementptr i32, ptr addrspace(1) %in, i32 11
  %in.gep.1 = getelementptr i32, ptr addrspace(1) %in, i32 12
  %in.gep.2 = getelementptr i32, ptr addrspace(1) %in, i32 13
  %in.gep.3 = getelementptr i32, ptr addrspace(1) %in, i32 14
  %out.gep.0 = getelementptr i32, ptr addrspace(1) %out, i32 7
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 8
  %out.gep.2 = getelementptr i32, ptr addrspace(1) %out, i32 9
  %out.gep.3 = getelementptr i32, ptr addrspace(1) %out, i32 10

  %x = load i32, ptr addrspace(1) %in.gep.0
  %y = load i32, ptr addrspace(1) %in.gep.1
  %z = load i32, ptr addrspace(1) %in.gep.2
  %w = load i32, ptr addrspace(1) %in.gep.3

  store i32 %x, ptr addrspace(1) %out.gep.0
  store i32 %y, ptr addrspace(1) %out.gep.1
  store i32 %z, ptr addrspace(1) %out.gep.2
  store i32 %w, ptr addrspace(1) %out.gep.3
  ret void
}

define amdgpu_kernel void @merge_global_store_4_adjacent_loads_inverse_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_4_adjacent_loads_inverse_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[X1:%.*]] = extractelement <4 x i32> [[TMP1]], i32 0
; CHECK-NEXT:    [[Y2:%.*]] = extractelement <4 x i32> [[TMP1]], i32 1
; CHECK-NEXT:    [[Z3:%.*]] = extractelement <4 x i32> [[TMP1]], i32 2
; CHECK-NEXT:    [[W4:%.*]] = extractelement <4 x i32> [[TMP1]], i32 3
; CHECK-NEXT:    tail call void @llvm.amdgcn.s.barrier() #[[ATTR3:[0-9]+]]
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i32> poison, i32 [[X1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x i32> [[TMP2]], i32 [[Y2]], i32 1
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i32> [[TMP3]], i32 [[Z3]], i32 2
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x i32> [[TMP4]], i32 [[W4]], i32 3
; CHECK-NEXT:    store <4 x i32> [[TMP5]], ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr i32, ptr addrspace(1) %out, i32 3
  %in.gep.1 = getelementptr i32, ptr addrspace(1) %in, i32 1
  %in.gep.2 = getelementptr i32, ptr addrspace(1) %in, i32 2
  %in.gep.3 = getelementptr i32, ptr addrspace(1) %in, i32 3

  %x = load i32, ptr addrspace(1) %in
  %y = load i32, ptr addrspace(1) %in.gep.1
  %z = load i32, ptr addrspace(1) %in.gep.2
  %w = load i32, ptr addrspace(1) %in.gep.3

  ; Make sure the barrier doesn't stop this
  tail call void @llvm.amdgcn.s.barrier() #1

  store i32 %w, ptr addrspace(1) %out.gep.3
  store i32 %z, ptr addrspace(1) %out.gep.2
  store i32 %y, ptr addrspace(1) %out.gep.1
  store i32 %x, ptr addrspace(1) %out

  ret void
}

define amdgpu_kernel void @merge_global_store_4_adjacent_loads_shuffle_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_4_adjacent_loads_shuffle_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[X1:%.*]] = extractelement <4 x i32> [[TMP1]], i32 0
; CHECK-NEXT:    [[Y2:%.*]] = extractelement <4 x i32> [[TMP1]], i32 1
; CHECK-NEXT:    [[Z3:%.*]] = extractelement <4 x i32> [[TMP1]], i32 2
; CHECK-NEXT:    [[W4:%.*]] = extractelement <4 x i32> [[TMP1]], i32 3
; CHECK-NEXT:    tail call void @llvm.amdgcn.s.barrier() #[[ATTR3]]
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i32> poison, i32 [[W4]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x i32> [[TMP2]], i32 [[Z3]], i32 1
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i32> [[TMP3]], i32 [[Y2]], i32 2
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x i32> [[TMP4]], i32 [[X1]], i32 3
; CHECK-NEXT:    store <4 x i32> [[TMP5]], ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr i32, ptr addrspace(1) %out, i32 3
  %in.gep.1 = getelementptr i32, ptr addrspace(1) %in, i32 1
  %in.gep.2 = getelementptr i32, ptr addrspace(1) %in, i32 2
  %in.gep.3 = getelementptr i32, ptr addrspace(1) %in, i32 3

  %x = load i32, ptr addrspace(1) %in
  %y = load i32, ptr addrspace(1) %in.gep.1
  %z = load i32, ptr addrspace(1) %in.gep.2
  %w = load i32, ptr addrspace(1) %in.gep.3

  ; Make sure the barrier doesn't stop this
  tail call void @llvm.amdgcn.s.barrier() #1

  store i32 %w, ptr addrspace(1) %out
  store i32 %z, ptr addrspace(1) %out.gep.1
  store i32 %y, ptr addrspace(1) %out.gep.2
  store i32 %x, ptr addrspace(1) %out.gep.3

  ret void
}

define amdgpu_kernel void @merge_global_store_4_adjacent_loads_i8(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_4_adjacent_loads_i8(
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i8>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[X1:%.*]] = extractelement <4 x i8> [[TMP1]], i32 0
; CHECK-NEXT:    [[Y2:%.*]] = extractelement <4 x i8> [[TMP1]], i32 1
; CHECK-NEXT:    [[Z3:%.*]] = extractelement <4 x i8> [[TMP1]], i32 2
; CHECK-NEXT:    [[W4:%.*]] = extractelement <4 x i8> [[TMP1]], i32 3
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i8> poison, i8 [[X1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x i8> [[TMP2]], i8 [[Y2]], i32 1
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i8> [[TMP3]], i8 [[Z3]], i32 2
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x i8> [[TMP4]], i8 [[W4]], i32 3
; CHECK-NEXT:    store <4 x i8> [[TMP5]], ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i8, ptr addrspace(1) %out, i8 1
  %out.gep.2 = getelementptr i8, ptr addrspace(1) %out, i8 2
  %out.gep.3 = getelementptr i8, ptr addrspace(1) %out, i8 3
  %in.gep.1 = getelementptr i8, ptr addrspace(1) %in, i8 1
  %in.gep.2 = getelementptr i8, ptr addrspace(1) %in, i8 2
  %in.gep.3 = getelementptr i8, ptr addrspace(1) %in, i8 3

  %x = load i8, ptr addrspace(1) %in, align 4
  %y = load i8, ptr addrspace(1) %in.gep.1
  %z = load i8, ptr addrspace(1) %in.gep.2
  %w = load i8, ptr addrspace(1) %in.gep.3

  store i8 %x, ptr addrspace(1) %out, align 4
  store i8 %y, ptr addrspace(1) %out.gep.1
  store i8 %z, ptr addrspace(1) %out.gep.2
  store i8 %w, ptr addrspace(1) %out.gep.3
  ret void
}

define amdgpu_kernel void @merge_global_store_4_adjacent_loads_i8_natural_align(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_4_adjacent_loads_i8_natural_align(
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i8>, ptr addrspace(1) [[IN:%.*]], align 1
; CHECK-NEXT:    [[X1:%.*]] = extractelement <4 x i8> [[TMP1]], i32 0
; CHECK-NEXT:    [[Y2:%.*]] = extractelement <4 x i8> [[TMP1]], i32 1
; CHECK-NEXT:    [[Z3:%.*]] = extractelement <4 x i8> [[TMP1]], i32 2
; CHECK-NEXT:    [[W4:%.*]] = extractelement <4 x i8> [[TMP1]], i32 3
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i8> poison, i8 [[X1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x i8> [[TMP2]], i8 [[Y2]], i32 1
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i8> [[TMP3]], i8 [[Z3]], i32 2
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x i8> [[TMP4]], i8 [[W4]], i32 3
; CHECK-NEXT:    store <4 x i8> [[TMP5]], ptr addrspace(1) [[OUT:%.*]], align 1
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i8, ptr addrspace(1) %out, i8 1
  %out.gep.2 = getelementptr i8, ptr addrspace(1) %out, i8 2
  %out.gep.3 = getelementptr i8, ptr addrspace(1) %out, i8 3
  %in.gep.1 = getelementptr i8, ptr addrspace(1) %in, i8 1
  %in.gep.2 = getelementptr i8, ptr addrspace(1) %in, i8 2
  %in.gep.3 = getelementptr i8, ptr addrspace(1) %in, i8 3

  %x = load i8, ptr addrspace(1) %in
  %y = load i8, ptr addrspace(1) %in.gep.1
  %z = load i8, ptr addrspace(1) %in.gep.2
  %w = load i8, ptr addrspace(1) %in.gep.3

  store i8 %x, ptr addrspace(1) %out
  store i8 %y, ptr addrspace(1) %out.gep.1
  store i8 %z, ptr addrspace(1) %out.gep.2
  store i8 %w, ptr addrspace(1) %out.gep.3
  ret void
}

define amdgpu_kernel void @merge_global_store_4_vector_elts_loads_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
; CHECK-LABEL: @merge_global_store_4_vector_elts_loads_v4i32(
; CHECK-NEXT:    [[VEC:%.*]] = load <4 x i32>, ptr addrspace(1) [[IN:%.*]], align 16
; CHECK-NEXT:    [[X:%.*]] = extractelement <4 x i32> [[VEC]], i32 0
; CHECK-NEXT:    [[Y:%.*]] = extractelement <4 x i32> [[VEC]], i32 1
; CHECK-NEXT:    [[Z:%.*]] = extractelement <4 x i32> [[VEC]], i32 2
; CHECK-NEXT:    [[W:%.*]] = extractelement <4 x i32> [[VEC]], i32 3
; CHECK-NEXT:    [[TMP1:%.*]] = insertelement <4 x i32> poison, i32 [[X]], i32 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i32> [[TMP1]], i32 [[Y]], i32 1
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x i32> [[TMP2]], i32 [[Z]], i32 2
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i32> [[TMP3]], i32 [[W]], i32 3
; CHECK-NEXT:    store <4 x i32> [[TMP4]], ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr i32, ptr addrspace(1) %out, i32 3
  %vec = load <4 x i32>, ptr addrspace(1) %in

  %x = extractelement <4 x i32> %vec, i32 0
  %y = extractelement <4 x i32> %vec, i32 1
  %z = extractelement <4 x i32> %vec, i32 2
  %w = extractelement <4 x i32> %vec, i32 3

  store i32 %x, ptr addrspace(1) %out
  store i32 %y, ptr addrspace(1) %out.gep.1
  store i32 %z, ptr addrspace(1) %out.gep.2
  store i32 %w, ptr addrspace(1) %out.gep.3
  ret void
}

define amdgpu_kernel void @merge_local_store_2_constants_i8(ptr addrspace(3) %out) #0 {
; CHECK-LABEL: @merge_local_store_2_constants_i8(
; CHECK-NEXT:    store <2 x i8> <i8 -56, i8 123>, ptr addrspace(3) [[OUT:%.*]], align 2
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i8, ptr addrspace(3) %out, i32 1

  store i8 123, ptr addrspace(3) %out.gep.1
  store i8 456, ptr addrspace(3) %out, align 2
  ret void
}

define amdgpu_kernel void @merge_local_store_2_constants_i32(ptr addrspace(3) %out) #0 {
; CHECK-LABEL: @merge_local_store_2_constants_i32(
; CHECK-NEXT:    store <2 x i32> <i32 456, i32 123>, ptr addrspace(3) [[OUT:%.*]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(3) %out, i32 1

  store i32 123, ptr addrspace(3) %out.gep.1
  store i32 456, ptr addrspace(3) %out
  ret void
}

define amdgpu_kernel void @merge_local_store_2_constants_i32_align_2(ptr addrspace(3) %out) #0 {
; CHECK-LABEL: @merge_local_store_2_constants_i32_align_2(
; CHECK-NEXT:    [[OUT_GEP_1:%.*]] = getelementptr i32, ptr addrspace(3) [[OUT:%.*]], i32 1
; CHECK-NEXT:    store i32 123, ptr addrspace(3) [[OUT_GEP_1]], align 2
; CHECK-NEXT:    store i32 456, ptr addrspace(3) [[OUT]], align 2
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(3) %out, i32 1

  store i32 123, ptr addrspace(3) %out.gep.1, align 2
  store i32 456, ptr addrspace(3) %out, align 2
  ret void
}

define amdgpu_kernel void @merge_local_store_4_constants_i32(ptr addrspace(3) %out) #0 {
; CHECK-LABEL: @merge_local_store_4_constants_i32(
; CHECK-NEXT:    [[OUT_GEP_2:%.*]] = getelementptr i32, ptr addrspace(3) [[OUT:%.*]], i32 2
; CHECK-NEXT:    store <2 x i32> <i32 456, i32 333>, ptr addrspace(3) [[OUT_GEP_2]], align 4
; CHECK-NEXT:    store <2 x i32> <i32 1234, i32 123>, ptr addrspace(3) [[OUT]], align 4
; CHECK-NEXT:    ret void
;
  %out.gep.1 = getelementptr i32, ptr addrspace(3) %out, i32 1
  %out.gep.2 = getelementptr i32, ptr addrspace(3) %out, i32 2
  %out.gep.3 = getelementptr i32, ptr addrspace(3) %out, i32 3

  store i32 123, ptr addrspace(3) %out.gep.1
  store i32 456, ptr addrspace(3) %out.gep.2
  store i32 333, ptr addrspace(3) %out.gep.3
  store i32 1234, ptr addrspace(3) %out
  ret void
}

define amdgpu_kernel void @merge_global_store_5_constants_i32(ptr addrspace(1) %out) {
; CHECK-LABEL: @merge_global_store_5_constants_i32(
; CHECK-NEXT:    store <4 x i32> <i32 9, i32 12, i32 16, i32 -12>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    [[IDX4:%.*]] = getelementptr inbounds i32, ptr addrspace(1) [[OUT]], i64 4
; CHECK-NEXT:    store i32 11, ptr addrspace(1) [[IDX4]], align 4
; CHECK-NEXT:    ret void
;
  store i32 9, ptr addrspace(1) %out, align 4
  %idx1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 12, ptr addrspace(1) %idx1, align 4
  %idx2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 16, ptr addrspace(1) %idx2, align 4
  %idx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 3
  store i32 -12, ptr addrspace(1) %idx3, align 4
  %idx4 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 4
  store i32 11, ptr addrspace(1) %idx4, align 4
  ret void
}

define amdgpu_kernel void @merge_global_store_6_constants_i32(ptr addrspace(1) %out) {
; CHECK-LABEL: @merge_global_store_6_constants_i32(
; CHECK-NEXT:    store <4 x i32> <i32 13, i32 15, i32 62, i32 63>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    [[IDX4:%.*]] = getelementptr inbounds i32, ptr addrspace(1) [[OUT]], i64 4
; CHECK-NEXT:    store <2 x i32> <i32 11, i32 123>, ptr addrspace(1) [[IDX4]], align 4
; CHECK-NEXT:    ret void
;
  store i32 13, ptr addrspace(1) %out, align 4
  %idx1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 15, ptr addrspace(1) %idx1, align 4
  %idx2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 62, ptr addrspace(1) %idx2, align 4
  %idx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 3
  store i32 63, ptr addrspace(1) %idx3, align 4
  %idx4 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 4
  store i32 11, ptr addrspace(1) %idx4, align 4
  %idx5 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 5
  store i32 123, ptr addrspace(1) %idx5, align 4
  ret void
}

define amdgpu_kernel void @merge_global_store_7_constants_i32(ptr addrspace(1) %out) {
; CHECK-LABEL: @merge_global_store_7_constants_i32(
; CHECK-NEXT:    store <4 x i32> <i32 34, i32 999, i32 65, i32 33>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    [[IDX4:%.*]] = getelementptr inbounds i32, ptr addrspace(1) [[OUT]], i64 4
; CHECK-NEXT:    store <3 x i32> <i32 98, i32 91, i32 212>, ptr addrspace(1) [[IDX4]], align 4
; CHECK-NEXT:    ret void
;
  store i32 34, ptr addrspace(1) %out, align 4
  %idx1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 999, ptr addrspace(1) %idx1, align 4
  %idx2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 65, ptr addrspace(1) %idx2, align 4
  %idx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 3
  store i32 33, ptr addrspace(1) %idx3, align 4
  %idx4 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 4
  store i32 98, ptr addrspace(1) %idx4, align 4
  %idx5 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 5
  store i32 91, ptr addrspace(1) %idx5, align 4
  %idx6 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 6
  store i32 212, ptr addrspace(1) %idx6, align 4
  ret void
}

define amdgpu_kernel void @merge_global_store_8_constants_i32(ptr addrspace(1) %out) {
; CHECK-LABEL: @merge_global_store_8_constants_i32(
; CHECK-NEXT:    store <4 x i32> <i32 34, i32 999, i32 65, i32 33>, ptr addrspace(1) [[OUT:%.*]], align 4
; CHECK-NEXT:    [[IDX4:%.*]] = getelementptr inbounds i32, ptr addrspace(1) [[OUT]], i64 4
; CHECK-NEXT:    store <4 x i32> <i32 98, i32 91, i32 212, i32 999>, ptr addrspace(1) [[IDX4]], align 4
; CHECK-NEXT:    ret void
;
  store i32 34, ptr addrspace(1) %out, align 4
  %idx1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 999, ptr addrspace(1) %idx1, align 4
  %idx2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 65, ptr addrspace(1) %idx2, align 4
  %idx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 3
  store i32 33, ptr addrspace(1) %idx3, align 4
  %idx4 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 4
  store i32 98, ptr addrspace(1) %idx4, align 4
  %idx5 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 5
  store i32 91, ptr addrspace(1) %idx5, align 4
  %idx6 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 6
  store i32 212, ptr addrspace(1) %idx6, align 4
  %idx7 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 7
  store i32 999, ptr addrspace(1) %idx7, align 4
  ret void
}

define amdgpu_kernel void @copy_v3i32_align4(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
; CHECK-LABEL: @copy_v3i32_align4(
; CHECK-NEXT:    [[VEC:%.*]] = load <3 x i32>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    store <3 x i32> [[VEC]], ptr addrspace(1) [[OUT:%.*]], align 16
; CHECK-NEXT:    ret void
;
  %vec = load <3 x i32>, ptr addrspace(1) %in, align 4
  store <3 x i32> %vec, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @copy_v3i64_align4(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
; CHECK-LABEL: @copy_v3i64_align4(
; CHECK-NEXT:    [[VEC:%.*]] = load <3 x i64>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    store <3 x i64> [[VEC]], ptr addrspace(1) [[OUT:%.*]], align 32
; CHECK-NEXT:    ret void
;
  %vec = load <3 x i64>, ptr addrspace(1) %in, align 4
  store <3 x i64> %vec, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @copy_v3f32_align4(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
; CHECK-LABEL: @copy_v3f32_align4(
; CHECK-NEXT:    [[VEC:%.*]] = load <3 x float>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[FADD:%.*]] = fadd <3 x float> [[VEC]], <float 1.000000e+00, float 2.000000e+00, float 4.000000e+00>
; CHECK-NEXT:    store <3 x float> [[FADD]], ptr addrspace(1) [[OUT:%.*]], align 16
; CHECK-NEXT:    ret void
;
  %vec = load <3 x float>, ptr addrspace(1) %in, align 4
  %fadd = fadd <3 x float> %vec, <float 1.0, float 2.0, float 4.0>
  store <3 x float> %fadd, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @copy_v3f64_align4(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
; CHECK-LABEL: @copy_v3f64_align4(
; CHECK-NEXT:    [[VEC:%.*]] = load <3 x double>, ptr addrspace(1) [[IN:%.*]], align 4
; CHECK-NEXT:    [[FADD:%.*]] = fadd <3 x double> [[VEC]], <double 1.000000e+00, double 2.000000e+00, double 4.000000e+00>
; CHECK-NEXT:    store <3 x double> [[FADD]], ptr addrspace(1) [[OUT:%.*]], align 32
; CHECK-NEXT:    ret void
;
  %vec = load <3 x double>, ptr addrspace(1) %in, align 4
  %fadd = fadd <3 x double> %vec, <double 1.0, double 2.0, double 4.0>
  store <3 x double> %fadd, ptr addrspace(1) %out
  ret void
}

; Verify that we no longer hit asserts for this test case. No change expected.
define amdgpu_kernel void @copy_vec_of_ptrs(ptr addrspace(1) %out,
; CHECK-LABEL: @copy_vec_of_ptrs(
; CHECK-NEXT:    [[IN_GEP_1:%.*]] = getelementptr <2 x ptr>, ptr addrspace(1) [[IN:%.*]], i32 1
; CHECK-NEXT:    [[VEC1:%.*]] = load <2 x ptr>, ptr addrspace(1) [[IN_GEP_1]], align 16
; CHECK-NEXT:    [[VEC2:%.*]] = load <2 x ptr>, ptr addrspace(1) [[IN]], align 4
; CHECK-NEXT:    [[OUT_GEP_1:%.*]] = getelementptr <2 x ptr>, ptr addrspace(1) [[OUT:%.*]], i32 1
; CHECK-NEXT:    store <2 x ptr> [[VEC1]], ptr addrspace(1) [[OUT_GEP_1]], align 16
; CHECK-NEXT:    store <2 x ptr> [[VEC2]], ptr addrspace(1) [[OUT]], align 4
; CHECK-NEXT:    ret void
;
  ptr addrspace(1) %in ) #0 {
  %in.gep.1 = getelementptr <2 x ptr>, ptr addrspace(1) %in, i32 1
  %vec1 = load <2 x ptr>, ptr addrspace(1) %in.gep.1
  %vec2 = load <2 x ptr>, ptr addrspace(1) %in, align 4

  %out.gep.1 = getelementptr <2 x ptr>, ptr addrspace(1) %out, i32 1
  store <2 x ptr> %vec1, ptr addrspace(1) %out.gep.1
  store <2 x ptr> %vec2, ptr addrspace(1) %out, align 4
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
