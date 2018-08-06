; RUN: opt -mtriple=amdgcn-- -S -separate-const-offset-from-gep -reassociate-geps-verify-no-dead-code -gvn < %s | FileCheck -check-prefix=IR %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

@array = internal addrspace(4) constant [4096 x [32 x float]] zeroinitializer, align 4

; IR-LABEL: @sum_of_array(
; IR: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [4096 x [32 x float]], [4096 x [32 x float]] addrspace(4)* @array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; IR: getelementptr inbounds float, float addrspace(4)* [[BASE_PTR]], i64 1
; IR: getelementptr inbounds float, float addrspace(4)* [[BASE_PTR]], i64 32
; IR: getelementptr inbounds float, float addrspace(4)* [[BASE_PTR]], i64 33
define amdgpu_kernel void @sum_of_array(i32 %x, i32 %y, float addrspace(1)* nocapture %output) {
  %tmp = sext i32 %y to i64
  %tmp1 = sext i32 %x to i64
  %tmp2 = getelementptr inbounds [4096 x [32 x float]], [4096 x [32 x float]] addrspace(4)* @array, i64 0, i64 %tmp1, i64 %tmp
  %tmp4 = load float, float addrspace(4)* %tmp2, align 4
  %tmp5 = fadd float %tmp4, 0.000000e+00
  %tmp6 = add i32 %y, 1
  %tmp7 = sext i32 %tmp6 to i64
  %tmp8 = getelementptr inbounds [4096 x [32 x float]], [4096 x [32 x float]] addrspace(4)* @array, i64 0, i64 %tmp1, i64 %tmp7
  %tmp10 = load float, float addrspace(4)* %tmp8, align 4
  %tmp11 = fadd float %tmp5, %tmp10
  %tmp12 = add i32 %x, 1
  %tmp13 = sext i32 %tmp12 to i64
  %tmp14 = getelementptr inbounds [4096 x [32 x float]], [4096 x [32 x float]] addrspace(4)* @array, i64 0, i64 %tmp13, i64 %tmp
  %tmp16 = load float, float addrspace(4)* %tmp14, align 4
  %tmp17 = fadd float %tmp11, %tmp16
  %tmp18 = getelementptr inbounds [4096 x [32 x float]], [4096 x [32 x float]] addrspace(4)* @array, i64 0, i64 %tmp13, i64 %tmp7
  %tmp20 = load float, float addrspace(4)* %tmp18, align 4
  %tmp21 = fadd float %tmp17, %tmp20
  store float %tmp21, float addrspace(1)* %output, align 4
  ret void
}

@array2 = internal addrspace(4) constant [4096 x [4 x float]] zeroinitializer, align 4

; Some of the indices go over the maximum mubuf offset, so don't split them.

; IR-LABEL: @sum_of_array_over_max_mubuf_offset(
; IR: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [4096 x [4 x float]], [4096 x [4 x float]] addrspace(4)* @array2, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; IR: getelementptr inbounds float, float addrspace(4)* [[BASE_PTR]], i64 255
; IR: add i32 %x, 256
; IR: getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(4)* @array2, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; IR: getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(4)* @array2, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
define amdgpu_kernel void @sum_of_array_over_max_mubuf_offset(i32 %x, i32 %y, float addrspace(1)* nocapture %output) {
  %tmp = sext i32 %y to i64
  %tmp1 = sext i32 %x to i64
  %tmp2 = getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(4)* @array2, i64 0, i64 %tmp1, i64 %tmp
  %tmp4 = load float, float addrspace(4)* %tmp2, align 4
  %tmp5 = fadd float %tmp4, 0.000000e+00
  %tmp6 = add i32 %y, 255
  %tmp7 = sext i32 %tmp6 to i64
  %tmp8 = getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(4)* @array2, i64 0, i64 %tmp1, i64 %tmp7
  %tmp10 = load float, float addrspace(4)* %tmp8, align 4
  %tmp11 = fadd float %tmp5, %tmp10
  %tmp12 = add i32 %x, 256
  %tmp13 = sext i32 %tmp12 to i64
  %tmp14 = getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(4)* @array2, i64 0, i64 %tmp13, i64 %tmp
  %tmp16 = load float, float addrspace(4)* %tmp14, align 4
  %tmp17 = fadd float %tmp11, %tmp16
  %tmp18 = getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(4)* @array2, i64 0, i64 %tmp13, i64 %tmp7
  %tmp20 = load float, float addrspace(4)* %tmp18, align 4
  %tmp21 = fadd float %tmp17, %tmp20
  store float %tmp21, float addrspace(1)* %output, align 4
  ret void
}


@lds_array = internal addrspace(3) global [4096 x [4 x float]] undef, align 4

; DS instructions have a larger immediate offset, so make sure these are OK.
; IR-LABEL: @sum_of_lds_array_over_max_mubuf_offset(
; IR: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [4096 x [4 x float]], [4096 x [4 x float]] addrspace(3)* @lds_array, i32 0, i32 %{{[a-zA-Z0-9]+}}, i32 %{{[a-zA-Z0-9]+}}
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i32 255
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i32 16128
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i32 16383
define amdgpu_kernel void @sum_of_lds_array_over_max_mubuf_offset(i32 %x, i32 %y, float addrspace(1)* nocapture %output) {
  %tmp2 = getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(3)* @lds_array, i32 0, i32 %x, i32 %y
  %tmp4 = load float, float addrspace(3)* %tmp2, align 4
  %tmp5 = fadd float %tmp4, 0.000000e+00
  %tmp6 = add i32 %y, 255
  %tmp8 = getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(3)* @lds_array, i32 0, i32 %x, i32 %tmp6
  %tmp10 = load float, float addrspace(3)* %tmp8, align 4
  %tmp11 = fadd float %tmp5, %tmp10
  %tmp12 = add i32 %x, 4032
  %tmp14 = getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(3)* @lds_array, i32 0, i32 %tmp12, i32 %y
  %tmp16 = load float, float addrspace(3)* %tmp14, align 4
  %tmp17 = fadd float %tmp11, %tmp16
  %tmp18 = getelementptr inbounds [4096 x [4 x float]], [4096 x [4 x float]] addrspace(3)* @lds_array, i32 0, i32 %tmp12, i32 %tmp6
  %tmp20 = load float, float addrspace(3)* %tmp18, align 4
  %tmp21 = fadd float %tmp17, %tmp20
  store float %tmp21, float addrspace(1)* %output, align 4
  ret void
}

; IR-LABEL: @keep_metadata(
; IR: getelementptr {{.*}} !amdgpu.uniform
; IR: getelementptr {{.*}} !amdgpu.uniform
; IR: getelementptr {{.*}} !amdgpu.uniform
define amdgpu_ps <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> @keep_metadata([0 x <4 x i32>] addrspace(4)* inreg noalias dereferenceable(18446744073709551615), [0 x <8 x i32>] addrspace(4)* inreg noalias dereferenceable(18446744073709551615), [0 x <4 x i32>] addrspace(4)* inreg noalias dereferenceable(18446744073709551615), [0 x <8 x i32>] addrspace(4)* inreg noalias dereferenceable(18446744073709551615), float inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, i32, i32, float, i32) #5 {
main_body:
  %22 = call nsz float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %5) #8
  %23 = bitcast float %22 to i32
  %24 = shl i32 %23, 1
  %25 = getelementptr [0 x <8 x i32>], [0 x <8 x i32>] addrspace(4)* %1, i32 0, i32 %24, !amdgpu.uniform !0
  %26 = load <8 x i32>, <8 x i32> addrspace(4)* %25, align 32, !invariant.load !0
  %27 = shl i32 %23, 2
  %28 = or i32 %27, 3
  %29 = bitcast [0 x <8 x i32>] addrspace(4)* %1 to [0 x <4 x i32>] addrspace(4)*
  %30 = getelementptr [0 x <4 x i32>], [0 x <4 x i32>] addrspace(4)* %29, i32 0, i32 %28, !amdgpu.uniform !0
  %31 = load <4 x i32>, <4 x i32> addrspace(4)* %30, align 16, !invariant.load !0
  %32 = call nsz <4 x float> @llvm.amdgcn.image.sample.v4f32.v2f32.v8i32(<2 x float> zeroinitializer, <8 x i32> %26, <4 x i32> %31, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false) #8
  %33 = extractelement <4 x float> %32, i32 0
  %34 = extractelement <4 x float> %32, i32 1
  %35 = extractelement <4 x float> %32, i32 2
  %36 = extractelement <4 x float> %32, i32 3
  %37 = bitcast float %4 to i32
  %38 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> undef, i32 %37, 4
  %39 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %38, float %33, 5
  %40 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %39, float %34, 6
  %41 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %40, float %35, 7
  %42 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %41, float %36, 8
  %43 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %42, float %20, 19
  ret <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %43
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) #6

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #7


!0 = !{}

attributes #5 = { "InitialPSInputAddr"="45175" }
attributes #6 = { nounwind readnone speculatable }
attributes #7 = { nounwind readonly }
attributes #8 = { nounwind readnone }
