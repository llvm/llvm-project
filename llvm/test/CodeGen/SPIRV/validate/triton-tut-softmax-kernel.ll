; This is an excerpt from the tutorial of the Triton language converted into
; LLVM IR via the Triton XPU backend and cleaned of irrelevant details.
; The only pass criterion is that spirv-val considers output valid.

; Ths particular case is related to translation of <1 x Ty> vectors.

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

define spir_kernel void @softmax_kernel(ptr addrspace(1) nocapture writeonly %0, ptr addrspace(1) nocapture readonly %1, i32 %2, i32 %3, i32 %4, i32 %5, ptr addrspace(3) nocapture %6) {
  %8 = tail call spir_func i64 @_Z12get_group_idj(i32 0)
  %9 = trunc i64 %8 to i32
  %10 = tail call spir_func i64 @_Z14get_num_groupsj(i32 0)
  %11 = trunc i64 %10 to i32
  %12 = tail call spir_func i64 @_Z12get_local_idj(i32 0)
  %13 = trunc i64 %12 to i32
  %14 = and i32 %13, 255
  %15 = or disjoint i32 %14, 256
  %16 = or disjoint i32 %14, 512
  %17 = or disjoint i32 %14, 768
  %18 = icmp slt i32 %14, %5
  %19 = icmp slt i32 %15, %5
  %20 = icmp slt i32 %16, %5
  %21 = icmp slt i32 %17, %5
  %22 = icmp sgt i32 %4, %9
  br i1 %22, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %7
  %23 = lshr i64 %12, 5
  %24 = and i32 %13, 31
  %25 = zext nneg i32 %15 to i64
  %26 = zext nneg i32 %16 to i64
  %27 = zext nneg i32 %17 to i64
  %28 = and i64 %12, 255
  %29 = and i64 %23, 7
  %30 = icmp eq i32 %24, 0
  %31 = getelementptr float, ptr addrspace(3) %6, i64 %29
  %32 = icmp slt i32 %13, 8
  %sext = shl i64 %12, 32
  %33 = ashr exact i64 %sext, 30
  %34 = getelementptr i8, ptr addrspace(3) %6, i64 %33
  %35 = and i32 %13, 7
  %36 = icmp eq i32 %35, 0
  %37 = and i1 %32, %36
  br label %38

38:                                               ; preds = %.lr.ph, %123
  %39 = phi i32 [ %9, %.lr.ph ], [ %124, %123 ]
  %40 = mul i32 %39, %2
  %41 = sext i32 %40 to i64
  %42 = getelementptr float, ptr addrspace(1) %1, i64 %41
  %43 = getelementptr float, ptr addrspace(1) %42, i64 %25
  %44 = getelementptr float, ptr addrspace(1) %42, i64 %26
  %45 = getelementptr float, ptr addrspace(1) %42, i64 %27
  br i1 %18, label %46, label %49

46:                                               ; preds = %38
  %47 = getelementptr float, ptr addrspace(1) %42, i64 %28
  %48 = load <1 x float>, ptr addrspace(1) %47, align 4
  br label %49

49:                                               ; preds = %46, %38
  %50 = phi <1 x float> [ %48, %46 ], [ splat (float 0xFFF0000000000000), %38 ]
  %51 = extractelement <1 x float> %50, i64 0
  br i1 %19, label %52, label %54

52:                                               ; preds = %49
  %53 = load <1 x float>, ptr addrspace(1) %43, align 4
  br label %54

54:                                               ; preds = %52, %49
  %55 = phi <1 x float> [ %53, %52 ], [ splat (float 0xFFF0000000000000), %49 ]
  %56 = extractelement <1 x float> %55, i64 0
  br i1 %20, label %57, label %59

57:                                               ; preds = %54
  %58 = load <1 x float>, ptr addrspace(1) %44, align 4
  br label %59

59:                                               ; preds = %57, %54
  %60 = phi <1 x float> [ %58, %57 ], [ splat (float 0xFFF0000000000000), %54 ]
  %61 = extractelement <1 x float> %60, i64 0
  br i1 %21, label %62, label %64

62:                                               ; preds = %59
  %63 = load <1 x float>, ptr addrspace(1) %45, align 4
  br label %64

64:                                               ; preds = %62, %59
  %65 = phi <1 x float> [ %63, %62 ], [ splat (float 0xFFF0000000000000), %59 ]
  %66 = extractelement <1 x float> %65, i64 0
  tail call spir_func void @_Z7barrierj(i32 1)
  %67 = tail call float @llvm.maxnum.f32(float %51, float %56)
  %68 = tail call float @llvm.maxnum.f32(float %67, float %61)
  %69 = tail call float @llvm.maxnum.f32(float %68, float %66)
  %70 = tail call spir_func float @_Z27__spirv_GroupNonUniformFMaxiif(i32 3, i32 0, float %69)
  br i1 %30, label %71, label %72

71:                                               ; preds = %64
  store float %70, ptr addrspace(3) %31, align 4
  br label %72

72:                                               ; preds = %71, %64
  tail call spir_func void @_Z7barrierj(i32 1)
  br i1 %32, label %74, label %.thread1

.thread1:                                         ; preds = %72
  %73 = tail call spir_func float @_Z27__spirv_GroupNonUniformFMaxiifj(i32 3, i32 3, float poison, i32 8)
  br label %78

74:                                               ; preds = %72
  %75 = load float, ptr addrspace(3) %34, align 4
  %76 = tail call spir_func float @_Z27__spirv_GroupNonUniformFMaxiifj(i32 3, i32 3, float %75, i32 8)
  br i1 %37, label %77, label %78

77:                                               ; preds = %74
  store float %76, ptr addrspace(3) %34, align 4
  br label %78

78:                                               ; preds = %.thread1, %77, %74
  tail call spir_func void @_Z7barrierj(i32 1)
  %79 = load float, ptr addrspace(3) %6, align 4
  %80 = fsub float %51, %79
  %81 = fsub float %56, %79
  %82 = fsub float %61, %79
  %83 = fsub float %66, %79
  %84 = fmul float %80, 0x3FF7154760000000
  %85 = tail call float @llvm.exp2.f32(float %84)
  %86 = fmul float %81, 0x3FF7154760000000
  %87 = tail call float @llvm.exp2.f32(float %86)
  %88 = fmul float %82, 0x3FF7154760000000
  %89 = tail call float @llvm.exp2.f32(float %88)
  %90 = fmul float %83, 0x3FF7154760000000
  %91 = tail call float @llvm.exp2.f32(float %90)
  tail call spir_func void @_Z7barrierj(i32 1)
  %92 = fadd float %85, %87
  %93 = fadd float %89, %92
  %94 = fadd float %91, %93
  %95 = tail call spir_func float @_Z27__spirv_GroupNonUniformFAddiif(i32 3, i32 0, float %94)
  br i1 %30, label %96, label %97

96:                                               ; preds = %78
  store float %95, ptr addrspace(3) %31, align 4
  br label %97

97:                                               ; preds = %96, %78
  tail call spir_func void @_Z7barrierj(i32 1)
  br i1 %32, label %99, label %.thread

.thread:                                          ; preds = %97
  %98 = tail call spir_func float @_Z27__spirv_GroupNonUniformFAddiifj(i32 3, i32 3, float poison, i32 8)
  br label %103

99:                                               ; preds = %97
  %100 = load float, ptr addrspace(3) %34, align 4
  %101 = tail call spir_func float @_Z27__spirv_GroupNonUniformFAddiifj(i32 3, i32 3, float %100, i32 8)
  br i1 %37, label %102, label %103

102:                                              ; preds = %99
  store float %101, ptr addrspace(3) %34, align 4
  br label %103

103:                                              ; preds = %.thread, %102, %99
  tail call spir_func void @_Z7barrierj(i32 1)
  %104 = load float, ptr addrspace(3) %6, align 4
  %105 = fdiv float %87, %104
  %106 = fdiv float %89, %104
  %107 = fdiv float %91, %104
  %108 = mul i32 %39, %3
  %109 = sext i32 %108 to i64
  %110 = getelementptr float, ptr addrspace(1) %0, i64 %109
  %111 = getelementptr float, ptr addrspace(1) %110, i64 %25
  %112 = getelementptr float, ptr addrspace(1) %110, i64 %26
  %113 = getelementptr float, ptr addrspace(1) %110, i64 %27
  br i1 %18, label %114, label %117

114:                                              ; preds = %103
  %115 = fdiv float %85, %104
  %116 = getelementptr float, ptr addrspace(1) %110, i64 %28
  store float %115, ptr addrspace(1) %116, align 4
  br label %117

117:                                              ; preds = %114, %103
  br i1 %19, label %118, label %119

118:                                              ; preds = %117
  store float %105, ptr addrspace(1) %111, align 4
  br label %119

119:                                              ; preds = %118, %117
  br i1 %20, label %120, label %121

120:                                              ; preds = %119
  store float %106, ptr addrspace(1) %112, align 4
  br label %121

121:                                              ; preds = %120, %119
  br i1 %21, label %122, label %123

122:                                              ; preds = %121
  store float %107, ptr addrspace(1) %113, align 4
  br label %123

123:                                              ; preds = %122, %121
  %124 = add i32 %39, %11
  %125 = icmp slt i32 %124, %4
  br i1 %125, label %38, label %._crit_edge

._crit_edge:                                      ; preds = %123, %7
  ret void
}

declare float @llvm.maxnum.f32(float, float)
declare spir_func float @_Z27__spirv_GroupNonUniformFAddiifj(i32, i32, float, i32)
declare spir_func float @_Z27__spirv_GroupNonUniformFAddiif(i32, i32, float)
declare spir_func float @_Z27__spirv_GroupNonUniformFMaxiifj(i32, i32, float, i32)
declare spir_func float @_Z27__spirv_GroupNonUniformFMaxiif(i32, i32, float)
declare spir_func void @_Z7barrierj(i32)
declare spir_func i64 @_Z12get_local_idj(i32)
declare spir_func i64 @_Z14get_num_groupsj(i32)
declare spir_func i64 @_Z12get_group_idj(i32)
declare float @llvm.exp2.f32(float)
