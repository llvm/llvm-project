; RUN: opt -S -instcombine %s | FileCheck %s

; --------------------------------------------------------------------
; llvm.amdgcn.buffer.load
; --------------------------------------------------------------------

; CHECK-LABEL: @buffer_load_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @buffer_load_f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  ret float %data
}

; CHECK-LABEL: @buffer_load_v1f32(
; CHECK-NEXT: %data = call <1 x float> @llvm.amdgcn.buffer.load.v1f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret <1 x float> %data
define amdgpu_ps <1 x float> @buffer_load_v1f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <1 x float> @llvm.amdgcn.buffer.load.v1f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  ret <1 x float> %data
}

; CHECK-LABEL: @buffer_load_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  ret <2 x float> %data
}

; CHECK-LABEL: @buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret <4 x float> %data
define amdgpu_ps <4 x float> @buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  ret <4 x float> %data
}

; CHECK-LABEL: @extract_elt0_buffer_load_v2f32(
; CHECK: %data = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_buffer_load_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt1 = extractelement <2 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_buffer_load_v4f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_buffer_load_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt1 = extractelement <4 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt1 = extractelement <4 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt3_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 3
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt3_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt1 = extractelement <4 x float> %data, i32 3
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_buffer_load_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt2_elt3_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt2_elt3_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_elt3_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt1_elt2_elt3_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt2_elt3_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt2_elt3_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
  ret <3 x float> %shuf
}

; FIXME: Not handled even though only 2 elts used
; CHECK-LABEL: @extract_elt0_elt1_buffer_load_v4f32_2(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %elt0 = extractelement <4 x float> %data, i32 0
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 1
; CHECK-NEXT: %ins0 = insertvalue { float, float } undef, float %elt0, 0
; CHECK-NEXT: %ins1 = insertvalue { float, float } %ins0, float %elt1, 1
; CHECK-NEXT: ret { float, float } %ins1
define amdgpu_ps { float, float } @extract_elt0_elt1_buffer_load_v4f32_2(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  %elt1 = extractelement <4 x float> %data, i32 1
  %ins0 = insertvalue { float, float } undef, float %elt0, 0
  %ins1 = insertvalue { float, float } %ins0, float %elt1, 1
  ret { float, float } %ins1
}

; CHECK-LABEL: @extract_elt0_buffer_load_v3f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt0 = extractelement <3 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_buffer_load_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt1 = extractelement <3 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_buffer_load_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %elt1 = extractelement <3 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %elt1 = extractelement <3 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_buffer_load_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_buffer_load_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @preserve_metadata_extract_elt0_buffer_load_v2f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false), !fpmath !0
; CHECK-NEXT: ret float %data
define amdgpu_ps float @preserve_metadata_extract_elt0_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false), !fpmath !0
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1) #1
declare <1 x float> @llvm.amdgcn.buffer.load.v1f32(<4 x i32>, i32, i32, i1, i1) #1
declare <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32>, i32, i32, i1, i1) #1
declare <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32>, i32, i32, i1, i1) #1
declare <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32>, i32, i32, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.buffer.load.format
; --------------------------------------------------------------------

; CHECK-LABEL: @buffer_load_format_v1f32(
; CHECK-NEXT: %data = call <1 x float> @llvm.amdgcn.buffer.load.format.v1f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 true)
; CHECK-NEXT: ret <1 x float> %data
define amdgpu_ps <1 x float> @buffer_load_format_v1f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <1 x float> @llvm.amdgcn.buffer.load.format.v1f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 true)
  ret <1 x float> %data
}

; CHECK-LABEL: @extract_elt0_buffer_load_format_v2f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 true, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_buffer_load_format_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <2 x float> @llvm.amdgcn.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 true, i1 false)
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_elt1_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt0_elt1_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <3 x float> @llvm.amdgcn.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt0_elt1_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) #0 {
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; The initial insertion point is at the extractelement
; CHECK-LABEL: @extract01_bitcast_buffer_load_format_v4f32(
; CHECK-NEXT: %tmp = call <2 x float> @llvm.amdgcn.buffer.load.format.v2f32(<4 x i32> undef, i32 %arg, i32 16, i1 false, i1 false)
; CHECK-NEXT: %1 = shufflevector <2 x float> %tmp, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; CHECK-NEXT: %tmp1 = bitcast <4 x float> %1 to <2 x double>
; CHECK-NEXT: %tmp2 = extractelement <2 x double> %tmp1, i32 0
; CHECK-NEXT: ret double %tmp2
define double @extract01_bitcast_buffer_load_format_v4f32(i32 %arg) #0 {
  %tmp = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> undef, i32 %arg, i32 16, i1 false, i1 false) #3
  %tmp1 = bitcast <4 x float> %tmp to <2 x double>
  %tmp2 = extractelement <2 x double> %tmp1, i32 0
  ret double %tmp2
}

; CHECK-LABEL: @extract0_bitcast_buffer_load_format_v4f32(
; CHECK-NEXT: %tmp = call float @llvm.amdgcn.buffer.load.format.f32(<4 x i32> undef, i32 %arg, i32 16, i1 false, i1 false)
; CHECK-NEXT: %tmp2 = bitcast float %tmp to i32
; CHECK-NEXT: ret i32 %tmp2
define i32 @extract0_bitcast_buffer_load_format_v4f32(i32 %arg) #0 {
  %tmp = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> undef, i32 %arg, i32 16, i1 false, i1 false) #3
  %tmp1 = bitcast <4 x float> %tmp to <4 x i32>
  %tmp2 = extractelement <4 x i32> %tmp1, i32 0
  ret i32 %tmp2
}

; CHECK-LABEL: @extract_lo16_0_bitcast_buffer_load_format_v4f32(
; CHECK-NEXT: %tmp = call float @llvm.amdgcn.buffer.load.format.f32(<4 x i32> undef, i32 %arg, i32 16, i1 false, i1 false)
; CHECK-NEXT: %1 = bitcast float %tmp to i32
; CHECK-NEXT: %tmp2 = trunc i32 %1 to i16
; CHECK-NEXT: ret i16 %tmp2
define i16 @extract_lo16_0_bitcast_buffer_load_format_v4f32(i32 %arg) #0 {
  %tmp = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> undef, i32 %arg, i32 16, i1 false, i1 false) #3
  %tmp1 = bitcast <4 x float> %tmp to <8 x i16>
  %tmp2 = extractelement <8 x i16> %tmp1, i32 0
  ret i16 %tmp2
}

declare float @llvm.amdgcn.buffer.load.format.f32(<4 x i32>, i32, i32, i1, i1) #1
declare <1 x float> @llvm.amdgcn.buffer.load.format.v1f32(<4 x i32>, i32, i32, i1, i1) #1
declare <2 x float> @llvm.amdgcn.buffer.load.format.v2f32(<4 x i32>, i32, i32, i1, i1) #1
declare <3 x float> @llvm.amdgcn.buffer.load.format.v3f32(<4 x i32>, i32, i32, i1, i1) #1
declare <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32>, i32, i32, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.raw.buffer.load
; --------------------------------------------------------------------

; CHECK-LABEL: @raw_buffer_load_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @raw_buffer_load_f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  ret float %data
}

; CHECK-LABEL: @raw_buffer_load_v1f32(
; CHECK-NEXT: %data = call <1 x float> @llvm.amdgcn.raw.buffer.load.v1f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <1 x float> %data
define amdgpu_ps <1 x float> @raw_buffer_load_v1f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <1 x float> @llvm.amdgcn.raw.buffer.load.v1f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <1 x float> %data
}

; CHECK-LABEL: @raw_buffer_load_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @raw_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <2 x float> %data
}

; CHECK-LABEL: @raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <4 x float> %data
define amdgpu_ps <4 x float> @raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <4 x float> %data
}

; CHECK-LABEL: @extract_elt0_raw_buffer_load_v2f32(
; CHECK: %data = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_raw_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_raw_buffer_load_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_raw_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <2 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt3_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 3
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt3_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 3
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt2_elt3_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt2_elt3_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_elt3_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt1_elt2_elt3_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt2_elt3_raw_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt2_elt3_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_raw_buffer_load_v3f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_raw_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <3 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_raw_buffer_load_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_raw_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <3 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_raw_buffer_load_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <3 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_raw_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <3 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_raw_buffer_load_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_raw_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_raw_buffer_load_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_raw_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract0_bitcast_raw_buffer_load_v4f32(
; CHECK-NEXT: %tmp = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %tmp2 = bitcast float %tmp to i32
; CHECK-NEXT: ret i32 %tmp2
define i32 @extract0_bitcast_raw_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %tmp = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %tmp1 = bitcast <4 x float> %tmp to <4 x i32>
  %tmp2 = extractelement <4 x i32> %tmp1, i32 0
  ret i32 %tmp2
}

; CHECK-LABEL: @extract0_bitcast_raw_buffer_load_v4i32(
; CHECK-NEXT: %tmp = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %tmp2 = bitcast i32 %tmp to float
; CHECK-NEXT: ret float %tmp2
define float @extract0_bitcast_raw_buffer_load_v4i32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %tmp = call <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %tmp1 = bitcast <4 x i32> %tmp to <4 x float>
  %tmp2 = extractelement <4 x float> %tmp1, i32 0
  ret float %tmp2
}

; CHECK-LABEL: @preserve_metadata_extract_elt0_raw_buffer_load_v2f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent), !fpmath !0
; CHECK-NEXT: ret float %data
define amdgpu_ps float @preserve_metadata_extract_elt0_raw_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent), !fpmath !0
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

declare float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32>, i32, i32, i32) #1
declare <1 x float> @llvm.amdgcn.raw.buffer.load.v1f32(<4 x i32>, i32, i32, i32) #1
declare <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32>, i32, i32, i32) #1
declare <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32>, i32, i32, i32) #1
declare <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32>, i32, i32, i32) #1

declare <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32>, i32, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.raw.buffer.load.format
; --------------------------------------------------------------------

; CHECK-LABEL: @raw_buffer_load_format_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @raw_buffer_load_format_f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  ret float %data
}

; CHECK-LABEL: @raw_buffer_load_format_v1f32(
; CHECK-NEXT: %data = call <1 x float> @llvm.amdgcn.raw.buffer.load.format.v1f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <1 x float> %data
define amdgpu_ps <1 x float> @raw_buffer_load_format_v1f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <1 x float> @llvm.amdgcn.raw.buffer.load.format.v1f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <1 x float> %data
}

; CHECK-LABEL: @raw_buffer_load_format_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @raw_buffer_load_format_v2f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <2 x float> %data
}

; CHECK-LABEL: @raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <4 x float> %data
define amdgpu_ps <4 x float> @raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <4 x float> %data
}

; CHECK-LABEL: @extract_elt0_raw_buffer_load_format_v2f32(
; CHECK: %data = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_raw_buffer_load_format_v2f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_raw_buffer_load_format_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_raw_buffer_load_format_v2f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <2 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt3_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 3
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt3_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 3
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt2_elt3_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt2_elt3_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_elt3_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt1_elt2_elt3_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt2_elt3_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt2_elt3_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_raw_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_raw_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <3 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_raw_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_raw_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <3 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_raw_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <3 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_raw_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <3 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_raw_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_raw_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_raw_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_raw_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract0_bitcast_raw_buffer_load_format_v4f32(
; CHECK-NEXT: %tmp = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %tmp2 = bitcast float %tmp to i32
; CHECK-NEXT: ret i32 %tmp2
define i32 @extract0_bitcast_raw_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %tmp = call <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %tmp1 = bitcast <4 x float> %tmp to <4 x i32>
  %tmp2 = extractelement <4 x i32> %tmp1, i32 0
  ret i32 %tmp2
}

; CHECK-LABEL: @extract0_bitcast_raw_buffer_load_format_v4i32(
; CHECK-NEXT: %tmp = call i32 @llvm.amdgcn.raw.buffer.load.format.i32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %tmp2 = bitcast i32 %tmp to float
; CHECK-NEXT: ret float %tmp2
define float @extract0_bitcast_raw_buffer_load_format_v4i32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %tmp = call <4 x i32> @llvm.amdgcn.raw.buffer.load.format.v4i32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent)
  %tmp1 = bitcast <4 x i32> %tmp to <4 x float>
  %tmp2 = extractelement <4 x float> %tmp1, i32 0
  ret float %tmp2
}

; CHECK-LABEL: @preserve_metadata_extract_elt0_raw_buffer_load_format_v2f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent), !fpmath !0
; CHECK-NEXT: ret float %data
define amdgpu_ps float @preserve_metadata_extract_elt0_raw_buffer_load_format_v2f32(<4 x i32> inreg %rsrc, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %ofs, i32 %sofs, i32 %coherent), !fpmath !0
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

declare float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32>, i32, i32, i32) #1
declare <1 x float> @llvm.amdgcn.raw.buffer.load.format.v1f32(<4 x i32>, i32, i32, i32) #1
declare <2 x float> @llvm.amdgcn.raw.buffer.load.format.v2f32(<4 x i32>, i32, i32, i32) #1
declare <3 x float> @llvm.amdgcn.raw.buffer.load.format.v3f32(<4 x i32>, i32, i32, i32) #1
declare <4 x float> @llvm.amdgcn.raw.buffer.load.format.v4f32(<4 x i32>, i32, i32, i32) #1

declare <4 x i32> @llvm.amdgcn.raw.buffer.load.format.v4i32(<4 x i32>, i32, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.struct.buffer.load
; --------------------------------------------------------------------

; CHECK-LABEL: @struct_buffer_load_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @struct_buffer_load_f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  ret float %data
}

; CHECK-LABEL: @struct_buffer_load_v1f32(
; CHECK-NEXT: %data = call <1 x float> @llvm.amdgcn.struct.buffer.load.v1f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <1 x float> %data
define amdgpu_ps <1 x float> @struct_buffer_load_v1f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <1 x float> @llvm.amdgcn.struct.buffer.load.v1f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <1 x float> %data
}

; CHECK-LABEL: @struct_buffer_load_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @struct_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <2 x float> %data
}

; CHECK-LABEL: @struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <4 x float> %data
define amdgpu_ps <4 x float> @struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <4 x float> %data
}

; CHECK-LABEL: @extract_elt0_struct_buffer_load_v2f32(
; CHECK: %data = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_struct_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_struct_buffer_load_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_struct_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <2 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt3_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 3
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt3_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 3
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt2_elt3_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt2_elt3_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_elt3_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt1_elt2_elt3_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt2_elt3_struct_buffer_load_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt2_elt3_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_struct_buffer_load_v3f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_struct_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <3 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_struct_buffer_load_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_struct_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <3 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_struct_buffer_load_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <3 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_struct_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <3 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_struct_buffer_load_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_struct_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_struct_buffer_load_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_struct_buffer_load_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract0_bitcast_struct_buffer_load_v4f32(
; CHECK-NEXT: %tmp = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %tmp2 = bitcast float %tmp to i32
; CHECK-NEXT: ret i32 %tmp2
define i32 @extract0_bitcast_struct_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %tmp = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %tmp1 = bitcast <4 x float> %tmp to <4 x i32>
  %tmp2 = extractelement <4 x i32> %tmp1, i32 0
  ret i32 %tmp2
}

; CHECK-LABEL: @extract0_bitcast_struct_buffer_load_v4i32(
; CHECK-NEXT: %tmp = call i32 @llvm.amdgcn.struct.buffer.load.i32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %tmp2 = bitcast i32 %tmp to float
; CHECK-NEXT: ret float %tmp2
define float @extract0_bitcast_struct_buffer_load_v4i32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %tmp = call <4 x i32> @llvm.amdgcn.struct.buffer.load.v4i32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %tmp1 = bitcast <4 x i32> %tmp to <4 x float>
  %tmp2 = extractelement <4 x float> %tmp1, i32 0
  ret float %tmp2
}

; CHECK-LABEL: @preserve_metadata_extract_elt0_struct_buffer_load_v2f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent), !fpmath !0
; CHECK-NEXT: ret float %data
define amdgpu_ps float @preserve_metadata_extract_elt0_struct_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent), !fpmath !0
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

declare float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32>, i32, i32, i32, i32) #1
declare <1 x float> @llvm.amdgcn.struct.buffer.load.v1f32(<4 x i32>, i32, i32, i32, i32) #1
declare <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32>, i32, i32, i32, i32) #1
declare <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32>, i32, i32, i32, i32) #1
declare <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32>, i32, i32, i32, i32) #1

declare <4 x i32> @llvm.amdgcn.struct.buffer.load.v4i32(<4 x i32>, i32, i32, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.struct.buffer.load.format
; --------------------------------------------------------------------

; CHECK-LABEL: @struct_buffer_load_format_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @struct_buffer_load_format_f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  ret float %data
}

; CHECK-LABEL: @struct_buffer_load_format_v1f32(
; CHECK-NEXT: %data = call <1 x float> @llvm.amdgcn.struct.buffer.load.format.v1f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <1 x float> %data
define amdgpu_ps <1 x float> @struct_buffer_load_format_v1f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <1 x float> @llvm.amdgcn.struct.buffer.load.format.v1f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <1 x float> %data
}

; CHECK-LABEL: @struct_buffer_load_format_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @struct_buffer_load_format_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <2 x float> %data
}

; CHECK-LABEL: @struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <4 x float> %data
define amdgpu_ps <4 x float> @struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  ret <4 x float> %data
}

; CHECK-LABEL: @extract_elt0_struct_buffer_load_format_v2f32(
; CHECK: %data = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_struct_buffer_load_format_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_struct_buffer_load_format_v2f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_struct_buffer_load_format_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <2 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt3_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 3
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt3_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <4 x float> %data, i32 3
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt2_elt3_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt2_elt3_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_elt3_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt1_elt2_elt3_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt2_elt3_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt2_elt3_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 2, i32 3>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_struct_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_struct_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt0 = extractelement <3 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt1_struct_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <2 x float> %data, i32 1
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt1_struct_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <3 x float> %data, i32 1
  ret float %elt1
}

; CHECK-LABEL: @extract_elt2_struct_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %elt1 = extractelement <3 x float> %data, i32 2
; CHECK-NEXT: ret float %elt1
define amdgpu_ps float @extract_elt2_struct_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %elt1 = extractelement <3 x float> %data, i32 2
  ret float %elt1
}

; CHECK-LABEL: @extract_elt0_elt1_struct_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: ret <2 x float>
define amdgpu_ps <2 x float> @extract_elt0_elt1_struct_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt1_elt2_struct_buffer_load_format_v3f32(
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: ret <2 x float> %shuf
define amdgpu_ps <2 x float> @extract_elt1_elt2_struct_buffer_load_format_v3f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %shuf = shufflevector <3 x float> %data, <3 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract0_bitcast_struct_buffer_load_format_v4f32(
; CHECK-NEXT: %tmp = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %tmp2 = bitcast float %tmp to i32
; CHECK-NEXT: ret i32 %tmp2
define i32 @extract0_bitcast_struct_buffer_load_format_v4f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %tmp = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %tmp1 = bitcast <4 x float> %tmp to <4 x i32>
  %tmp2 = extractelement <4 x i32> %tmp1, i32 0
  ret i32 %tmp2
}

; CHECK-LABEL: @extract0_bitcast_struct_buffer_load_format_v4i32(
; CHECK-NEXT: %tmp = call i32 @llvm.amdgcn.struct.buffer.load.format.i32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
; CHECK-NEXT: %tmp2 = bitcast i32 %tmp to float
; CHECK-NEXT: ret float %tmp2
define float @extract0_bitcast_struct_buffer_load_format_v4i32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %tmp = call <4 x i32> @llvm.amdgcn.struct.buffer.load.format.v4i32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent)
  %tmp1 = bitcast <4 x i32> %tmp to <4 x float>
  %tmp2 = extractelement <4 x float> %tmp1, i32 0
  ret float %tmp2
}

; CHECK-LABEL: @preserve_metadata_extract_elt0_struct_buffer_load_format_v2f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent), !fpmath !0
; CHECK-NEXT: ret float %data
define amdgpu_ps float @preserve_metadata_extract_elt0_struct_buffer_load_format_v2f32(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent) #0 {
  %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i32 %sofs, i32 %coherent), !fpmath !0
  %elt0 = extractelement <2 x float> %data, i32 0
  ret float %elt0
}

declare float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32>, i32, i32, i32, i32) #1
declare <1 x float> @llvm.amdgcn.struct.buffer.load.format.v1f32(<4 x i32>, i32, i32, i32, i32) #1
declare <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32>, i32, i32, i32, i32) #1
declare <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32>, i32, i32, i32, i32) #1
declare <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32>, i32, i32, i32, i32) #1

declare <4 x i32> @llvm.amdgcn.struct.buffer.load.format.v4i32(<4 x i32>, i32, i32, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_1d_v4f32_f32(float %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; Check that the intrinsic remains unchanged in the presence of TFE or LWE
; CHECK-LABEL: @extract_elt0_image_sample_1d_v4f32_f32_tfe(
; CHECK-NEXT: %data = call { <4 x float>, i32 } @llvm.amdgcn.image.sample.1d.sl_v4f32i32s.f32(i32 15, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 1, i32 0)
; CHECK: ret float %elt0
define amdgpu_ps float @extract_elt0_image_sample_1d_v4f32_f32_tfe(float %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.sl_v4f32i32s.f32(i32 15, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 1, i32 0)
  %data.vec = extractvalue {<4 x float>,i32} %data, 0
  %elt0 = extractelement <4 x float> %data.vec, i32 0
  ret float %elt0
}

; Check that the intrinsic remains unchanged in the presence of TFE or LWE
; CHECK-LABEL: @extract_elt0_image_sample_1d_v4f32_f32_lwe(
; CHECK-NEXT: %data = call { <4 x float>, i32 } @llvm.amdgcn.image.sample.1d.sl_v4f32i32s.f32(i32 15, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 2, i32 0)
; CHECK: ret float %elt0
define amdgpu_ps float @extract_elt0_image_sample_1d_v4f32_f32_lwe(float %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.sl_v4f32i32s.f32(i32 15, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 2, i32 0)
  %data.vec = extractvalue {<4 x float>,i32} %data, 0
  %elt0 = extractelement <4 x float> %data.vec, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_sample_2d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.2d.f32.f32(i32 1, float %s, float %t, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_2d_v4f32_f32(float %s, float %t, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %s, float %t, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_invalid_dmask_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 %dmask, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: %elt0 = extractelement <4 x float> %data, i32 0
; CHECK-NEXT: ret float %elt0
define amdgpu_ps float @extract_elt0_invalid_dmask_image_sample_1d_v4f32_f32(float %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc, i32 %dmask) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 %dmask, float %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_0000_image_sample_3d_v4f32_f32(
; CHECK-NEXT: ret float undef
define amdgpu_ps float @extract_elt0_dmask_0000_image_sample_3d_v4f32_f32(float %s, float %t, float %r, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32 0, float %s, float %t, float %r, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_0001_image_sample_1darray_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1darray.f32.f32(i32 1, float %s, float %slice, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0001_image_sample_1darray_v4f32_f32(float %s, float %slice, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f32(i32 1, float %s, float %slice, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_0010_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 2, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0010_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 2, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_0100_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 4, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0100_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 4, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_1000_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 8, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_1000_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 8, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_1001_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_1001_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 9, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_0011_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0011_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 3, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_0111_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0111_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 7, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_elt1_dmask_0001_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: %1 = insertelement <2 x float> undef, float %data, i32 0
; CHECK-NEXT: ret <2 x float> %1
define amdgpu_ps <2 x float> @extract_elt0_elt1_dmask_0001_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 1, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_dmask_0011_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32 3, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt0_elt1_dmask_0011_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 3, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_dmask_0111_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32 3, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt0_elt1_dmask_0111_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 7, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_dmask_0101_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32 5, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt0_elt1_dmask_0101_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 5, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_dmask_0001_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: %1 = insertelement <3 x float> undef, float %data, i32 0
; CHECK-NEXT: ret <3 x float> %1
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_dmask_0001_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 1, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_dmask_0011_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32 3, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: %shuf = shufflevector <2 x float> %data, <2 x float> undef, <3 x i32> <i32 0, i32 1, i32 undef>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_dmask_0011_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 3, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_dmask_0101_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32 5, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: %shuf = shufflevector <2 x float> %data, <2 x float> undef, <3 x i32> <i32 0, i32 1, i32 undef>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_dmask_0101_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 5, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_dmask_0111_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 7, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_dmask_0111_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 7, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_dmask_1111_image_sample_1d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 7, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_dmask_1111_image_sample_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.sl_v4f32i32s.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt1_image_sample_cl_2darray_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cl.2darray.f32.f32(i32 2, float %s, float %t, float %slice, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt1_image_sample_cl_2darray_v4f32_f32(float %s, float %t, float %slice, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cl.2darray.v4f32.f32(i32 15, float %s, float %t, float %slice, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 1
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cl.2darray.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.d
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt2_image_sample_d_cube_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.d.cube.f32.f32.f32(i32 4, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %face, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt2_image_sample_d_cube_v4f32_f32_f32(float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %face, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.d.cube.v4f32.f32.f32(i32 15, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %face, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 2
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.d.cube.v4f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.d.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt3_image_sample_d_cl_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.d.cl.1d.f32.f32.f32(i32 8, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt3_image_sample_d_cl_1d_v4f32_f32_f32(float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.d.cl.1d.v4f32.f32.f32(i32 15, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 3
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.d.cl.1d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.l
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt1_dmask_0110_image_sample_l_1d_v2f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.l.1d.f32.f32(i32 4, float %s, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt1_dmask_0110_image_sample_l_1d_v2f32_f32(float %s, float %lod, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <2 x float> @llvm.amdgcn.image.sample.l.1d.v2f32.f32(i32 6, float %s, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <2 x float> %data, i32 1
  ret float %elt0
}

declare <2 x float> @llvm.amdgcn.image.sample.l.1d.v2f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.b
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt1_dmask_1001_image_sample_b_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.b.1d.f32.f32.f32(i32 8, float %bias, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt1_dmask_1001_image_sample_b_1d_v4f32_f32_f32(float %bias, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.b.1d.v4f32.f32.f32(i32 9, float %bias, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 1
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.b.1d.v4f32.f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.b.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt1_elt2_dmask_1101_image_sample_b_cl_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.b.cl.1d.v2f32.f32.f32(i32 12, float %bias, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt1_elt2_dmask_1101_image_sample_b_cl_1d_v4f32_f32_f32(float %bias, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.b.cl.1d.v4f32.f32.f32(i32 13, float %bias, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %shuf
}

declare <4 x float> @llvm.amdgcn.image.sample.b.cl.1d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.lz
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt1_elt3_image_sample_lz_1d_v4f32_f32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.lz.1d.v2f32.f32(i32 10, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt1_elt3_image_sample_lz_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32 15, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 1, i32 3>
  ret <2 x float> %shuf
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cd
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt1_elt2_elt3_image_sample_cd_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.sample.cd.1d.v4f32.f32.f32(i32 14, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)

; TODO: This should be combined with the following shufflevector
; CHECK-NEXT: %1 = shufflevector <4 x float> %data, <4 x float> undef, <4 x i32> <i32 undef, i32 0, i32 1, i32 2>

; CHECK-NEXT: %shuf = shufflevector <4 x float> %1, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt1_elt2_elt3_image_sample_cd_1d_v4f32_f32_f32(float %dsdh, float %dsdv, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cd.1d.v4f32.f32.f32(i32 15, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
  ret <3 x float> %shuf
}

declare <4 x float> @llvm.amdgcn.image.sample.cd.1d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cd.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cd_cl_1d_v4f16_f32_f32(
; CHECK-NEXT: %data = call half @llvm.amdgcn.image.sample.cd.cl.1d.f16.f32.f32(i32 1, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret half %data
define amdgpu_ps half @extract_elt0_image_sample_cd_cl_1d_v4f16_f32_f32(float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x half> @llvm.amdgcn.image.sample.cd.cl.1d.v4f16.f32.f32(i32 15, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x half> %data, i32 0
  ret half %elt0
}

declare <4 x half> @llvm.amdgcn.image.sample.cd.cl.1d.v4f16.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.1d.f32.f32(i32 1, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_1d_v4f32_f32(float %zcompare, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.1d.v4f32.f32(i32 15, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.1d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cl_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cl.1d.f32.f32(i32 1, float %zcompare, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cl_1d_v4f32_f32(float %zcompare, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cl.1d.v4f32.f32(i32 15, float %zcompare, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cl.1d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.d
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_d_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.d.1d.f32.f32.f32(i32 1, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_d_1d_v4f32_f32_f32(float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.d.1d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.d.1d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.d.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_d_cl_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.d.cl.1d.f32.f32.f32(i32 1, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_d_cl_1d_v4f32_f32_f32(float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.1d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.1d.v4f32.f32.f32(i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.l
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_l_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.l.1d.f32.f32(i32 1, float %zcompare, float %s, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_l_1d_v4f32_f32(float %zcompare, float %s, float %lod, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.l.1d.v4f32.f32(i32 15, float %zcompare, float %s, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.l.1d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.b
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_b_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.b.1d.f32.f32.f32(i32 1, float %bias, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_b_1d_v4f32_f32_f32(float %bias, float %zcompare, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.b.1d.v4f32.f32.f32(i32 15, float %bias, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.b.1d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.b.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_b_cl_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.b.cl.1d.f32.f32.f32(i32 1, float %bias, float %zcompare, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_b_cl_1d_v4f32_f32_f32(float %bias, float %zcompare, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.1d.v4f32.f32.f32(i32 15, float %bias, float %zcompare, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.1d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.lz
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_lz_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.lz.1d.f32.f32(i32 1, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_lz_1d_v4f32_f32(float %zcompare, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.lz.1d.v4f32.f32(i32 15, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.lz.1d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cd
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cd_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cd.1d.f32.f32.f32(i32 1, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cd_1d_v4f32_f32_f32(float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cd.1d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cd.1d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cd.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cd_cl_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cd.cl.1d.f32.f32.f32(i32 1, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cd_cl_1d_v4f32_f32_f32(float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.1d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.1d.v4f32.f32.f32(i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_o_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.o.1d.f32.f32(i32 1, i32 %offset, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_o_1d_v4f32_f32(i32 %offset, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.o.1d.v4f32.f32(i32 15, i32 %offset, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.o.1d.v4f32.f32(i32, i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cl_o_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cl.o.1d.f32.f32(i32 1, i32 %offset, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_cl_o_1d_v4f32_f32(i32 %offset, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cl.o.1d.v4f32.f32(i32 15, i32 %offset, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cl.o.1d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.d.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_d_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.d.o.1d.f32.f32.f32(i32 1, i32 %offset, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_d_o_1d_v4f32_f32_f32(i32 %offset, float %dsdh, float %dsdv, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.d.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.d.o.1d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.d.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_d_cl_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.d.cl.o.1d.f32.f32.f32(i32 1, i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_d_cl_o_1d_v4f32_f32_f32(i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.d.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.d.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.l.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_l_o_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.l.o.1d.f32.f32(i32 1, i32 %offset, float %s, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_l_o_1d_v4f32_f32(i32 %offset, float %s, float %lod, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.l.o.1d.v4f32.f32(i32 15, i32 %offset, float %s, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.l.o.1d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.b.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_b_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.b.o.1d.f32.f32.f32(i32 1, i32 %offset, float %bias, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_b_o_1d_v4f32_f32_f32(i32 %offset, float %bias, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.b.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.b.o.1d.v4f32.f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.b.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_b_cl_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.b.cl.o.1d.f32.f32.f32(i32 1, i32 %offset, float %bias, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_b_cl_o_1d_v4f32_f32_f32(i32 %offset, float %bias, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.b.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.b.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.lz.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_lz_o_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.lz.o.1d.f32.f32(i32 1, i32 %offset, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_lz_o_1d_v4f32_f32(i32 %offset, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.lz.o.1d.v4f32.f32(i32 15, i32 %offset, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.o.1d.v4f32.f32(i32, i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cd.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cd_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cd.o.1d.f32.f32.f32(i32 1, i32 %offset, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_cd_o_1d_v4f32_f32_f32(i32 %offset, float %dsdh, float %dsdv, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cd.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cd.o.1d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cd.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cd_cl_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cd.cl.o.1d.f32.f32.f32(i32 1, i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_cd_cl_o_1d_v4f32_f32_f32(i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_o_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.o.1d.f32.f32(i32 1, i32 %offset, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_o_1d_v4f32_f32(i32 %offset, float %zcompare, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.o.1d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.o.1d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cl_o_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cl.o.1d.f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cl_o_1d_v4f32_f32(i32 %offset, float %zcompare, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cl.o.1d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cl.o.1d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.d.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_d_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.d.o.1d.f32.f32.f32(i32 1, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_d_o_1d_v4f32_f32_f32(i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.d.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.d.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.d.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_d_cl_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.d.cl.o.1d.f32.f32.f32(i32 1, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_d_cl_o_1d_v4f32_f32_f32(i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.l.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_l_o_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.l.o.1d.f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_l_o_1d_v4f32_f32(i32 %offset, float %zcompare, float %s, float %lod, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.l.o.1d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.l.o.1d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.b.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_b_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.b.o.1d.f32.f32.f32(i32 1, i32 %offset, float %bias, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_b_o_1d_v4f32_f32_f32(i32 %offset, float %bias, float %zcompare, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.b.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.b.o.1d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.b.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_b_cl_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.b.cl.o.1d.f32.f32.f32(i32 1, i32 %offset, float %bias, float %zcompare, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_b_cl_o_1d_v4f32_f32_f32(i32 %offset, float %bias, float %zcompare, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %zcompare, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.lz.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_lz_o_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.lz.o.1d.f32.f32(i32 1, i32 %offset, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_lz_o_1d_v4f32_f32(i32 %offset, float %zcompare, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.lz.o.1d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.lz.o.1d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cd.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cd_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cd.o.1d.f32.f32.f32(i32 1, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cd_o_1d_v4f32_f32_f32(i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cd.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cd.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cd.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cd_cl_o_1d_v4f32_f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cd.cl.o.1d.f32.f32.f32(i32 1, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cd_cl_o_1d_v4f32_f32_f32(i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4
; --------------------------------------------------------------------

; Don't handle gather4*

; CHECK-LABEL: @extract_elt0_image_gather4_2d_v4f32_f32(
; CHECK: %data = call <4 x float> @llvm.amdgcn.image.gather4.2d.v4f32.f32(i32 1, float %s, float %t, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_2d_v4f32_f32(float %s, float %t, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.2d.v4f32.f32(i32 1, float %s, float %t, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_cl_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.cl.2d.v4f32.f32(i32 2, float %s, float %t, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_cl_2d_v4f32_f32(float %s, float %t, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.cl.2d.v4f32.f32(i32 2, float %s, float %t, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.cl.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.l
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_l_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.l.2d.v4f32.f32(i32 4, float %s, float %t, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_l_2d_v4f32_f32(float %s, float %t, float %lod, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.l.2d.v4f32.f32(i32 4, float %s, float %t, float %lod, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.l.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.b
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_b_2darray_v4f32_f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.b.2darray.v4f32.f32.f32(i32 8, float %bias, float %s, float %t, float %slice, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_b_2darray_v4f32_f32_f32(float %bias, float %s, float %t, float %slice, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.b.2darray.v4f32.f32.f32(i32 8, float %bias, float %s, float %t, float %slice, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.b.2darray.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.b.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_b_cl_cube_v4f32_f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.cube.v4f32.f32.f32(i32 1, float %bias, float %s, float %t, float %face, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_b_cl_cube_v4f32_f32_f32(float %bias, float %s, float %t, float %face, float %clamp, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.cube.v4f32.f32.f32(i32 1, float %bias, float %s, float %t, float %face, float %clamp, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.b.cl.cube.v4f32.f32.f32(i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.lz
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_lz_2d_v4f32_f16(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f16(i32 1, half %s, half %t, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_lz_2d_v4f32_f16(half %s, half %t, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f16(i32 1, half %s, half %t, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f16(i32, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_o_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_o_2d_v4f32_f32(i32 %offset, float %s, float %t, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.o.2d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_cl_o_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.cl.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, float %clamp, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_cl_o_2d_v4f32_f32(i32 %offset, float %s, float %t, float %clamp, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.cl.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, float %clamp, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.cl.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.l.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_l_o_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.l.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, float %lod, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_l_o_2d_v4f32_f32(i32 %offset, float %s, float %t, float %lod, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.l.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, float %lod, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.l.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.b.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_b_o_2d_v4f32_f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.b.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_b_o_2d_v4f32_f32_f32(i32 %offset, float %bias, float %s, float %t, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.b.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.b.o.2d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.b.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_b_cl_o_2d_v4f32_f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %s, float %t, float %clamp, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_b_cl_o_2d_v4f32_f32_f32(i32 %offset, float %bias, float %s, float %t, float %clamp, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %s, float %t, float %clamp, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.lz.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_lz_o_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.lz.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_lz_o_2d_v4f32_f32(i32 %offset, float %s, float %t, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.lz.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.lz.o.2d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_o_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_c_o_2d_v4f32_f32(i32 %offset, float %zcompare, float %s, float %t, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_cl_o_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_c_cl_o_2d_v4f32_f32(i32 %offset, float %zcompare, float %s, float %t, float %clamp, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.2d.v4f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.l.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_l_o_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.l.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, float %lod, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_c_l_o_2d_v4f32_f32(i32 %offset, float %zcompare, float %s, float %t, float %lod, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.l.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, float %lod, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.l.o.2d.v4f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.b.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_b_o_2d_v4f32_f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.b.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %zcompare, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_c_b_o_2d_v4f32_f32_f32(i32 %offset, float %bias, float %zcompare, float %s, float %t, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.b.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %zcompare, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.b.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.b.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_b_cl_o_2d_v4f32_f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_c_b_cl_o_2d_v4f32_f32_f32(i32 %offset, float %bias, float %zcompare, float %s, float %t, float %clamp, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.lz.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_lz_o_2d_v4f32_f32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
define amdgpu_ps float @extract_elt0_image_gather4_c_lz_o_2d_v4f32_f32(i32 %offset, float %zcompare, float %s, float %t, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, <8 x i32> %gather4r, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.getlod
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_getlod_1d_v4f32_f32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.getlod.1d.f32.f32(i32 1, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_getlod_1d_v4f32_f32(float %s, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.getlod.1d.v4f32.f32(i32 15, float %s, <8 x i32> %sampler, <4 x i32> %rsrc, i1 false, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.getlod.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.load
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_load_2dmsaa_v4f32_i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 %sample, <8 x i32> %sampler, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_load_2dmsaa_v4f32_i32(i32 %s, i32 %t, i32 %sample, <8 x i32> inreg %sampler) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %sample, <8 x i32> %sampler, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.load.mip
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_load_mip_1d_v4f32_i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.load.mip.1d.f32.i32(i32 1, i32 %s, i32 %mip, <8 x i32> %sampler, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_load_mip_1d_v4f32_i32(i32 %s, i32 %mip, <8 x i32> inreg %sampler) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32 15, i32 %s, i32 %mip, <8 x i32> %sampler, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.getresinfo
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_getresinfo_1d_v4f32_i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.getresinfo.1d.f32.i32(i32 1, i32 %mip, <8 x i32> %sampler, i32 0, i32 0)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_getresinfo_1d_v4f32_i32(i32 %mip, <8 x i32> inreg %sampler) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32 15, i32 %mip, <8 x i32> %sampler, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }

!0 = !{float 2.500000e+00}
