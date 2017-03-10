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
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %elt1 = extractelement <4 x float> %data, i32 1
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
; CHECK-NEXT: %data = call <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 false, i1 false)
; CHECK-NEXT: %elt1 = extractelement <3 x float> %data, i32 1
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

declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1) #1
declare <1 x float> @llvm.amdgcn.buffer.load.v1f32(<4 x i32>, i32, i32, i1, i1) #1
declare <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32>, i32, i32, i1, i1) #1
declare <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32>, i32, i32, i1, i1) #1
declare <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32>, i32, i32, i1, i1) #1

declare float @llvm.amdgcn.buffer.load.format.f32(<4 x i32>, i32, i32, i1, i1) #1
declare <1 x float> @llvm.amdgcn.buffer.load.format.v1f32(<4 x i32>, i32, i32, i1, i1) #1
declare <2 x float> @llvm.amdgcn.buffer.load.format.v2f32(<4 x i32>, i32, i32, i1, i1) #1
declare <3 x float> @llvm.amdgcn.buffer.load.format.v3f32(<4 x i32>, i32, i32, i1, i1) #1
declare <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32>, i32, i32, i1, i1) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }

!0 = !{float 2.500000e+00}
