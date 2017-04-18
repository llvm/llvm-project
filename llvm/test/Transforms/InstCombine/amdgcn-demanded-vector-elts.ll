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
; CHECK-NEXT: %1 = insertelement <4 x float> undef, float %tmp, i64 0
; CHECK-NEXT: %tmp1 = bitcast <4 x float> %1 to <8 x i16>
; CHECK-NEXT: %tmp2 = extractelement <8 x i16> %tmp1, i32 0
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
; llvm.amdgcn.image.sample
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_sample_v4f32_v4f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_v4f32_v4f32_v4i32(<4 x float> %vaddr, <4 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_sample_v4f32_v2f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_v4f32_v2f32_v4i32(<2 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_invalid_dmask_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 %dmask, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_invalid_dmask_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc, i32 %dmask) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 %dmask, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; FIXME: Should really fold to undef
; CHECK-LABEL: @extract_elt0_dmask_0000_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0000_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_0001_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0001_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; FIXME: Should really fold to undef
; CHECK-LABEL: @extract_elt0_dmask_0010_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 2, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0010_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 2, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; FIXME: Should really fold to undef
; CHECK-LABEL: @extract_elt0_dmask_0100_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 4, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0100_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 4, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; FIXME: Should really fold to undef
; CHECK-LABEL: @extract_elt0_dmask_1000_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 8, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_1000_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 8, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_1001_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_1001_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 9, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_0011_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0011_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_dmask_0111_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_dmask_0111_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 7, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_elt1_dmask_0001_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.v2f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt0_elt1_dmask_0001_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_dmask_0011_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.v2f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt0_elt1_dmask_0011_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_dmask_0111_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.v2f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt0_elt1_dmask_0111_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 7, i1 false, i1 false, i1 false, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_dmask_0101_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <2 x float> @llvm.amdgcn.image.sample.v2f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 5, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret <2 x float> %data
define amdgpu_ps <2 x float> @extract_elt0_elt1_dmask_0101_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 5, i1 false, i1 false, i1 false, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_dmask_0001_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_dmask_0001_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_dmask_0011_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_dmask_0011_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_dmask_0101_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 5, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_dmask_0101_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 5, i1 false, i1 false, i1 false, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

; CHECK-LABEL: @extract_elt0_elt1_elt2_dmask_0111_image_sample_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 7, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT: ret <3 x float> %shuf
define amdgpu_ps <3 x float> @extract_elt0_elt1_elt2_dmask_0111_image_sample_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 7, i1 false, i1 false, i1 false, i1 false, i1 false)
  %shuf = shufflevector <4 x float> %data, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %shuf
}

declare <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v4i32(<4 x float>, <4 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.sample.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cl.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.d
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_d_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.d.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_d_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.d.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.d.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.d.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_d_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.d.cl.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_d_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.d.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.d.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.l
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_l_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.l.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_l_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.l.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.l.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.b
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_b_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.b.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_b_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.b.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.b.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.b.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_b_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.b.cl.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_b_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.b.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.b.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.lz
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_lz_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.lz.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_lz_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.lz.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cd
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cd_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cd.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_cd_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cd.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cd.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cd.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cd_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cd.cl.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_cd_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_sample_c_v4f32_v4f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_v4f32_v4f32_v4i32(<4 x float> %vaddr, <4 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_sample_c_v4f32_v2f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_v4f32_v2f32_v4i32(<2 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v4f32.v4i32(<4 x float>, <4 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cl.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.d
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_d_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.d.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_d_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.d.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.d.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.d.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_d_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.d.cl.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_d_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.l
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_l_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.l.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_l_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.l.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.l.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.b
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_b_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.b.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_b_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.b.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.b.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.b.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_b_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.b.cl.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_b_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.lz
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_lz_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.lz.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_lz_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.lz.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.lz.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cd
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cd_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cd.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cd_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cd.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cd.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cd.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cd_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cd.cl.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cd_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_sample_o_v4f32_v4f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.o.f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_o_v4f32_v4f32_v4i32(<4 x float> %vaddr, <4 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.o.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_sample_o_v4f32_v2f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.o.f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_o_v4f32_v2f32_v4i32(<2 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.o.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.sample.o.v4f32.v4f32.v4i32(<4 x float>, <4 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.sample.o.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cl.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.d.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_d_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.d.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_d_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.d.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.d.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.d.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_d_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.d.cl.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_d_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.d.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.d.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.l.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_l_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.l.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_l_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.l.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.l.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.b.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_b_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.b.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_b_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.b.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.b.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.b.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_b_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.b.cl.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_b_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.b.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.b.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.lz.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_lz_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.lz.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_lz_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.lz.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cd.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cd_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cd.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_cd_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cd.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cd.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.cd.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_cd_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.cd.cl.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_cd_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_sample_c_o_v4f32_v4f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.o.f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_o_v4f32_v4f32_v4i32(<4 x float> %vaddr, <4 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.o.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_sample_c_o_v4f32_v2f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.o.f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_o_v4f32_v2f32_v4i32(<2 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.o.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.o.v4f32.v4f32.v4i32(<4 x float>, <4 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.o.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cl.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.d.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_d_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.d.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_d_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.d.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.d.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.d.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_d_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.d.cl.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_d_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.l.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_l_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.l.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_l_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.l.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.l.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.b.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_b_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.b.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_b_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.b.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.b.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.b.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_b_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.b.cl.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_b_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.lz.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_lz_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.lz.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_lz_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.lz.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.lz.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cd.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cd_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cd.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cd_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cd.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cd.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.sample.c.cd.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_sample_c_cd_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.sample.c.cd.cl.o.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_sample_c_cd_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4
; --------------------------------------------------------------------

; Don't handle gather4*

; CHECK-LABEL: @extract_elt0_image_gather4_v4f32_v4f32_v8i32(
; CHECK: %data = call <4 x float> @llvm.amdgcn.image.gather4.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i3
define amdgpu_ps float @extract_elt0_image_gather4_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_gather4_v4f32_v4f32_v4i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_v4f32_v4f32_v4i32(<4 x float> %vaddr, <4 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_gather4_v4f32_v2f32_v4i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_v4f32_v2f32_v4i32(<2 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.gather4.v4f32.v4f32.v4i32(<4 x float>, <4 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.gather4.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.l
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_l_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.l.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_l_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.l.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.l.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.b
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_b_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.b.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_b_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.b.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.b.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.b.cl
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_b_cl_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_b_cl_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.b.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.lz
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_lz_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.lz.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_lz_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.lz.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.lz.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_gather4_o_v4f32_v4f32_v4i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_o_v4f32_v4f32_v4i32(<4 x float> %vaddr, <4 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_gather4_o_v4f32_v2f32_v4i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_o_v4f32_v2f32_v4i32(<2 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v4f32.v4i32(<4 x float>, <4 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.l.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_l_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.l.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_l_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.l.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.l.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.b.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_b_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.b.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_b_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.b.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.b.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.b.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_b_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_b_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.lz.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_lz_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.lz.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_lz_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.lz.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.lz.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_c_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_gather4_c_o_v4f32_v4f32_v4i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_c_o_v4f32_v4f32_v4i32(<4 x float> %vaddr, <4 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_gather4_c_o_v4f32_v2f32_v4i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_c_o_v4f32_v2f32_v4i32(<2 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v4f32.v4i32(<4 x float>, <4 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_c_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.l.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_l_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.l.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_c_l_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.l.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.l.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.b.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_b_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.b.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_c_b_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.b.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.b.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.b.cl.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_b_cl_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_c_b_cl_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.gather4.c.lz.o
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_gather4_c_lz_o_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
define amdgpu_ps float @extract_elt0_image_gather4_c_lz_o_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %gather4r, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %gather4r, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

; --------------------------------------------------------------------
; llvm.amdgcn.image.getlod
; --------------------------------------------------------------------

; CHECK-LABEL: @extract_elt0_image_getlod_v4f32_v4f32_v8i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.getlod.f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_getlod_v4f32_v4f32_v8i32(<4 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.getlod.v4f32.v4f32.v8i32(<4 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_getlod_v4f32_v4f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.getlod.f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_getlod_v4f32_v4f32_v4i32(<4 x float> %vaddr, <4 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.getlod.v4f32.v4f32.v4i32(<4 x float> %vaddr, <4 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

; CHECK-LABEL: @extract_elt0_image_getlod_v4f32_v2f32_v4i32(
; CHECK-NEXT: %data = call float @llvm.amdgcn.image.getlod.f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false, i1 false)
; CHECK-NEXT: ret float %data
define amdgpu_ps float @extract_elt0_image_getlod_v4f32_v2f32_v4i32(<2 x float> %vaddr, <8 x i32> inreg %sampler, <4 x i32> inreg %rsrc) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.getlod.v4f32.v2f32.v8i32(<2 x float> %vaddr, <8 x i32> %sampler, <4 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %data, i32 0
  ret float %elt0
}

declare <4 x float> @llvm.amdgcn.image.getlod.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.getlod.v4f32.v4f32.v4i32(<4 x float>, <4 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.getlod.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }

!0 = !{float 2.500000e+00}
