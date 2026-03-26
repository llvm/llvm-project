define <4 x float> @canon_fp32_varargsv4f32(<4 x float> %a) {
  %canonicalized = call <4 x float> @llvm.canonicalize.v4f32(<4 x float> %a)
  ret <4 x float> %canonicalized
}