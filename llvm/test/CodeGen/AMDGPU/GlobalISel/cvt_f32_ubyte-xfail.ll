; RUN: not --crash llc -global-isel -mtriple=amdgcn-- -mcpu=tahiti -verify-machineinstrs < %s
; RUN: not --crash llc -global-isel -mtriple=amdgcn-- -mcpu=tonga -verify-machineinstrs < %s

define <3 x float> @v_uitofp_v3i8_to_v3f32(i32 %arg0) nounwind {
  %trunc = trunc i32 %arg0 to i24
  %val = bitcast i24 %trunc to <3 x i8>
  %cvt = uitofp <3 x i8> %val to <3 x float>
  ret <3 x float> %cvt
}