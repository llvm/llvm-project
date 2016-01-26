//===-- R600Intrinsics.td - R600 Instrinsic defs -------*- tablegen -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// R600 Intrinsic Definitions
//
//===----------------------------------------------------------------------===//

// FIXME: Should migrate to using TargetPrefix that matches triple arch name.
let TargetPrefix = "AMDGPU", isTarget = 1 in {
  def int_AMDGPU_dp4 : Intrinsic<[llvm_float_ty], [llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
  def int_AMDGPU_tex : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txb : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txf : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txq : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txd : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txl : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_ddx : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_ddy : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
}

let TargetPrefix = "R600", isTarget = 1 in {
  class TextureIntrinsicFloatInput :
    Intrinsic<[llvm_v4f32_ty], [
      llvm_v4f32_ty, // Coord
      llvm_i32_ty, // offset_x
      llvm_i32_ty, // offset_y,
      llvm_i32_ty, // offset_z,
      llvm_i32_ty, // resource_id
      llvm_i32_ty, // samplerid
      llvm_i32_ty, // coord_type_x
      llvm_i32_ty, // coord_type_y
      llvm_i32_ty, // coord_type_z
      llvm_i32_ty // coord_type_w
    ], [IntrNoMem]>;
  class TextureIntrinsicInt32Input :
    Intrinsic<[llvm_v4i32_ty], [
      llvm_v4i32_ty, // Coord
      llvm_i32_ty, // offset_x
      llvm_i32_ty, // offset_y,
      llvm_i32_ty, // offset_z,
      llvm_i32_ty, // resource_id
      llvm_i32_ty, // samplerid
      llvm_i32_ty, // coord_type_x
      llvm_i32_ty, // coord_type_y
      llvm_i32_ty, // coord_type_z
      llvm_i32_ty // coord_type_w
    ], [IntrNoMem]>;

  def int_R600_ldptr : TextureIntrinsicInt32Input;

  def int_R600_interp_const :
    Intrinsic<[llvm_v4f32_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_R600_interp_xy :
    Intrinsic<[llvm_v2f32_ty], [llvm_i32_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_R600_interp_zw :
    Intrinsic<[llvm_v2f32_ty], [llvm_i32_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_R600_load_texbuf :
    Intrinsic<[llvm_v4f32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_R600_store_swizzle :
    Intrinsic<[], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_R600_store_stream_output :
    Intrinsic<[], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_R600_store_pixel_depth :
      Intrinsic<[], [llvm_float_ty], []>;
  def int_R600_store_pixel_stencil :
      Intrinsic<[], [llvm_float_ty], []>;
  def int_R600_store_dummy :
      Intrinsic<[], [llvm_i32_ty], []>;
} // End TargetPrefix = "R600", isTarget = 1

let TargetPrefix = "r600", isTarget = 1 in {
  def int_r600_tex : TextureIntrinsicFloatInput;
  def int_r600_texc : TextureIntrinsicFloatInput;
  def int_r600_txl : TextureIntrinsicFloatInput;
  def int_r600_txlc : TextureIntrinsicFloatInput;
  def int_r600_txb : TextureIntrinsicFloatInput;
  def int_r600_txbc : TextureIntrinsicFloatInput;
  def int_r600_txf : TextureIntrinsicInt32Input;
  def int_r600_txq : TextureIntrinsicInt32Input;
  def int_r600_ddx : TextureIntrinsicFloatInput;
  def int_r600_ddy : TextureIntrinsicFloatInput;
} // End TargetPrefix = "r600", isTarget = 1
