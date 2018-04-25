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

class TextureIntrinsicFloatInput : Intrinsic<[llvm_v4f32_ty], [
  llvm_v4f32_ty, // Coord
  llvm_i32_ty,   // offset_x
  llvm_i32_ty,   // offset_y,
  llvm_i32_ty,   // offset_z,
  llvm_i32_ty,   // resource_id
  llvm_i32_ty,   // samplerid
  llvm_i32_ty,   // coord_type_x
  llvm_i32_ty,   // coord_type_y
  llvm_i32_ty,   // coord_type_z
  llvm_i32_ty],  // coord_type_w
  [IntrNoMem]
>;

class TextureIntrinsicInt32Input : Intrinsic<[llvm_v4i32_ty], [
    llvm_v4i32_ty, // Coord
    llvm_i32_ty,   // offset_x
    llvm_i32_ty,   // offset_y,
    llvm_i32_ty,   // offset_z,
    llvm_i32_ty,   // resource_id
    llvm_i32_ty,   // samplerid
    llvm_i32_ty,   // coord_type_x
    llvm_i32_ty,   // coord_type_y
    llvm_i32_ty,   // coord_type_z
    llvm_i32_ty],  // coord_type_w
    [IntrNoMem]
>;

let TargetPrefix = "r600", isTarget = 1 in {

def int_r600_store_swizzle :
  Intrinsic<[], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty], []
>;

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

def int_r600_dot4 : Intrinsic<[llvm_float_ty],
  [llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem, IntrSpeculatable]
>;

} // End TargetPrefix = "r600", isTarget = 1
