//===-- SIIntrinsics.td - SI Intrinsic defs ----------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SI Intrinsic Definitions
//
//===----------------------------------------------------------------------===//


let TargetPrefix = "SI", isTarget = 1 in {

  def int_SI_tid : Intrinsic <[llvm_i32_ty], [], [IntrNoMem]>;
  def int_SI_packf16 : Intrinsic <[llvm_i32_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_SI_export : Intrinsic <[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], []>;
  def int_SI_load_const : Intrinsic <[llvm_float_ty], [llvm_anyint_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_SI_vs_load_input : Intrinsic <[llvm_v4f32_ty], [llvm_anyint_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]> ;

  // Fully-flexible TBUFFER_STORE_FORMAT_* except for the ADDR64 bit, which is not exposed
  def int_SI_tbuffer_store : Intrinsic <
    [],
    [llvm_anyint_ty, // rsrc(SGPR)
     llvm_anyint_ty, // vdata(VGPR), overloaded for types i32, v2i32, v4i32
     llvm_i32_ty,    // num_channels(imm), selects opcode suffix: 1=X, 2=XY, 3=XYZ, 4=XYZW
     llvm_i32_ty,    // vaddr(VGPR)
     llvm_i32_ty,    // soffset(SGPR)
     llvm_i32_ty,    // inst_offset(imm)
     llvm_i32_ty,    // dfmt(imm)
     llvm_i32_ty,    // nfmt(imm)
     llvm_i32_ty,    // offen(imm)
     llvm_i32_ty,    // idxen(imm)
     llvm_i32_ty,    // glc(imm)
     llvm_i32_ty,    // slc(imm)
     llvm_i32_ty],   // tfe(imm)
    []>;

  // Fully-flexible BUFFER_LOAD_DWORD_* except for the ADDR64 bit, which is not exposed
  def int_SI_buffer_load_dword : Intrinsic <
    [llvm_anyint_ty], // vdata(VGPR), overloaded for types i32, v2i32, v4i32
    [llvm_anyint_ty,  // rsrc(SGPR)
     llvm_anyint_ty,  // vaddr(VGPR)
     llvm_i32_ty,     // soffset(SGPR)
     llvm_i32_ty,     // inst_offset(imm)
     llvm_i32_ty,     // offen(imm)
     llvm_i32_ty,     // idxen(imm)
     llvm_i32_ty,     // glc(imm)
     llvm_i32_ty,     // slc(imm)
     llvm_i32_ty],    // tfe(imm)
    [IntrReadArgMem]>;

  def int_SI_sendmsg : Intrinsic <[], [llvm_i32_ty, llvm_i32_ty], []>;

  // Fully-flexible SAMPLE instruction.
  class SampleRaw : Intrinsic <
    [llvm_v4f32_ty],    // vdata(VGPR)
    [llvm_anyint_ty,    // vaddr(VGPR)
     llvm_v8i32_ty,     // rsrc(SGPR)
     llvm_v4i32_ty,     // sampler(SGPR)
     llvm_i32_ty,       // dmask(imm)
     llvm_i32_ty,       // unorm(imm)
     llvm_i32_ty,       // r128(imm)
     llvm_i32_ty,       // da(imm)
     llvm_i32_ty,       // glc(imm)
     llvm_i32_ty,       // slc(imm)
     llvm_i32_ty,       // tfe(imm)
     llvm_i32_ty],      // lwe(imm)
    [IntrNoMem]>;

  // Image instruction without a sampler.
  class Image : Intrinsic <
    [llvm_v4f32_ty],    // vdata(VGPR)
    [llvm_anyint_ty,    // vaddr(VGPR)
     llvm_v8i32_ty,     // rsrc(SGPR)
     llvm_i32_ty,       // dmask(imm)
     llvm_i32_ty,       // unorm(imm)
     llvm_i32_ty,       // r128(imm)
     llvm_i32_ty,       // da(imm)
     llvm_i32_ty,       // glc(imm)
     llvm_i32_ty,       // slc(imm)
     llvm_i32_ty,       // tfe(imm)
     llvm_i32_ty],      // lwe(imm)
    [IntrNoMem]>;

  // Basic sample
  def int_SI_image_sample : SampleRaw;
  def int_SI_image_sample_cl : SampleRaw;
  def int_SI_image_sample_d : SampleRaw;
  def int_SI_image_sample_d_cl : SampleRaw;
  def int_SI_image_sample_l : SampleRaw;
  def int_SI_image_sample_b : SampleRaw;
  def int_SI_image_sample_b_cl : SampleRaw;
  def int_SI_image_sample_lz : SampleRaw;
  def int_SI_image_sample_cd : SampleRaw;
  def int_SI_image_sample_cd_cl : SampleRaw;

  // Sample with comparison
  def int_SI_image_sample_c : SampleRaw;
  def int_SI_image_sample_c_cl : SampleRaw;
  def int_SI_image_sample_c_d : SampleRaw;
  def int_SI_image_sample_c_d_cl : SampleRaw;
  def int_SI_image_sample_c_l : SampleRaw;
  def int_SI_image_sample_c_b : SampleRaw;
  def int_SI_image_sample_c_b_cl : SampleRaw;
  def int_SI_image_sample_c_lz : SampleRaw;
  def int_SI_image_sample_c_cd : SampleRaw;
  def int_SI_image_sample_c_cd_cl : SampleRaw;

  // Sample with offsets
  def int_SI_image_sample_o : SampleRaw;
  def int_SI_image_sample_cl_o : SampleRaw;
  def int_SI_image_sample_d_o : SampleRaw;
  def int_SI_image_sample_d_cl_o : SampleRaw;
  def int_SI_image_sample_l_o : SampleRaw;
  def int_SI_image_sample_b_o : SampleRaw;
  def int_SI_image_sample_b_cl_o : SampleRaw;
  def int_SI_image_sample_lz_o : SampleRaw;
  def int_SI_image_sample_cd_o : SampleRaw;
  def int_SI_image_sample_cd_cl_o : SampleRaw;

  // Sample with comparison and offsets
  def int_SI_image_sample_c_o : SampleRaw;
  def int_SI_image_sample_c_cl_o : SampleRaw;
  def int_SI_image_sample_c_d_o : SampleRaw;
  def int_SI_image_sample_c_d_cl_o : SampleRaw;
  def int_SI_image_sample_c_l_o : SampleRaw;
  def int_SI_image_sample_c_b_o : SampleRaw;
  def int_SI_image_sample_c_b_cl_o : SampleRaw;
  def int_SI_image_sample_c_lz_o : SampleRaw;
  def int_SI_image_sample_c_cd_o : SampleRaw;
  def int_SI_image_sample_c_cd_cl_o : SampleRaw;

  // Basic gather4
  def int_SI_gather4 : SampleRaw;
  def int_SI_gather4_cl : SampleRaw;
  def int_SI_gather4_l : SampleRaw;
  def int_SI_gather4_b : SampleRaw;
  def int_SI_gather4_b_cl : SampleRaw;
  def int_SI_gather4_lz : SampleRaw;

  // Gather4 with comparison
  def int_SI_gather4_c : SampleRaw;
  def int_SI_gather4_c_cl : SampleRaw;
  def int_SI_gather4_c_l : SampleRaw;
  def int_SI_gather4_c_b : SampleRaw;
  def int_SI_gather4_c_b_cl : SampleRaw;
  def int_SI_gather4_c_lz : SampleRaw;

  // Gather4 with offsets
  def int_SI_gather4_o : SampleRaw;
  def int_SI_gather4_cl_o : SampleRaw;
  def int_SI_gather4_l_o : SampleRaw;
  def int_SI_gather4_b_o : SampleRaw;
  def int_SI_gather4_b_cl_o : SampleRaw;
  def int_SI_gather4_lz_o : SampleRaw;

  // Gather4 with comparison and offsets
  def int_SI_gather4_c_o : SampleRaw;
  def int_SI_gather4_c_cl_o : SampleRaw;
  def int_SI_gather4_c_l_o : SampleRaw;
  def int_SI_gather4_c_b_o : SampleRaw;
  def int_SI_gather4_c_b_cl_o : SampleRaw;
  def int_SI_gather4_c_lz_o : SampleRaw;

  def int_SI_getlod : SampleRaw;

  // Image instrinsics.
  def int_SI_image_load : Image;
  def int_SI_image_load_mip : Image;
  def int_SI_getresinfo : Image;

  // Deprecated image and sample intrinsics.
  class Sample : Intrinsic <[llvm_v4f32_ty], [llvm_anyvector_ty, llvm_v32i8_ty, llvm_anyint_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_SI_sample : Sample;
  def int_SI_sampleb : Sample;
  def int_SI_sampled : Sample;
  def int_SI_samplel : Sample;
  def int_SI_imageload : Intrinsic <[llvm_v4i32_ty], [llvm_anyvector_ty, llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_SI_resinfo : Intrinsic <[llvm_v4i32_ty], [llvm_i32_ty, llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;

  /* Interpolation Intrinsics */

  def int_SI_fs_constant : Intrinsic <[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_SI_fs_interp : Intrinsic <[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

  /* Control flow Intrinsics */

  def int_SI_if : Intrinsic<[llvm_i64_ty], [llvm_i1_ty, llvm_empty_ty], []>;
  def int_SI_else : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_empty_ty], []>;
  def int_SI_break : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], []>;
  def int_SI_if_break : Intrinsic<[llvm_i64_ty], [llvm_i1_ty, llvm_i64_ty], []>;
  def int_SI_else_break : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], []>;
  def int_SI_loop : Intrinsic<[], [llvm_i64_ty, llvm_empty_ty], []>;
  def int_SI_end_cf : Intrinsic<[], [llvm_i64_ty], []>;
}
