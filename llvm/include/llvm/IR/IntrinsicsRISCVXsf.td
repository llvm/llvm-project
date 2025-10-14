//===- IntrinsicsRISCVXsf.td - SiFive intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the SiFive vendor intrinsics for RISC-V.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {
  // Output: (vector_out)
  // Input: (bit<27-26>, bit<24-20>, scalar_in, vl) or
  class RISCVSFCustomVC_X<bit HasSE, bit ImmScalar>
        : Intrinsic<[llvm_anyvector_ty],
                    [llvm_anyint_ty, LLVMMatchType<1>, llvm_any_ty, llvm_anyint_ty],
                    !listconcat([IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>],    // bit<27-26> and bit<24-20>
                                !if(ImmScalar, [ImmArg<ArgIndex<2>>], []),                // ScalarOperand
                                !if(HasSE, [IntrHasSideEffects], []))>,
          RISCVVIntrinsic {
    let ScalarOperand = !cond(ImmScalar: NoScalarOperand,
                              true: 2);
    let VLOperand = 3;
  }
  // Output: ()
  // Input: (bit<27-26>, bit<24-20>, bit<11-7>, scalar_in, sew, log2lmul, vl)
  class RISCVSFCustomVC_X_WO_Suffix<bit ImmScalar>
        : Intrinsic<[],
                    [llvm_anyint_ty, LLVMMatchType<0>, LLVMMatchType<0>,
                     llvm_any_ty, LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty],
                    !listconcat([IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>,
                                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>,
                                 ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>],    // bit<27-26>, bit<24-20>, bit<11-7>, sew, log2lmul
                                !if(ImmScalar, [ImmArg<ArgIndex<3>>], []))>,   // ScalarOperand
          RISCVVIntrinsic {
    let ScalarOperand = !cond(ImmScalar: NoScalarOperand,
                              true: 3);
    let VLOperand = 6;
  }
  // Output: (vector_out) or ()
  // Input: (bit<27-26>, vector_in, vector_in/scalar_in, vl) or
  //        (bit<27-26>, bit<11-7>, vector_in, vector_in/scalar_in, vl)
  class RISCVSFCustomVC_XV<bit HasDst, bit HasSE, bit ImmScalar>
        : Intrinsic<!if(HasDst, [llvm_anyvector_ty], []),
                    !listconcat(!if(HasDst, [llvm_anyint_ty, llvm_anyvector_ty],
                                            [llvm_anyint_ty, LLVMMatchType<0>, llvm_anyvector_ty]),
                                [llvm_any_ty, llvm_anyint_ty]),
                    !listconcat([IntrNoMem, ImmArg<ArgIndex<0>>],                        // bit<27-26>
                                !if(HasDst, [], [ImmArg<ArgIndex<1>>]),                  // Vd or bit<11-7>
                                !if(ImmScalar, !if(HasDst, [ImmArg<ArgIndex<2>>],
                                                           [ImmArg<ArgIndex<3>>]), []),  // ScalarOperand
                                !if(HasSE, [IntrHasSideEffects], []))>,
          RISCVVIntrinsic {
    let ScalarOperand = !cond(ImmScalar: NoScalarOperand,
                              HasDst: 2,
                              true: 3);
    let VLOperand = !if(HasDst, 3, 4);
  }
  // Output: (vector_out) or ()
  // Input: (bit<27-26>, passthru, vector_in, vector_in/scalar_in, vl) or
  //        (bit<27-26>, vector_in, vector_in, vector_in/scalar_in, vl)
  class RISCVSFCustomVC_XVV<bit HasDst, bit HasSE, bit ImmScalar>
        : Intrinsic<!if(HasDst, [llvm_anyvector_ty], []),
                    !listconcat(!if(HasDst, [llvm_anyint_ty, llvm_anyvector_ty, llvm_anyvector_ty],
                                            [llvm_anyint_ty, llvm_anyvector_ty, llvm_anyvector_ty]),
                                [llvm_any_ty, llvm_anyint_ty]),
                    !listconcat([IntrNoMem, ImmArg<ArgIndex<0>>],                        // bit<27-26>
                                !if(ImmScalar, [ImmArg<ArgIndex<3>>], []),               // ScalarOperand
                                !if(HasSE, [IntrHasSideEffects], []))>,
          RISCVVIntrinsic {
    let ScalarOperand = !if(ImmScalar, NoScalarOperand, 3);
    let VLOperand = 4;
  }
  // Output: (wvector_out) or ()
  // Input: (bit<27-26>, passthru, vector_in, vector_in/scalar_in, vl) or
  //        (bit<27-26>, wvector_in, vector_in, vector_in/scalar_in, vl)
  class RISCVSFCustomVC_XVW<bit HasDst, bit HasSE, bit ImmScalar>
        : Intrinsic<!if(HasDst, [llvm_anyvector_ty], []),
                    !listconcat(!if(HasDst, [llvm_anyint_ty, llvm_anyvector_ty, llvm_anyvector_ty],
                                            [llvm_anyint_ty, llvm_anyvector_ty, llvm_anyvector_ty]),
                                [llvm_any_ty, llvm_anyint_ty]),
                    !listconcat([IntrNoMem, ImmArg<ArgIndex<0>>],                        // bit<27-26>
                                !if(ImmScalar, [ImmArg<ArgIndex<3>>], []),               // ScalarOperand
                                !if(HasSE, [IntrHasSideEffects], []))>,
          RISCVVIntrinsic {
    let ScalarOperand = !if(ImmScalar, NoScalarOperand, 3);
    let VLOperand = 4;
  }

  multiclass RISCVSFCustomVC_X<list<string> type> {
    foreach t = type in {
      defvar ImmScalar = !eq(t, "i");
      def "int_riscv_sf_vc_" # t # "_se"   : RISCVSFCustomVC_X_WO_Suffix<ImmScalar=ImmScalar>;
      def "int_riscv_sf_vc_v_" # t # "_se" : RISCVSFCustomVC_X<HasSE=1, ImmScalar=ImmScalar>;
      def "int_riscv_sf_vc_v_" # t         : RISCVSFCustomVC_X<HasSE=0, ImmScalar=ImmScalar>;
    }
  }

  multiclass RISCVSFCustomVC_XV<list<string> type> {
    foreach t = type in {
      defvar ImmScalar = !eq(t, "i");
      def "int_riscv_sf_vc_" # t # "v_se"   : RISCVSFCustomVC_XV<HasDst=0, HasSE=1, ImmScalar=ImmScalar>;
      def "int_riscv_sf_vc_v_" # t # "v_se" : RISCVSFCustomVC_XV<HasDst=1, HasSE=1, ImmScalar=ImmScalar>;
      def "int_riscv_sf_vc_v_" # t # "v"    : RISCVSFCustomVC_XV<HasDst=1, HasSE=0, ImmScalar=ImmScalar>;
    }
  }

  multiclass RISCVSFCustomVC_XVV<list<string> type> {
    foreach t = type in {
      defvar ImmScalar = !eq(t, "i");
      def "int_riscv_sf_vc_" # t # "vv_se"   : RISCVSFCustomVC_XVV<HasDst=0, HasSE=1, ImmScalar=ImmScalar>;
      def "int_riscv_sf_vc_v_" # t # "vv_se" : RISCVSFCustomVC_XVV<HasDst=1, HasSE=1, ImmScalar=ImmScalar>;
      def "int_riscv_sf_vc_v_" # t # "vv"    : RISCVSFCustomVC_XVV<HasDst=1, HasSE=0, ImmScalar=ImmScalar>;
    }
  }

  multiclass RISCVSFCustomVC_XVW<list<string> type> {
    foreach t = type in {
      defvar ImmScalar = !eq(t, "i");
      def "int_riscv_sf_vc_" # t # "vw_se"   : RISCVSFCustomVC_XVW<HasDst=0, HasSE=1, ImmScalar=ImmScalar>;
      def "int_riscv_sf_vc_v_" # t # "vw_se" : RISCVSFCustomVC_XVW<HasDst=1, HasSE=1, ImmScalar=ImmScalar>;
      def "int_riscv_sf_vc_v_" # t # "vw"    : RISCVSFCustomVC_XVW<HasDst=1, HasSE=0, ImmScalar=ImmScalar>;
    }
  }

  class RISCVSFCustomVMACC
      : DefaultAttrsIntrinsic< [llvm_anyvector_ty],
                   [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty,
                    llvm_anyint_ty, LLVMMatchType<3>],
                   [ImmArg<ArgIndex<4>>, IntrNoMem] >, RISCVVIntrinsic {
    let VLOperand = 3;
  }

  // Input: (passthru, vector_in, scalar_in, frm, vl)
  class RISCVSFCustomVFNRCLIPUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyfloat_ty, LLVMVectorElementType<1>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<3>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }

  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, frm, vl, policy)
  class RISCVSFCustomVFNRCLIPMasked
       : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, llvm_anyfloat_ty, LLVMVectorElementType<1>,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                    LLVMMatchType<2>, LLVMMatchType<2>],
                   [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<6>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 5;
  }

  multiclass RISCVSFCustomVFNRCLIP {
    def NAME : RISCVSFCustomVFNRCLIPUnMasked;
    def NAME # "_mask" : RISCVSFCustomVFNRCLIPMasked;
  }

  defm "" : RISCVSFCustomVC_X<["x", "i"]>;
  defm "" : RISCVSFCustomVC_XV<["x", "i", "v", "f"]>;
  defm "" : RISCVSFCustomVC_XVV<["x", "i", "v", "f"]>;
  defm "" : RISCVSFCustomVC_XVW<["x", "i", "v", "f"]>;

  // XSfvqmaccdod
  def int_riscv_sf_vqmaccu_2x8x2  : RISCVSFCustomVMACC;
  def int_riscv_sf_vqmacc_2x8x2   : RISCVSFCustomVMACC;
  def int_riscv_sf_vqmaccus_2x8x2 : RISCVSFCustomVMACC;
  def int_riscv_sf_vqmaccsu_2x8x2 : RISCVSFCustomVMACC;

  // XSfvqmaccqoq
  def int_riscv_sf_vqmaccu_4x8x4  : RISCVSFCustomVMACC;
  def int_riscv_sf_vqmacc_4x8x4   : RISCVSFCustomVMACC;
  def int_riscv_sf_vqmaccus_4x8x4 : RISCVSFCustomVMACC;
  def int_riscv_sf_vqmaccsu_4x8x4 : RISCVSFCustomVMACC;

  // XSfvfwmaccqqq
  def int_riscv_sf_vfwmacc_4x4x4 : RISCVSFCustomVMACC;

  // XSfvfnrclipxfqf
  defm int_riscv_sf_vfnrclip_x_f_qf : RISCVSFCustomVFNRCLIP;
  defm int_riscv_sf_vfnrclip_xu_f_qf : RISCVSFCustomVFNRCLIP;

  // XSfmm
  // Output: (output_len)
  // Input: (input_len, vsew, twiden)
  class RISCVSFVSet
      : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                              [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                              [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, IntrNoMem]>;

  // Input: (tss, base, tn)
  class RISCVSFTileLoad
      : DefaultAttrsIntrinsic<[],
                              [llvm_anyint_ty, llvm_ptr_ty, LLVMMatchType<0>],
                              [NoCapture<ArgIndex<1>>, IntrHasSideEffects]>,
        RISCVVIntrinsic;

  // Input: (tss, base, tn)
  class RISCVSFTileStore
      : DefaultAttrsIntrinsic<[],
                              [llvm_anyint_ty, llvm_ptr_ty, LLVMMatchType<0>],
                              [NoCapture<ArgIndex<1>>, IntrWriteMem,
                               IntrHasSideEffects]>,
        RISCVVIntrinsic;

  // Output: ()
  // Input: (mtd, mat1, mat2, tm, tn, tk, twiden)
  class RISCVSFCustomMatMul<bit is_float = false>
      : DefaultAttrsIntrinsic<[], [llvm_anyint_ty, llvm_anyvector_ty,
                                   !if(is_float, LLVMMatchType<1>,
                                                 llvm_anyvector_ty),
                                   LLVMMatchType<0>, LLVMMatchType<0>,
                                   LLVMMatchType<0>, LLVMMatchType<0>],
                              [IntrNoMem, IntrHasSideEffects,
                               ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<6>>]>,
        RISCVVIntrinsic;

  def int_riscv_sf_vsettnt  : RISCVSFVSet;
  def int_riscv_sf_vsettm   : RISCVSFVSet;
  def int_riscv_sf_vsettk   : RISCVSFVSet;

  def int_riscv_sf_vlte8    : RISCVSFTileLoad;
  def int_riscv_sf_vlte16   : RISCVSFTileLoad;
  def int_riscv_sf_vlte32   : RISCVSFTileLoad;
  def int_riscv_sf_vlte64   : RISCVSFTileLoad;
  def int_riscv_sf_vste8    : RISCVSFTileStore;
  def int_riscv_sf_vste16   : RISCVSFTileStore;
  def int_riscv_sf_vste32   : RISCVSFTileStore;
  def int_riscv_sf_vste64   : RISCVSFTileStore;

  // Output: (vd)
  // Input: (tss, tn)
  def int_riscv_sf_vtmv_v_t
      : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                              [llvm_anyint_ty, LLVMMatchType<1>],
                              [IntrNoMem, IntrHasSideEffects]>,
      RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // Output: ()
  // Input: (tss, vs2, tn)
  def int_riscv_sf_vtmv_t_v
      : DefaultAttrsIntrinsic<[], [LLVMMatchType<1>, llvm_anyvector_ty,
                                   llvm_anyint_ty], [IntrNoMem, IntrHasSideEffects]>,
      RISCVVIntrinsic {
    let VLOperand = 2;
  }

  foreach a = ["u", "s"] in {
    foreach b = ["u", "s"] in {
      def int_riscv_sf_mm_ # a # _ # b   : RISCVSFCustomMatMul;
    }
  }

  def int_riscv_sf_mm_f_f : RISCVSFCustomMatMul<true>;
  foreach e1 = [5, 4] in
    foreach e2 = [5, 4] in
      def int_riscv_sf_mm_e # e1 # m # !sub(7, e1) # _e # e2 # m # !sub(7, e2)
          : RISCVSFCustomMatMul<true>;

  // Output: ()
  // Input: (mtd)
  def int_riscv_sf_vtzero_t
      : DefaultAttrsIntrinsic<[],
                              [llvm_anyint_ty, LLVMMatchType<0>,LLVMMatchType<0>,
                               LLVMMatchType<0>, LLVMMatchType<0>],
                              [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>,
                               ImmArg<ArgIndex<4>>, IntrNoMem, IntrHasSideEffects]>,
        RISCVVIntrinsic;

  // Output: ()
  // Input: ()
  def int_riscv_sf_vtdiscard
      : DefaultAttrsIntrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>,
        RISCVVIntrinsic;
} // TargetPrefix = "riscv"
