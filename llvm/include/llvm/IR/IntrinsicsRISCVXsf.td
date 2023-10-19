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

class VCIXSuffix<string range> {
  list<string> suffix = !cond(!eq(range, "c"): ["e8mf8", "e8mf4", "e8mf2", "e8m1", "e8m2", "e8m4", "e8m8"],
                              !eq(range, "s"): ["e16mf4", "e16mf2", "e16m1", "e16m2", "e16m4", "e16m8"],
                              !eq(range, "i"): ["e32mf2", "e32m1", "e32m2", "e32m4", "e32m8"],
                              !eq(range, "l"): ["e64m1", "e64m2", "e64m4", "e64m8"]);
}

let TargetPrefix = "riscv" in {
  // Output: (vector_out) or ()
  // Input: (bit<27-26>, bit<24-20>, scalar_in, vl) or
  //        (bit<27-26>, bit<24-20>, bit<11-7>, scalar_in, vl)
  class RISCVSFCustomVC_X<bit HasDst, bit HasSE, bit ImmScalar>
        : Intrinsic<!if(HasDst, [llvm_anyvector_ty], []),
                    !listconcat(!if(HasDst, [llvm_anyint_ty, LLVMMatchType<1>],
                                            [llvm_anyint_ty, LLVMMatchType<0>, LLVMMatchType<0>]),
                                [llvm_any_ty, llvm_anyint_ty]),
                    !listconcat([IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>],    // bit<27-26> and bit<24-20>
                                !if(HasDst, [], [ImmArg<ArgIndex<2>>]),                   // Vd or bit<11-7>
                                !if(ImmScalar, !if(HasDst, [ImmArg<ArgIndex<2>>],
                                                           [ImmArg<ArgIndex<3>>]), []),   // ScalarOperand
                                !if(HasSE, [IntrHasSideEffects], []))>,
          RISCVVIntrinsic {
    let ScalarOperand = !cond(ImmScalar: NoScalarOperand,
                              HasDst: 2,
                              true: 3);
    let VLOperand = !if(HasDst, 3, 4);
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
      defvar range = ["c", "s", "i", "l"];
      foreach r = range in {
        foreach s = VCIXSuffix<r>.suffix in {
          def "int_riscv_sf_vc_" # t # "_se_" # s : RISCVSFCustomVC_X<HasDst=0, HasSE=1, ImmScalar=ImmScalar>;
        }
      }
      def "int_riscv_sf_vc_v_" # t # "_se" : RISCVSFCustomVC_X<HasDst=1, HasSE=1, ImmScalar=ImmScalar>;
      def "int_riscv_sf_vc_v_" # t         : RISCVSFCustomVC_X<HasDst=1, HasSE=0, ImmScalar=ImmScalar>;
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

  defm "" : RISCVSFCustomVC_X<["x", "i"]>;
  defm "" : RISCVSFCustomVC_XV<["x", "i", "v", "f"]>;
  defm "" : RISCVSFCustomVC_XVV<["x", "i", "v", "f"]>;
  defm "" : RISCVSFCustomVC_XVW<["x", "i", "v", "f"]>;
} // TargetPrefix = "riscv"
