//===-- IntrinsicsMBlaze.td - Defines MBlaze intrinsics ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the MicroBlaze-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Definitions for all MBlaze intrinsics.
//

// MBlaze intrinsic classes.
let TargetPrefix = "mblaze", isTarget = 1 in {
  class MBFSL_Get_Intrinsic : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], []>;

  class MBFSL_Put_Intrinsic : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

  class MBFSL_PutT_Intrinsic : Intrinsic<[], [llvm_i32_ty], []>;
}

//===----------------------------------------------------------------------===//
// MicroBlaze FSL Get Intrinsic Definitions.
//

def int_mblaze_fsl_get      : GCCBuiltin<"__builtin_mblaze_fsl_get">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_aget     : GCCBuiltin<"__builtin_mblaze_fsl_aget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_cget     : GCCBuiltin<"__builtin_mblaze_fsl_cget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_caget    : GCCBuiltin<"__builtin_mblaze_fsl_caget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_eget     : GCCBuiltin<"__builtin_mblaze_fsl_eget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_eaget    : GCCBuiltin<"__builtin_mblaze_fsl_eaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_ecget    : GCCBuiltin<"__builtin_mblaze_fsl_ecget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_ecaget   : GCCBuiltin<"__builtin_mblaze_fsl_ecaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_nget     : GCCBuiltin<"__builtin_mblaze_fsl_nget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_naget    : GCCBuiltin<"__builtin_mblaze_fsl_naget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_ncget    : GCCBuiltin<"__builtin_mblaze_fsl_ncget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_ncaget   : GCCBuiltin<"__builtin_mblaze_fsl_ncaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_neget    : GCCBuiltin<"__builtin_mblaze_fsl_neget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_neaget   : GCCBuiltin<"__builtin_mblaze_fsl_neaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_necget   : GCCBuiltin<"__builtin_mblaze_fsl_necget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_necaget  : GCCBuiltin<"__builtin_mblaze_fsl_necaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tget     : GCCBuiltin<"__builtin_mblaze_fsl_tget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_taget    : GCCBuiltin<"__builtin_mblaze_fsl_taget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tcget    : GCCBuiltin<"__builtin_mblaze_fsl_tcget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tcaget   : GCCBuiltin<"__builtin_mblaze_fsl_tcaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_teget    : GCCBuiltin<"__builtin_mblaze_fsl_teget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_teaget   : GCCBuiltin<"__builtin_mblaze_fsl_teaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tecget   : GCCBuiltin<"__builtin_mblaze_fsl_tecget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tecaget  : GCCBuiltin<"__builtin_mblaze_fsl_tecaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tnget    : GCCBuiltin<"__builtin_mblaze_fsl_tnget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tnaget   : GCCBuiltin<"__builtin_mblaze_fsl_tnaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tncget   : GCCBuiltin<"__builtin_mblaze_fsl_tncget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tncaget  : GCCBuiltin<"__builtin_mblaze_fsl_tncaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tneget   : GCCBuiltin<"__builtin_mblaze_fsl_tneget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tneaget  : GCCBuiltin<"__builtin_mblaze_fsl_tneaget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tnecget  : GCCBuiltin<"__builtin_mblaze_fsl_tnecget">,
                              MBFSL_Get_Intrinsic;
def int_mblaze_fsl_tnecaget : GCCBuiltin<"__builtin_mblaze_fsl_tnecaget">,
                              MBFSL_Get_Intrinsic;

//===----------------------------------------------------------------------===//
// MicroBlaze FSL Put Intrinsic Definitions.
//

def int_mblaze_fsl_put     : GCCBuiltin<"__builtin_mblaze_fsl_put">,
                             MBFSL_Put_Intrinsic;
def int_mblaze_fsl_aput    : GCCBuiltin<"__builtin_mblaze_fsl_aput">,
                             MBFSL_Put_Intrinsic;
def int_mblaze_fsl_cput    : GCCBuiltin<"__builtin_mblaze_fsl_cput">,
                             MBFSL_Put_Intrinsic;
def int_mblaze_fsl_caput   : GCCBuiltin<"__builtin_mblaze_fsl_caput">,
                             MBFSL_Put_Intrinsic;
def int_mblaze_fsl_nput    : GCCBuiltin<"__builtin_mblaze_fsl_nput">,
                             MBFSL_Put_Intrinsic;
def int_mblaze_fsl_naput   : GCCBuiltin<"__builtin_mblaze_fsl_naput">,
                             MBFSL_Put_Intrinsic;
def int_mblaze_fsl_ncput   : GCCBuiltin<"__builtin_mblaze_fsl_ncput">,
                             MBFSL_Put_Intrinsic;
def int_mblaze_fsl_ncaput  : GCCBuiltin<"__builtin_mblaze_fsl_ncaput">,
                             MBFSL_Put_Intrinsic;
def int_mblaze_fsl_tput    : GCCBuiltin<"__builtin_mblaze_fsl_tput">,
                             MBFSL_PutT_Intrinsic;
def int_mblaze_fsl_taput   : GCCBuiltin<"__builtin_mblaze_fsl_taput">,
                             MBFSL_PutT_Intrinsic;
def int_mblaze_fsl_tcput   : GCCBuiltin<"__builtin_mblaze_fsl_tcput">,
                             MBFSL_PutT_Intrinsic;
def int_mblaze_fsl_tcaput  : GCCBuiltin<"__builtin_mblaze_fsl_tcaput">,
                             MBFSL_PutT_Intrinsic;
def int_mblaze_fsl_tnput   : GCCBuiltin<"__builtin_mblaze_fsl_tnput">,
                             MBFSL_PutT_Intrinsic;
def int_mblaze_fsl_tnaput  : GCCBuiltin<"__builtin_mblaze_fsl_tnaput">,
                             MBFSL_PutT_Intrinsic;
def int_mblaze_fsl_tncput  : GCCBuiltin<"__builtin_mblaze_fsl_tncput">,
                             MBFSL_PutT_Intrinsic;
def int_mblaze_fsl_tncaput : GCCBuiltin<"__builtin_mblaze_fsl_tncaput">,
                             MBFSL_PutT_Intrinsic;
