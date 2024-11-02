//===-- ArmSMEIntrinsicOps.td ------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the intrinsic Ops for the ArmSME dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSME_INTRINSIC_OPS
#define ARMSME_INTRINSIC_OPS

include "ArmSME.td"

//===----------------------------------------------------------------------===//
// ArmSME Intrinsic op definitions
//===----------------------------------------------------------------------===//

def MOPPredicate : ScalableVectorOfRankAndLengthAndType<[1], [16, 8, 4, 2], [I1]>
{
  let summary = "a vector type that is a supported predicate for the SME MOP instructions";
  let description = [{
    Possible vector types:

    * `vector<[16]xi1>`
    * `vector<[8]xi1>`
    * `vector<[4]xi1>`
    * `vector<[2]xi1>`
  }];
}

// FIXME: This allows types that are not SVE vectors, e.g. vector<[16]xf32>.
def MOPVector : ScalableVectorOfRankAndLengthAndType<[1], [16, 8, 4, 2],
                                              [I8, I16, BF16, F16, F32, F64]>
{
  let summary = "a vector type that is a supported input for the SME MOP instructions";
  let description = [{
    Possible vector types:

    Integer elements:

    * `vector<[16]xi8>`
    * `vector<[8]xi16>`

    Floating point elements:

    * `vector<[8]xf16>`
    * `vector<[8]xbf16>`
    * `vector<[4]xf32>`
    * `vector<[2]xf64>`
  }];
}

class ArmSME_IntrOp<string mnemonic,
                    list<int> immArgPositions = [],
                    list<string> immArgAttrNames = [],
                    list<int> overloadedOperands = [],
                    list<Trait> traits = [], int numResults = 0,
                    list<int> overloadedResults = []>
    : LLVM_IntrOpBase<
          /*Dialect dialect=*/ArmSME_Dialect,
          /*string opName=*/"intr." # mnemonic,
          /*string enumName=*/"aarch64_sme_" # !subst(".", "_", mnemonic),
          /*list<int> overloadedResults=*/overloadedResults,
          /*list<int> overloadedOperands=*/overloadedOperands,
          /*list<Trait> traits=*/traits,
          /*int numResults=*/numResults,
          /*bit requiresAccessGroup=*/0,
          /*bit requiresAliasAnalysis=*/0,
          /*bit requiresFastmath=*/0,
          /*list<int> immArgPositions=*/immArgPositions,
          /*list<string> immArgAttrNames=*/immArgAttrNames>;

// Zero
def LLVM_aarch64_sme_zero
   : ArmSME_IntrOp<"zero",
                   /*immArgPositions=*/[0],
                   /*immArgAttrNames=*/["tile_mask"]>,
     Arguments<(ins Arg<I32Attr, "Tile mask">:$tile_mask)>;

// MOP's
class ArmSME_IntrMopOverloadedOp<string mnemonic>
    : ArmSME_IntrOp<mnemonic,
                    /*immArgPositions=*/[0],
                    /*immArgAttrNames=*/["tile_id"],
                    /*overloadedOperands=*/[4]>,
      Arguments<(ins Arg<I32Attr, "Virtual tile ID">:$tile_id,
                 Arg<MOPPredicate, "LHS predicate">:$lhs_predicate,
                 Arg<MOPPredicate, "RHS predicate">:$rhs_predicate,
                 Arg<MOPVector, "LHS vector operand">:$lhs_vector,
                 Arg<MOPVector, "RHS vector operand">:$rhs_vector)>;

def LLVM_aarch64_sme_mopa : ArmSME_IntrMopOverloadedOp<"mopa">;
def LLVM_aarch64_sme_mops : ArmSME_IntrMopOverloadedOp<"mops">;
def LLVM_aarch64_sme_mopa_wide : ArmSME_IntrMopOverloadedOp<"mopa.wide">;
def LLVM_aarch64_sme_mops_wide : ArmSME_IntrMopOverloadedOp<"mops.wide">;
def LLVM_aarch64_sme_smopa_wide : ArmSME_IntrMopOverloadedOp<"smopa.wide">;
def LLVM_aarch64_sme_smops_wide : ArmSME_IntrMopOverloadedOp<"smops.wide">;
def LLVM_aarch64_sme_umopa_wide : ArmSME_IntrMopOverloadedOp<"umopa.wide">;
def LLVM_aarch64_sme_umops_wide : ArmSME_IntrMopOverloadedOp<"umops.wide">;
def LLVM_aarch64_sme_sumopa_wide : ArmSME_IntrMopOverloadedOp<"sumopa.wide">;
def LLVM_aarch64_sme_sumops_wide : ArmSME_IntrMopOverloadedOp<"sumops.wide">;
def LLVM_aarch64_sme_usmopa_wide : ArmSME_IntrMopOverloadedOp<"usmopa.wide">;
def LLVM_aarch64_sme_usmops_wide : ArmSME_IntrMopOverloadedOp<"usmops.wide">;

class ArmSME_IntrLoadStoreOp<string mnemonic>
    : ArmSME_IntrOp<mnemonic,
                    /*immArgPositions=*/[2],
                    /*immArgAttrNames=*/["tile_id"]>;

// Loads
class ArmSME_IntrLoadOp<string mnemonic>
    : ArmSME_IntrLoadStoreOp<mnemonic>,
      Arguments<(ins Arg<SVEPredicate, "Vector predicate">:$predicate,
                 Arg<LLVM_AnyPointer, "Load address">:$load_address,
                 Arg<I32Attr, "Virtual tile ID">:$tile_id,
                 Arg<I32, "Tile slice">:$tile_slice_index)>;

def LLVM_aarch64_sme_ld1b_horiz : ArmSME_IntrLoadOp<"ld1b.horiz">;
def LLVM_aarch64_sme_ld1h_horiz : ArmSME_IntrLoadOp<"ld1h.horiz">;
def LLVM_aarch64_sme_ld1w_horiz : ArmSME_IntrLoadOp<"ld1w.horiz">;
def LLVM_aarch64_sme_ld1d_horiz : ArmSME_IntrLoadOp<"ld1d.horiz">;
def LLVM_aarch64_sme_ld1q_horiz : ArmSME_IntrLoadOp<"ld1q.horiz">;
def LLVM_aarch64_sme_ld1b_vert : ArmSME_IntrLoadOp<"ld1b.vert">;
def LLVM_aarch64_sme_ld1h_vert : ArmSME_IntrLoadOp<"ld1h.vert">;
def LLVM_aarch64_sme_ld1w_vert : ArmSME_IntrLoadOp<"ld1w.vert">;
def LLVM_aarch64_sme_ld1d_vert : ArmSME_IntrLoadOp<"ld1d.vert">;
def LLVM_aarch64_sme_ld1q_vert : ArmSME_IntrLoadOp<"ld1q.vert">;

// Stores
class ArmSME_IntrStoreOp<string mnemonic>
    : ArmSME_IntrLoadStoreOp<mnemonic>,
      Arguments<(ins Arg<SVEPredicate, "Vector predicate">:$predicate,
                 Arg<LLVM_AnyPointer, "Store address", [MemWrite]>:$store_address,
                 Arg<I32Attr, "Virtual tile ID">:$tile_id,
                 Arg<I32, "Tile slice">:$tile_slice_index)>;

def LLVM_aarch64_sme_st1b_horiz : ArmSME_IntrStoreOp<"st1b.horiz">;
def LLVM_aarch64_sme_st1h_horiz : ArmSME_IntrStoreOp<"st1h.horiz">;
def LLVM_aarch64_sme_st1w_horiz : ArmSME_IntrStoreOp<"st1w.horiz">;
def LLVM_aarch64_sme_st1d_horiz : ArmSME_IntrStoreOp<"st1d.horiz">;
def LLVM_aarch64_sme_st1q_horiz : ArmSME_IntrStoreOp<"st1q.horiz">;
def LLVM_aarch64_sme_st1b_vert : ArmSME_IntrStoreOp<"st1b.vert">;
def LLVM_aarch64_sme_st1h_vert : ArmSME_IntrStoreOp<"st1h.vert">;
def LLVM_aarch64_sme_st1w_vert : ArmSME_IntrStoreOp<"st1w.vert">;
def LLVM_aarch64_sme_st1d_vert : ArmSME_IntrStoreOp<"st1d.vert">;
def LLVM_aarch64_sme_st1q_vert : ArmSME_IntrStoreOp<"st1q.vert">;

def LLVM_aarch64_sme_str
    : ArmSME_IntrOp<"str">,
      Arguments<(ins Arg<I32, "Index">:$index,
                 Arg<LLVM_AnyPointer, "Store address", [MemWrite]>:$store_address,
                 Arg<I32, "Offset">:$offset)>;

// Vector to tile slice
class LLVM_aarch64_sme_write<string direction>
    : ArmSME_IntrOp<"write." # direction,
                    /*immArgPositions=*/[0],
                    /*immArgAttrNames=*/["tile_id"],
                    /*overloadedOperands=*/[3],
                    [AllShapesMatch<["predicate", "vector"]>]>,
      Arguments<(ins Arg<I32Attr, "Virtual tile ID">:$tile_id,
                     Arg<I32, "Tile slice">:$tile_slice_index,
                     Arg<SVEPredicate, "Vector predicate">:$predicate,
                     Arg<SVEVector, "Vector operand">:$vector)>;

// Tile slice to vector
class LLVM_aarch64_sme_read<string direction>
    : ArmSME_IntrOp<"read." # direction,
                    /*immArgPositions=*/[2],
                    /*immArgAttrNames=*/["tile_id"],
                    /*overloadedOperands=*/[],
                    [AllShapesMatch<["vector", "predicate", "res"]>,
                     AllElementTypesMatch<["vector", "res"]>],
                    /*numResults=*/1, /*overloadedResults=*/[0]>,
      Arguments<(ins Arg<SVEVector, "Vector operand">:$vector,
                     Arg<SVEPredicate, "Vector predicate">:$predicate,
                     Arg<I32Attr, "Virtual tile ID">:$tile_id,
                     Arg<I32, "Tile slice">:$tile_slice_index)>;

def LLVM_aarch64_sme_write_horiz : LLVM_aarch64_sme_write<"horiz">;
def LLVM_aarch64_sme_write_vert : LLVM_aarch64_sme_write<"vert">;

def LLVM_aarch64_sme_read_horiz : LLVM_aarch64_sme_read<"horiz">;
def LLVM_aarch64_sme_read_vert : LLVM_aarch64_sme_read<"vert">;

class ArmSME_IntrCountOp<string mnemonic>
    : ArmSME_IntrOp<mnemonic,
                    /*immArgPositions=*/[],
                    /*immArgAttrNames=*/[],
                    /*overloadedOperands=*/[],
                    /*traits*/[PredOpTrait<"`res` is i64", TypeIsPred<"res", I64>>],
                    /*numResults=*/1, /*overloadedResults=*/[]>;

def LLVM_aarch64_sme_cntsb : ArmSME_IntrCountOp<"cntsb">;
def LLVM_aarch64_sme_cntsh : ArmSME_IntrCountOp<"cntsh">;
def LLVM_aarch64_sme_cntsw : ArmSME_IntrCountOp<"cntsw">;
def LLVM_aarch64_sme_cntsd : ArmSME_IntrCountOp<"cntsd">;

#endif // ARMSME_INTRINSIC_OPS
