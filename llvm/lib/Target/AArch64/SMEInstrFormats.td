//=-- SMEInstrFormats.td -  AArch64 SME Instruction classes -*- tablegen -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AArch64 Scalable Matrix Extension (SME) Instruction Class Definitions.
//
//===----------------------------------------------------------------------===//

def imm_to_tile8   : ComplexPattern<i64, 1, "ImmToTile<AArch64::ZAB0>", []>;
def imm_to_tile16  : ComplexPattern<i64, 1, "ImmToTile<AArch64::ZAH0>", []>;
def imm_to_tile32  : ComplexPattern<i64, 1, "ImmToTile<AArch64::ZAS0>", []>;
def imm_to_tile64  : ComplexPattern<i64, 1, "ImmToTile<AArch64::ZAD0>", []>;
def imm_to_tile128 : ComplexPattern<i64, 1, "ImmToTile<AArch64::ZAQ0>", []>;

def tileslice8   : ComplexPattern<i32 , 2, "SelectSMETileSlice<4>", []>;
def tileslice16  : ComplexPattern<i32 , 2, "SelectSMETileSlice<3>", []>;
def tileslice32  : ComplexPattern<i32 , 2, "SelectSMETileSlice<2>", []>;
def tileslice64  : ComplexPattern<i32 , 2, "SelectSMETileSlice<1>", []>;
def tileslice128 : ComplexPattern<i32 , 2, "SelectSMETileSlice<0>", []>; // nop

def am_sme_indexed_b4 :ComplexPattern<iPTR, 2, "SelectAddrModeIndexedSVE<0,15>", [], [SDNPWantRoot]>;

//===----------------------------------------------------------------------===//
// SME Outer Products
//===----------------------------------------------------------------------===//

class sme_fp_outer_product_inst<bit S, bit sz, MatrixTileOperand za_ty,
                                ZPRRegOp zpr_ty, string mnemonic>
    : I<(outs za_ty:$ZAda),
      (ins za_ty:$_ZAda, PPR3bAny:$Pn, PPR3bAny:$Pm, zpr_ty:$Zn, zpr_ty:$Zm),
        mnemonic, "\t$ZAda, $Pn/m, $Pm/m, $Zn, $Zm",
        "", []>,
      Sched<[]> {
  bits<5> Zm;
  bits<3> Pm;
  bits<3> Pn;
  bits<5> Zn;
  let Inst{31-23} = 0b100000001;
  let Inst{22}    = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15-13} = Pm;
  let Inst{12-10} = Pn;
  let Inst{9-5}   = Zn;
  let Inst{4}     = S;
  let Inst{3}     = 0b0;

  let Constraints = "$ZAda = $_ZAda";
}

class sme_outer_product_fp32<bit S, string mnemonic>
    : sme_fp_outer_product_inst<S, 0b0, TileOp32, ZPR32, mnemonic> {
  bits<2> ZAda;
  let Inst{1-0} = ZAda;
  let Inst{2}   = 0b0;
}

class sme_outer_product_fp64<bit S, string mnemonic>
    : sme_fp_outer_product_inst<S, 0b1, TileOp64, ZPR64, mnemonic> {
  bits<3> ZAda;
  let Inst{2-0} = ZAda;
}

class sme_int_outer_product_inst<bit u0, bit u1, bit S, bit sz,
                                 MatrixTileOperand za_ty, ZPRRegOp zpr_ty,
                                 string mnemonic>
    : I<(outs za_ty:$ZAda),
        (ins za_ty:$_ZAda, PPR3bAny:$Pn, PPR3bAny:$Pm, zpr_ty:$Zn, zpr_ty:$Zm),
        mnemonic, "\t$ZAda, $Pn/m, $Pm/m, $Zn, $Zm",
        "", []>,
      Sched<[]> {
  bits<5> Zm;
  bits<3> Pm;
  bits<3> Pn;
  bits<5> Zn;
  let Inst{31-25} = 0b1010000;
  let Inst{24}    = u0;
  let Inst{23}    = 0b1;
  let Inst{22}    = sz;
  let Inst{21}    = u1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = Pm;
  let Inst{12-10} = Pn;
  let Inst{9-5}   = Zn;
  let Inst{4}     = S;
  let Inst{3}     = 0b0;

  let Constraints = "$ZAda = $_ZAda";
}

class sme_int_outer_product_i32<bits<3> opc, string mnemonic>
    : sme_int_outer_product_inst<opc{2}, opc{1}, opc{0}, 0b0, TileOp32, ZPR8,
                                 mnemonic> {
  bits<2> ZAda;
  let Inst{1-0} = ZAda;
  let Inst{2}   = 0b0;
}

class sme_int_outer_product_i64<bits<3> opc, string mnemonic>
    : sme_int_outer_product_inst<opc{2}, opc{1}, opc{0}, 0b1, TileOp64, ZPR16,
                                 mnemonic> {
  bits<3> ZAda;
  let Inst{2-0} = ZAda;
}

class sme_outer_product_widening_inst<bit op, bit S, string mnemonic>
    : I<(outs TileOp32:$ZAda),
        (ins  TileOp32:$_ZAda, PPR3bAny:$Pn, PPR3bAny:$Pm, ZPR16:$Zn, ZPR16:$Zm),
        mnemonic, "\t$ZAda, $Pn/m, $Pm/m, $Zn, $Zm",
        "", []>,
      Sched<[]> {
  bits<5> Zm;
  bits<3> Pm;
  bits<3> Pn;
  bits<5> Zn;
  bits<2> ZAda;
  let Inst{31-22} = 0b1000000110;
  let Inst{21}    = op;
  let Inst{20-16} = Zm;
  let Inst{15-13} = Pm;
  let Inst{12-10} = Pn;
  let Inst{9-5}   = Zn;
  let Inst{4}     = S;
  let Inst{3-2}   = 0b00;
  let Inst{1-0}   = ZAda;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme_bf16_outer_product<bit S, string mnemonic> {
  def : sme_outer_product_widening_inst<0b0, S, mnemonic>;
}

multiclass sme_f16_outer_product<bit S, string mnemonic> {
  def : sme_outer_product_widening_inst<0b1, S, mnemonic>;
}

//===----------------------------------------------------------------------===//
// SME Add Vector to Tile
//===----------------------------------------------------------------------===//

class sme_add_vector_to_tile_inst<bit op, bit V, MatrixTileOperand tile_ty,
                                  ZPRRegOp zpr_ty, string mnemonic>
    : I<(outs tile_ty:$ZAda),
        (ins tile_ty:$_ZAda, PPR3bAny:$Pn, PPR3bAny:$Pm, zpr_ty:$Zn),
        mnemonic, "\t$ZAda, $Pn/m, $Pm/m, $Zn",
        "", []>, Sched<[]> {
  bits<3> Pm;
  bits<3> Pn;
  bits<5> Zn;
  let Inst{31-23} = 0b110000001;
  let Inst{22}    = op;
  let Inst{21-17} = 0b01000;
  let Inst{16}    = V;
  let Inst{15-13} = Pm;
  let Inst{12-10} = Pn;
  let Inst{9-5}   = Zn;
  let Inst{4-3}   = 0b00;

  let Constraints = "$ZAda = $_ZAda";
}

class sme_add_vector_to_tile_u32<bit V, string mnemonic>
    : sme_add_vector_to_tile_inst<0b0, V, TileOp32, ZPR32, mnemonic> {
  bits<2> ZAda;
  let Inst{2}   = 0b0;
  let Inst{1-0} = ZAda;
}

class sme_add_vector_to_tile_u64<bit V, string mnemonic>
    : sme_add_vector_to_tile_inst<0b1, V, TileOp64, ZPR64, mnemonic> {
  bits<3> ZAda;
  let Inst{2-0} = ZAda;
}

//===----------------------------------------------------------------------===//
// SME Contiguous Loads
//===----------------------------------------------------------------------===//

class sme_mem_ld_ss_base<bit Q, bit V, bits<2> msz, dag outs, dag ins,
                         string mnemonic, string argstr>
    : I<outs, ins, mnemonic, argstr, "", []>, Sched<[]> {
  bits<5> Rm;
  bits<2> Rv;
  bits<3> Pg;
  bits<5> Rn;
  let Inst{31-25} = 0b1110000;
  let Inst{24}    = Q;
  let Inst{23-22} = msz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Rm;
  let Inst{15}    = V;
  let Inst{14-13} = Rv;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4}     = 0b0;

  let mayLoad = 1;
}

class sme_mem_ld_ss_inst<bit Q, bits<2> msz, string mnemonic,
                         MatrixTileVectorOperand tile_ty, bit is_col,
                         Operand imm_ty, RegisterOperand gpr_ty>
    : sme_mem_ld_ss_base<
        Q, is_col, msz, (outs tile_ty:$ZAt),
        (ins MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, GPR64sp:$Rn,
             gpr_ty:$Rm),
        mnemonic, "\t\\{$ZAt[$Rv, $imm]\\}, $Pg/z, [$Rn, $Rm]">;

multiclass sme_mem_ss_aliases_base<string mnemonic, Instruction inst,
                                   MatrixTileVectorOperand tile_ty,
                                   Operand imm_ty, RegisterOperand gpr_ty,
                                   string pg_suffix=""> {
  def : InstAlias<mnemonic # "\t$ZAt[$Rv, $imm], $Pg" # pg_suffix # ", [$Rn, $Rm]",
                  (inst tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, GPR64sp:$Rn, gpr_ty:$Rm), 0>;
  // Default XZR offset aliases
  def : InstAlias<mnemonic # "\t\\{$ZAt[$Rv, $imm]\\}, $Pg" # pg_suffix # ", [$Rn]",
                  (inst tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, GPR64sp:$Rn, XZR), 1>;
  def : InstAlias<mnemonic # "\t$ZAt[$Rv, $imm], $Pg" # pg_suffix # ", [$Rn]",
                  (inst tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, GPR64sp:$Rn, XZR), 0>;
}

multiclass sme_mem_ss_aliases<string mnemonic, string inst, bit is_col,
                              string pg_suffix=""> {
  defm : sme_mem_ss_aliases_base<mnemonic # "b", !cast<Instruction>(inst # _B),
                                 !if(is_col, TileVectorOpV8, TileVectorOpH8),
                                 sme_elm_idx0_15, GPR64shifted8, pg_suffix>;
  defm : sme_mem_ss_aliases_base<mnemonic # "h", !cast<Instruction>(inst # _H),
                                 !if(is_col, TileVectorOpV16, TileVectorOpH16),
                                 sme_elm_idx0_7, GPR64shifted16, pg_suffix>;
  defm : sme_mem_ss_aliases_base<mnemonic # "w", !cast<Instruction>(inst # _S),
                                 !if(is_col, TileVectorOpV32, TileVectorOpH32),
                                 sme_elm_idx0_3, GPR64shifted32, pg_suffix>;
  defm : sme_mem_ss_aliases_base<mnemonic # "d", !cast<Instruction>(inst # _D),
                                 !if(is_col, TileVectorOpV64, TileVectorOpH64),
                                 sme_elm_idx0_1, GPR64shifted64, pg_suffix>;
  defm : sme_mem_ss_aliases_base<mnemonic # "q", !cast<Instruction>(inst # _Q),
                                 !if(is_col, TileVectorOpV128, TileVectorOpH128),
                                 sme_elm_idx0_0, GPR64shifted128, pg_suffix>;
}

multiclass sme_mem_ld_ss_aliases<string inst, bit is_col> {
  defm NAME : sme_mem_ss_aliases<"ld1", inst, is_col, "/z">;
}

multiclass sme_mem_ld_ss_patterns<Instruction Inst, SDPatternOperator Load,
                                  Operand tile_ty, Operand offset_ty,
                                  ComplexPattern addr,
                                  ComplexPattern tileslice> {
  // base
  def : Pat<(Load PPR3bAny:$pg, GPR64sp:$base, tile_ty:$tile,
                  MatrixIndexGPR32Op12_15:$idx),
            (Inst tile_ty:$tile, $idx, 0, $pg, $base, XZR)>;
  // reg + reg
  let AddedComplexity = 1 in {
    def : Pat<(Load PPR3bAny:$pg, (addr GPR64sp:$base, GPR64:$offset),
                    tile_ty:$tile, MatrixIndexGPR32Op12_15:$idx),
              (Inst tile_ty:$tile, $idx, 0, $pg, $base, $offset)>;
  }

  // base, tileslice
  let AddedComplexity = 1 in {
    def : Pat<(Load PPR3bAny:$pg, GPR64sp:$base, tile_ty:$tile,
                    (i32 (tileslice MatrixIndexGPR32Op12_15:$idx, offset_ty:$imm))),
              (Inst tile_ty:$tile, $idx, $imm, $pg, $base, XZR)>;
  }
  // reg + reg, tileslice
  let AddedComplexity = 2 in {
    def : Pat<(Load PPR3bAny:$pg, (addr GPR64sp:$base, GPR64:$offset),
                    tile_ty:$tile, (i32 (tileslice MatrixIndexGPR32Op12_15:$idx,
                                              offset_ty:$imm))),
              (Inst tile_ty:$tile, $idx, $imm, $pg, $base, $offset)>;
  }
}

class sme_load_pseudo
    : Pseudo<(outs), (ins i64imm:$tile, MatrixIndexGPR32Op12_15:$idx,
                          i64imm:$imm, PPR3bAny:$pg, GPR64sp:$base, GPR64:$offset), []>,
      Sched<[]> {
  // Translated to the actual instructions in AArch64ISelLowering.cpp
  let usesCustomInserter = 1;
  let mayLoad = 1;
}

multiclass sme_mem_ld_v_ss<string mnemonic, bit is_col> {
  def _B : sme_mem_ld_ss_inst<0b0, 0b00, mnemonic # "b",
                              !if(is_col, TileVectorOpV8, TileVectorOpH8),
                              is_col, sme_elm_idx0_15, GPR64shifted8> {
    bits<4> imm;
    let Inst{3-0} = imm;
  }
  def _H : sme_mem_ld_ss_inst<0b0, 0b01, mnemonic # "h",
                              !if(is_col, TileVectorOpV16, TileVectorOpH16),
                              is_col, sme_elm_idx0_7, GPR64shifted16> {
    bits<1> ZAt;
    bits<3> imm;
    let Inst{3}   = ZAt;
    let Inst{2-0} = imm;
  }
  def _S : sme_mem_ld_ss_inst<0b0, 0b10, mnemonic # "w",
                              !if(is_col, TileVectorOpV32, TileVectorOpH32),
                              is_col, sme_elm_idx0_3, GPR64shifted32> {
    bits<2> ZAt;
    bits<2> imm;
    let Inst{3-2} = ZAt;
    let Inst{1-0} = imm;
  }
  def _D : sme_mem_ld_ss_inst<0b0, 0b11, mnemonic # "d",
                              !if(is_col, TileVectorOpV64, TileVectorOpH64),
                              is_col, sme_elm_idx0_1, GPR64shifted64> {
    bits<3> ZAt;
    bits<1> imm;
    let Inst{3-1} = ZAt;
    let Inst{0}   = imm;
  }
  def _Q : sme_mem_ld_ss_inst<0b1, 0b11, mnemonic # "q",
                              !if(is_col, TileVectorOpV128, TileVectorOpH128),
                              is_col, sme_elm_idx0_0, GPR64shifted128> {
    bits<4> ZAt;
    let Inst{3-0} = ZAt;
  }

  defm : sme_mem_ld_ss_aliases<NAME, is_col>;

  // Pseudo instructions for lowering intrinsics, using immediates instead of
  // tile registers.
  def _PSEUDO_B : sme_load_pseudo;
  def _PSEUDO_H : sme_load_pseudo;
  def _PSEUDO_S : sme_load_pseudo;
  def _PSEUDO_D : sme_load_pseudo;
  def _PSEUDO_Q : sme_load_pseudo;

  defm : sme_mem_ld_ss_patterns<!cast<Instruction>(NAME # _PSEUDO_B),
                                !if(is_col, int_aarch64_sme_ld1b_vert,
                                            int_aarch64_sme_ld1b_horiz),
                                sme_elm_idx0_0, imm0_15, am_sve_regreg_lsl0,
                                tileslice8>;
  defm : sme_mem_ld_ss_patterns<!cast<Instruction>(NAME # _PSEUDO_H),
                                !if(is_col, int_aarch64_sme_ld1h_vert,
                                            int_aarch64_sme_ld1h_horiz),
                                imm0_1, imm0_7, am_sve_regreg_lsl1,
                                tileslice16>;
  defm : sme_mem_ld_ss_patterns<!cast<Instruction>(NAME # _PSEUDO_S),
                                !if(is_col, int_aarch64_sme_ld1w_vert,
                                            int_aarch64_sme_ld1w_horiz),
                                imm0_3, imm0_3, am_sve_regreg_lsl2,
                                tileslice32>;
  defm : sme_mem_ld_ss_patterns<!cast<Instruction>(NAME # _PSEUDO_D),
                                !if(is_col, int_aarch64_sme_ld1d_vert,
                                            int_aarch64_sme_ld1d_horiz),
                                imm0_7, imm0_1, am_sve_regreg_lsl3,
                                tileslice64>;
  defm : sme_mem_ld_ss_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                !if(is_col, int_aarch64_sme_ld1q_vert,
                                            int_aarch64_sme_ld1q_horiz),
                                imm0_15, sme_elm_idx0_0, am_sve_regreg_lsl4,
                                tileslice128>;
}

multiclass sme_mem_ld_ss<string mnemonic> {
  defm _H : sme_mem_ld_v_ss<mnemonic, /*is_col=*/0b0>;
  defm _V : sme_mem_ld_v_ss<mnemonic, /*is_col=*/0b1>;
}

//===----------------------------------------------------------------------===//
// SME Contiguous Stores
//===----------------------------------------------------------------------===//

class sme_mem_st_ss_base<bit Q, bit V, bits<2> msz, dag ins,
                         string mnemonic, string argstr>
    : I<(outs), ins, mnemonic, argstr, "", []>, Sched<[]> {
  bits<5> Rm;
  bits<2> Rv;
  bits<3> Pg;
  bits<5> Rn;
  let Inst{31-25} = 0b1110000;
  let Inst{24}    = Q;
  let Inst{23-22} = msz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Rm;
  let Inst{15}    = V;
  let Inst{14-13} = Rv;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4}     = 0b0;

  let mayStore = 1;
  let hasSideEffects = 1;
}

class sme_mem_st_ss_inst<bit Q, bits<2> msz, string mnemonic,
                         MatrixTileVectorOperand tile_ty, bit is_col,
                         Operand imm_ty, RegisterOperand gpr_ty>
    : sme_mem_st_ss_base<
        Q, is_col, msz,
        (ins tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg,
             GPR64sp:$Rn, gpr_ty:$Rm),
        mnemonic, "\t\\{$ZAt[$Rv, $imm]\\}, $Pg, [$Rn, $Rm]">;

multiclass sme_mem_st_ss_aliases<string inst, bit is_col> {
  defm NAME : sme_mem_ss_aliases<"st1", inst, is_col>;
}

multiclass sme_mem_st_ss_patterns<Instruction Inst, SDPatternOperator Store,
                                  Operand offset_ty,
                                  ComplexPattern imm2tile,
                                  ComplexPattern addr,
                                  ComplexPattern tileslice> {
  // base
  def : Pat<(Store PPR3bAny:$pg, GPR64sp:$base, (imm2tile untyped:$tile),
                   MatrixIndexGPR32Op12_15:$idx),
            (Inst $tile, $idx, 0, $pg, $base, XZR)>;
  // reg + reg
  let AddedComplexity = 1 in {
    def : Pat<(Store PPR3bAny:$pg, (addr GPR64sp:$base, GPR64:$offset),
                     (imm2tile untyped:$tile), MatrixIndexGPR32Op12_15:$idx),
              (Inst $tile, $idx, 0, $pg, $base, $offset)>;
  }
  // base, tileslice
  let AddedComplexity = 1 in {
    def : Pat<(Store PPR3bAny:$pg, GPR64sp:$base, (imm2tile untyped:$tile),
                     (i32 (tileslice MatrixIndexGPR32Op12_15:$idx, offset_ty:$imm))),
              (Inst $tile, $idx, $imm, $pg, $base, XZR)>;
  }
  // reg + reg, tileslice
  let AddedComplexity = 2 in {
    def : Pat<(Store PPR3bAny:$pg, (addr GPR64sp:$base, GPR64:$offset),
                     (imm2tile untyped:$tile),
                     (i32 (tileslice MatrixIndexGPR32Op12_15:$idx, offset_ty:$imm))),
              (Inst $tile, $idx, $imm, $pg, $base, $offset)>;
  }
}

multiclass sme_mem_st_v_ss<string mnemonic, bit is_col> {
  def _B : sme_mem_st_ss_inst<0b0, 0b00, mnemonic # "b",
                              !if(is_col, TileVectorOpV8, TileVectorOpH8),
                              is_col, sme_elm_idx0_15, GPR64shifted8> {
    bits<4> imm;
    let Inst{3-0} = imm;
  }
  def _H : sme_mem_st_ss_inst<0b0, 0b01, mnemonic # "h",
                              !if(is_col, TileVectorOpV16, TileVectorOpH16),
                              is_col, sme_elm_idx0_7, GPR64shifted16> {
    bits<1> ZAt;
    bits<3> imm;
    let Inst{3}   = ZAt;
    let Inst{2-0} = imm;
  }
  def _S : sme_mem_st_ss_inst<0b0, 0b10, mnemonic # "w",
                              !if(is_col, TileVectorOpV32, TileVectorOpH32),
                              is_col, sme_elm_idx0_3, GPR64shifted32> {
    bits<2> ZAt;
    bits<2> imm;
    let Inst{3-2} = ZAt;
    let Inst{1-0} = imm;
  }
  def _D : sme_mem_st_ss_inst<0b0, 0b11, mnemonic # "d",
                              !if(is_col, TileVectorOpV64, TileVectorOpH64),
                              is_col, sme_elm_idx0_1, GPR64shifted64> {
    bits<3> ZAt;
    bits<1> imm;
    let Inst{3-1} = ZAt;
    let Inst{0}   = imm;
  }
  def _Q : sme_mem_st_ss_inst<0b1, 0b11, mnemonic # "q",
                              !if(is_col, TileVectorOpV128, TileVectorOpH128),
                              is_col, sme_elm_idx0_0, GPR64shifted128> {
    bits<4> ZAt;
    let Inst{3-0} = ZAt;
  }

  defm : sme_mem_st_ss_aliases<NAME, is_col>;

  defm : sme_mem_st_ss_patterns<!cast<Instruction>(NAME # _B),
                                !if(is_col, int_aarch64_sme_st1b_vert,
                                            int_aarch64_sme_st1b_horiz),
                                imm0_15, imm_to_tile8, am_sve_regreg_lsl0,
                                tileslice8>;
  defm : sme_mem_st_ss_patterns<!cast<Instruction>(NAME # _H),
                                !if(is_col, int_aarch64_sme_st1h_vert,
                                            int_aarch64_sme_st1h_horiz),
                                imm0_7, imm_to_tile16, am_sve_regreg_lsl1,
                                tileslice16>;
  defm : sme_mem_st_ss_patterns<!cast<Instruction>(NAME # _S),
                                !if(is_col, int_aarch64_sme_st1w_vert,
                                            int_aarch64_sme_st1w_horiz),
                                imm0_3, imm_to_tile32, am_sve_regreg_lsl2,
                                tileslice32>;
  defm : sme_mem_st_ss_patterns<!cast<Instruction>(NAME # _D),
                                !if(is_col, int_aarch64_sme_st1d_vert,
                                            int_aarch64_sme_st1d_horiz),
                                imm0_1, imm_to_tile64, am_sve_regreg_lsl3,
                                tileslice64>;
  defm : sme_mem_st_ss_patterns<!cast<Instruction>(NAME # _Q),
                                !if(is_col, int_aarch64_sme_st1q_vert,
                                            int_aarch64_sme_st1q_horiz),
                                sme_elm_idx0_0, imm_to_tile128,
                                am_sve_regreg_lsl4, tileslice128>;
}

multiclass sme_mem_st_ss<string mnemonic> {
  defm _H : sme_mem_st_v_ss<mnemonic, /*is_col=*/0b0>;
  defm _V : sme_mem_st_v_ss<mnemonic, /*is_col=*/0b1>;
}

//===----------------------------------------------------------------------===//
// SME Save and Restore Array
//===----------------------------------------------------------------------===//

class sme_spill_fill_base<bit isStore, dag outs, dag ins, string opcodestr>
    : I<outs, ins, opcodestr, "\t$ZAt[$Rv, $imm4], [$Rn, $offset, mul vl]", "",
        []>,
      Sched<[]> {
  bits<2> Rv;
  bits<5> Rn;
  bits<4> imm4;
  let Inst{31-22} = 0b1110000100;
  let Inst{21}    = isStore;
  let Inst{20-15} = 0b000000;
  let Inst{14-13} = Rv;
  let Inst{12-10} = 0b000;
  let Inst{9-5}   = Rn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = imm4;
}

let mayStore = 1 in
class sme_spill_inst<string opcodestr>
    : sme_spill_fill_base<0b1, (outs),
                          (ins MatrixOp:$ZAt, MatrixIndexGPR32Op12_15:$Rv,
                               sme_elm_idx0_15:$imm4, GPR64sp:$Rn,
                               imm0_15:$offset),
                          opcodestr>;
let mayLoad = 1 in
class sme_fill_inst<string opcodestr>
    : sme_spill_fill_base<0b0, (outs MatrixOp:$ZAt),
                          (ins MatrixIndexGPR32Op12_15:$Rv,
                               sme_elm_idx0_15:$imm4, GPR64sp:$Rn,
                               imm0_15:$offset),
                          opcodestr>;
multiclass sme_spill<string opcodestr> {
  def NAME : sme_spill_inst<opcodestr>;
  def : InstAlias<opcodestr # "\t$ZAt[$Rv, $imm4], [$Rn]",
                  (!cast<Instruction>(NAME) MatrixOp:$ZAt,
                   MatrixIndexGPR32Op12_15:$Rv, sme_elm_idx0_15:$imm4, GPR64sp:$Rn, 0), 1>;
  // base
  def : Pat<(int_aarch64_sme_str MatrixIndexGPR32Op12_15:$idx, GPR64sp:$base),
            (!cast<Instruction>(NAME) ZA, $idx, 0, $base, 0)>;
  // scalar + immediate (mul vl)
  let AddedComplexity = 2 in {
    def : Pat<(int_aarch64_sme_str MatrixIndexGPR32Op12_15:$idx,
                                   (am_sme_indexed_b4 GPR64sp:$base, imm0_15:$imm4)),
              (!cast<Instruction>(NAME) ZA, $idx, 0, $base, $imm4)>;
  }
}

multiclass sme_fill<string opcodestr> {
  def NAME : sme_fill_inst<opcodestr>;
  def : InstAlias<opcodestr # "\t$ZAt[$Rv, $imm4], [$Rn]",
                  (!cast<Instruction>(NAME) MatrixOp:$ZAt,
                   MatrixIndexGPR32Op12_15:$Rv, sme_elm_idx0_15:$imm4, GPR64sp:$Rn, 0), 1>;
  def NAME # _PSEUDO
      : Pseudo<(outs),
               (ins MatrixIndexGPR32Op12_15:$idx, imm0_15:$imm4,
                    GPR64sp:$base), []>,
        Sched<[]> {
    // Translated to actual instruction in AArch64ISelLowering.cpp
    let usesCustomInserter = 1;
    let mayLoad = 1;
  }
  // base
  def : Pat<(int_aarch64_sme_ldr MatrixIndexGPR32Op12_15:$idx, GPR64sp:$base),
            (!cast<Instruction>(NAME # _PSEUDO) $idx, 0, $base)>;
  // scalar + immediate (mul vl)
  let AddedComplexity = 2 in {
    def : Pat<(int_aarch64_sme_ldr MatrixIndexGPR32Op12_15:$idx,
                                   (am_sme_indexed_b4 GPR64sp:$base, imm0_15:$imm4)),
              (!cast<Instruction>(NAME # _PSEUDO) $idx, $imm4, $base)>;
  }
}

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//

class sme_vector_to_tile_base<bit Q, bit V, bits<2> sz, dag outs, dag ins,
                              string mnemonic, string argstr>
    : I<outs, ins, mnemonic, argstr, "", []>, Sched<[]> {
  bits<2> Rv;
  bits<3> Pg;
  bits<5> Zn;
  let Inst{31-24} = 0b11000000;
  let Inst{23-22} = sz;
  let Inst{21-17} = 0b00000;
  let Inst{16}    = Q;
  let Inst{15}    = V;
  let Inst{14-13} = Rv;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4}     = 0b0;
}

class sme_vector_to_tile_inst<bit Q, bits<2> sz, MatrixTileVectorOperand tile_ty,
                              bit is_col, Operand imm_ty, ZPRRegOp zpr_ty,
                              string mnemonic>
    : sme_vector_to_tile_base<Q, is_col, sz, (outs tile_ty:$ZAd),
        (ins tile_ty:$_ZAd, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, zpr_ty:$Zn),
        mnemonic, "\t$ZAd[$Rv, $imm], $Pg/m, $Zn">{

  let Constraints = "$ZAd = $_ZAd";
}


multiclass sme_vector_to_tile_aliases<Instruction inst,
                                      MatrixTileVectorOperand tile_ty,
                                      ZPRRegOp zpr_ty, Operand imm_ty> {
  def : InstAlias<"mov\t$ZAd[$Rv, $imm], $Pg/m, $Zn",
                  (inst tile_ty:$ZAd, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, zpr_ty:$Zn), 1>;
}

multiclass sme_vector_to_tile_patterns<Instruction inst, ValueType zpr_vt,
                                       ValueType ppr_vt, Operand imm_ty,
                                       Operand offset_ty,
                                       SDPatternOperator op,
                                       ComplexPattern tileslice> {
  def : Pat<(op imm_ty:$tile, MatrixIndexGPR32Op12_15:$idx,
                (ppr_vt PPR3bAny:$pg), (zpr_vt ZPRAny:$zn)),
            (inst imm_ty:$tile, $idx, 0, $pg, $zn)>;
  let AddedComplexity = 1 in {
    def : Pat<(op imm_ty:$tile, (i32 (tileslice MatrixIndexGPR32Op12_15:$idx,
                                                offset_ty:$imm)),
                  (ppr_vt PPR3bAny:$pg), (zpr_vt ZPRAny:$zn)),
              (inst imm_ty:$tile, $idx, $imm, $pg, $zn)>;
  }
}

class sme_mova_insert_pseudo
    : Pseudo<(outs), (ins i64imm:$tile, MatrixIndexGPR32Op12_15:$idx,
                          i64imm:$imm, PPR3bAny:$pg, ZPRAny:$zn), []>,
      Sched<[]> {
  // Translated to the actual instructions in AArch64ISelLowering.cpp
  let usesCustomInserter = 1;
}

multiclass sme_vector_v_to_tile<string mnemonic, bit is_col> {
  def _B : sme_vector_to_tile_inst<0b0, 0b00, !if(is_col, TileVectorOpV8,
                                                          TileVectorOpH8),
                                   is_col, sme_elm_idx0_15, ZPR8, mnemonic> {
    bits<4> imm;
    let Inst{3-0} = imm;
  }
  def _H : sme_vector_to_tile_inst<0b0, 0b01, !if(is_col, TileVectorOpV16,
                                                          TileVectorOpH16),
                                   is_col, sme_elm_idx0_7, ZPR16, mnemonic> {
    bits<1> ZAd;
    bits<3> imm;
    let Inst{3}   = ZAd;
    let Inst{2-0} = imm;
  }
  def _S : sme_vector_to_tile_inst<0b0, 0b10, !if(is_col, TileVectorOpV32,
                                                          TileVectorOpH32),
                                   is_col, sme_elm_idx0_3, ZPR32, mnemonic> {
    bits<2> ZAd;
    bits<2> imm;
    let Inst{3-2} = ZAd;
    let Inst{1-0} = imm;
  }
  def _D : sme_vector_to_tile_inst<0b0, 0b11, !if(is_col, TileVectorOpV64,
                                                          TileVectorOpH64),
                                   is_col, sme_elm_idx0_1, ZPR64, mnemonic> {
    bits<3> ZAd;
    bits<1> imm;
    let Inst{3-1} = ZAd;
    let Inst{0}   = imm;
  }
  def _Q : sme_vector_to_tile_inst<0b1, 0b11, !if(is_col, TileVectorOpV128,
                                                          TileVectorOpH128),
                                   is_col, sme_elm_idx0_0, ZPR128, mnemonic> {
    bits<4> ZAd;
    bits<1> imm;
    let Inst{3-0} = ZAd;
  }

  // Pseudo instructions for lowering intrinsics, using immediates instead of
  // tile registers.
  def _PSEUDO_B : sme_mova_insert_pseudo;
  def _PSEUDO_H : sme_mova_insert_pseudo;
  def _PSEUDO_S : sme_mova_insert_pseudo;
  def _PSEUDO_D : sme_mova_insert_pseudo;
  def _PSEUDO_Q : sme_mova_insert_pseudo;

  defm : sme_vector_to_tile_aliases<!cast<Instruction>(NAME # _B),
                                    !if(is_col, TileVectorOpV8,
                                                TileVectorOpH8),
                                    ZPR8, sme_elm_idx0_15>;
  defm : sme_vector_to_tile_aliases<!cast<Instruction>(NAME # _H),
                                    !if(is_col, TileVectorOpV16,
                                                TileVectorOpH16),
                                    ZPR16, sme_elm_idx0_7>;
  defm : sme_vector_to_tile_aliases<!cast<Instruction>(NAME # _S),
                                    !if(is_col, TileVectorOpV32,
                                                TileVectorOpH32),
                                    ZPR32, sme_elm_idx0_3>;
  defm : sme_vector_to_tile_aliases<!cast<Instruction>(NAME # _D),
                                    !if(is_col, TileVectorOpV64,
                                                TileVectorOpH64),
                                    ZPR64, sme_elm_idx0_1>;
  defm : sme_vector_to_tile_aliases<!cast<Instruction>(NAME # _Q),
                                    !if(is_col, TileVectorOpV128,
                                                TileVectorOpH128),
                                    ZPR128, sme_elm_idx0_0>;

  defvar op = !if(is_col, int_aarch64_sme_write_vert,
                          int_aarch64_sme_write_horiz);

  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_B),
                                     nxv16i8, nxv16i1, sme_elm_idx0_0, imm0_15,
                                     op, tileslice8>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_H),
                                     nxv8i16, nxv8i1, sme_elm_idx0_1, imm0_7,
                                     op, tileslice16>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_H),
                                     nxv8f16, nxv8i1, sme_elm_idx0_1, imm0_7,
                                     op, tileslice16>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_H),
                                     nxv8bf16, nxv8i1, sme_elm_idx0_1, imm0_7,
                                     op, tileslice16>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_S),
                                     nxv4i32, nxv4i1, sme_elm_idx0_3, imm0_3,
                                     op, tileslice32>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_S),
                                     nxv4f32, nxv4i1, sme_elm_idx0_3, imm0_3,
                                     op, tileslice32>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_D),
                                     nxv2i64, nxv2i1, sme_elm_idx0_7, imm0_1,
                                     op, tileslice64>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_D),
                                     nxv2f64, nxv2i1, sme_elm_idx0_7, imm0_1,
                                     op, tileslice64>;

  defvar opq = !if(is_col, int_aarch64_sme_writeq_vert,
                           int_aarch64_sme_writeq_horiz);

  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                     nxv16i8, nxv16i1, sme_elm_idx0_15,
                                     sme_elm_idx0_0, opq, tileslice128>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                     nxv8i16, nxv8i1, sme_elm_idx0_15,
                                     sme_elm_idx0_0, opq, tileslice128>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                     nxv8f16, nxv8i1, sme_elm_idx0_15,
                                     sme_elm_idx0_0, opq, tileslice128>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                     nxv8bf16, nxv8i1, sme_elm_idx0_15,
                                     sme_elm_idx0_0, opq, tileslice128>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                     nxv4i32, nxv4i1, sme_elm_idx0_15,
                                     sme_elm_idx0_0, opq, tileslice128>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                     nxv4f32, nxv4i1, sme_elm_idx0_15,
                                     sme_elm_idx0_0, opq, tileslice128>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                     nxv2i64, nxv2i1, sme_elm_idx0_15,
                                     sme_elm_idx0_0, opq, tileslice128>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                     nxv2f64, nxv2i1, sme_elm_idx0_15,
                                     sme_elm_idx0_0, opq, tileslice128>;
}

multiclass sme_vector_to_tile<string mnemonic> {
  defm _H : sme_vector_v_to_tile<mnemonic, /*is_col=*/0b0>;
  defm _V : sme_vector_v_to_tile<mnemonic, /*is_col=*/0b1>;
}

class sme_tile_to_vector_base<bit Q, bit V, bits<2> sz, dag outs, dag ins,
                              string mnemonic, string argstr>
    : I<outs, ins, mnemonic, argstr, "", []>, Sched<[]> {
  bits<2> Rv;
  bits<3> Pg;
  bits<5> Zd;
  let Inst{31-24} = 0b11000000;
  let Inst{23-22} = sz;
  let Inst{21-17} = 0b00001;
  let Inst{16}    = Q;
  let Inst{15}    = V;
  let Inst{14-13} = Rv;
  let Inst{12-10} = Pg;
  let Inst{9}     = 0b0;
  let Inst{4-0}   = Zd;
}

class sme_tile_to_vector_inst<bit Q, bits<2> sz, ZPRRegOp zpr_ty,
                              MatrixTileVectorOperand tile_ty,
                              bit is_col, Operand imm_ty, string mnemonic>
    : sme_tile_to_vector_base<Q, is_col, sz, (outs zpr_ty:$Zd),
        (ins zpr_ty:$_Zd, PPR3bAny:$Pg, tile_ty:$ZAn, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm),
        mnemonic, "\t$Zd, $Pg/m, $ZAn[$Rv, $imm]"> {

  let Constraints = "$Zd = $_Zd";
}

multiclass sme_tile_to_vector_aliases<Instruction inst, ZPRRegOp zpr_ty,
                                      MatrixTileVectorOperand tile_ty,
                                      Operand imm_ty > {
  def : InstAlias<"mov\t$Zd, $Pg/m, $ZAn[$Rv, $imm]",
                  (inst zpr_ty:$Zd, PPR3bAny:$Pg, tile_ty:$ZAn, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm), 1>;
}

multiclass sme_tile_to_vector_patterns<Instruction inst, ValueType zpr_vt,
                                       ValueType ppr_vt, Operand offset_ty,
                                       ComplexPattern imm2tile,
                                       ComplexPattern tileslice,
                                       SDPatternOperator op> {
  def : Pat<(zpr_vt (op (zpr_vt ZPRAny:$passthru), (ppr_vt PPR3bAny:$pg),
                        (imm2tile untyped:$tile), MatrixIndexGPR32Op12_15:$idx)),
            (inst $passthru, $pg, $tile, $idx, 0)>;
  let AddedComplexity = 1 in {
    def : Pat<(zpr_vt (op (zpr_vt ZPRAny:$passthru), (ppr_vt PPR3bAny:$pg),
                          (imm2tile untyped:$tile),
                          (i32 (tileslice MatrixIndexGPR32Op12_15:$idx,
                                          offset_ty:$imm)))),
              (inst $passthru, $pg, $tile, $idx, $imm)>;
  }
}

multiclass sme_tile_to_vector_v<string mnemonic, bit is_col> {
  def _B : sme_tile_to_vector_inst<0b0, 0b00, ZPR8, !if(is_col, TileVectorOpV8,
                                                                TileVectorOpH8),
                                   is_col, sme_elm_idx0_15, mnemonic> {
    bits<4> imm;
    let Inst{8-5} = imm;
  }
  def _H : sme_tile_to_vector_inst<0b0, 0b01, ZPR16, !if(is_col, TileVectorOpV16,
                                                                 TileVectorOpH16),
                                   is_col, sme_elm_idx0_7, mnemonic> {
    bits<1> ZAn;
    bits<3> imm;
    let Inst{8}   = ZAn;
    let Inst{7-5} = imm;
  }
  def _S : sme_tile_to_vector_inst<0b0, 0b10, ZPR32, !if(is_col, TileVectorOpV32,
                                                                 TileVectorOpH32),
                                   is_col, sme_elm_idx0_3, mnemonic> {
    bits<2> ZAn;
    bits<2> imm;
    let Inst{8-7} = ZAn;
    let Inst{6-5} = imm;
  }
  def _D : sme_tile_to_vector_inst<0b0, 0b11, ZPR64, !if(is_col, TileVectorOpV64,
                                                                 TileVectorOpH64),
                                   is_col, sme_elm_idx0_1, mnemonic> {
    bits<3> ZAn;
    bits<1> imm;
    let Inst{8-6} = ZAn;
    let Inst{5}   = imm;
  }
  def _Q : sme_tile_to_vector_inst<0b1, 0b11, ZPR128, !if(is_col, TileVectorOpV128,
                                                                  TileVectorOpH128),
                                   is_col, sme_elm_idx0_0, mnemonic> {
    bits<4> ZAn;
    let Inst{8-5} = ZAn;
  }

  defm : sme_tile_to_vector_aliases<!cast<Instruction>(NAME # _B), ZPR8,
                                    !if(is_col, TileVectorOpV8,
                                                TileVectorOpH8), sme_elm_idx0_15>;
  defm : sme_tile_to_vector_aliases<!cast<Instruction>(NAME # _H), ZPR16,
                                    !if(is_col, TileVectorOpV16,
                                                TileVectorOpH16), sme_elm_idx0_7>;
  defm : sme_tile_to_vector_aliases<!cast<Instruction>(NAME # _S), ZPR32,
                                    !if(is_col, TileVectorOpV32,
                                                TileVectorOpH32), sme_elm_idx0_3>;
  defm : sme_tile_to_vector_aliases<!cast<Instruction>(NAME # _D), ZPR64,
                                    !if(is_col, TileVectorOpV64,
                                                TileVectorOpH64), sme_elm_idx0_1>;
  defm : sme_tile_to_vector_aliases<!cast<Instruction>(NAME # _Q), ZPR128,
                                    !if(is_col, TileVectorOpV128,
                                                TileVectorOpH128), sme_elm_idx0_0>;

  defvar op = !if(is_col, int_aarch64_sme_read_vert,
                          int_aarch64_sme_read_horiz);

  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _B),
                                     nxv16i8, nxv16i1, imm0_15,
                                     imm_to_tile8, tileslice8, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _H),
                                     nxv8i16, nxv8i1, imm0_7,
                                     imm_to_tile16, tileslice16, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _H),
                                     nxv8f16, nxv8i1, imm0_7,
                                     imm_to_tile16, tileslice16, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _H),
                                     nxv8bf16, nxv8i1, imm0_7,
                                     imm_to_tile16, tileslice16, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _S),
                                     nxv4i32, nxv4i1, imm0_3,
                                     imm_to_tile32, tileslice32, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _S),
                                     nxv4f32, nxv4i1, imm0_3,
                                     imm_to_tile32, tileslice32, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _D),
                                     nxv2i64, nxv2i1, imm0_1,
                                     imm_to_tile64, tileslice64, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _D),
                                     nxv2f64, nxv2i1, imm0_1,
                                     imm_to_tile64, tileslice64, op>;

  defvar opq = !if(is_col, int_aarch64_sme_readq_vert,
                           int_aarch64_sme_readq_horiz);

  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _Q),
                                     nxv16i8, nxv16i1, sme_elm_idx0_0,
                                     imm_to_tile128, tileslice128, opq>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _Q),
                                     nxv8i16, nxv8i1, sme_elm_idx0_0,
                                     imm_to_tile128, tileslice128, opq>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _Q),
                                     nxv8f16, nxv8i1, sme_elm_idx0_0,
                                     imm_to_tile128, tileslice128, opq>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _Q),
                                     nxv8bf16, nxv8i1, sme_elm_idx0_0,
                                     imm_to_tile128, tileslice128, opq>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _Q),
                                     nxv4i32, nxv4i1, sme_elm_idx0_0,
                                     imm_to_tile128, tileslice128, opq>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _Q),
                                     nxv4f32, nxv4i1, sme_elm_idx0_0,
                                     imm_to_tile128, tileslice128, opq>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _Q),
                                     nxv2i64, nxv2i1, sme_elm_idx0_0,
                                     imm_to_tile128, tileslice128, opq>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _Q),
                                     nxv2f64, nxv2i1, sme_elm_idx0_0,
                                     imm_to_tile128, tileslice128, opq>;
}

multiclass sme_tile_to_vector<string mnemonic> {
  defm _H : sme_tile_to_vector_v<mnemonic, /*is_col=*/0b0>;
  defm _V : sme_tile_to_vector_v<mnemonic, /*is_col=*/0b1>;
}

//===----------------------------------------------------------------------===//
// SME Zero
//===----------------------------------------------------------------------===//

// NOTE: This definition isn't really correct because there are outputs, i.e.
// the tile registers being zeroed. We fix this up in a custom inserter that
// marks the appropriate registers as being implicitly defined.
class sme_zero_inst<string mnemonic>
    : I<(outs), (ins MatrixTileList:$imm),
        mnemonic, "\t$imm", "", []>, Sched<[]> {
  bits<8> imm;
  let Inst{31-8} = 0b110000000000100000000000;
  let Inst{7-0}  = imm;
}

multiclass sme_zero<string mnemonic> {
  def NAME : sme_zero_inst<mnemonic>;

  def : InstAlias<"zero\t\\{za\\}", (!cast<Instruction>(NAME) 0b11111111), 1>;
  def : InstAlias<"zero\t\\{za0.h\\}", (!cast<Instruction>(NAME) 0b01010101), 1>;
  def : InstAlias<"zero\t\\{za1.h\\}", (!cast<Instruction>(NAME) 0b10101010), 1>;
  def : InstAlias<"zero\t\\{za0.s\\}", (!cast<Instruction>(NAME) 0b00010001), 1>;
  def : InstAlias<"zero\t\\{za1.s\\}", (!cast<Instruction>(NAME) 0b00100010), 1>;
  def : InstAlias<"zero\t\\{za2.s\\}", (!cast<Instruction>(NAME) 0b01000100), 1>;
  def : InstAlias<"zero\t\\{za3.s\\}", (!cast<Instruction>(NAME) 0b10001000), 1>;
  def : InstAlias<"zero\t\\{za0.s,za1.s\\}", (!cast<Instruction>(NAME) 0b00110011), 1>;
  def : InstAlias<"zero\t\\{za0.s,za3.s\\}", (!cast<Instruction>(NAME) 0b10011001), 1>;
  def : InstAlias<"zero\t\\{za1.s,za2.s\\}", (!cast<Instruction>(NAME) 0b01100110), 1>;
  def : InstAlias<"zero\t\\{za2.s,za3.s\\}", (!cast<Instruction>(NAME) 0b11001100), 1>;
  def : InstAlias<"zero\t\\{za0.s,za1.s,za2.s\\}", (!cast<Instruction>(NAME) 0b01110111), 1>;
  def : InstAlias<"zero\t\\{za0.s,za1.s,za3.s\\}", (!cast<Instruction>(NAME) 0b10111011), 1>;
  def : InstAlias<"zero\t\\{za0.s,za2.s,za3.s\\}", (!cast<Instruction>(NAME) 0b11011101), 1>;
  def : InstAlias<"zero\t\\{za1.s,za2.s,za3.s\\}", (!cast<Instruction>(NAME) 0b11101110), 1>;

  def NAME # _PSEUDO : Pseudo<(outs), (ins i64imm:$tilelist), []>,
      Sched<[]> {
    // Translated to the actual instructions in AArch64ISelLowering.cpp
    let usesCustomInserter = 1;
  }

  def : Pat<(int_aarch64_sme_zero imm:$imm),
            (!cast<Instruction>(NAME # _PSEUDO) imm:$imm)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Instructions
//===----------------------------------------------------------------------===//

class sve2_int_perm_revd<string asm>
    : I<(outs ZPR128:$Zd), (ins ZPR128:$_Zd, PPR3bAny:$Pg, ZPR128:$Zn),
        asm, "\t$Zd, $Pg/m, $Zn", "", []>,
      Sched<[]> {
  bits<5> Zd;
  bits<3> Pg;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = 0b00; // size
  let Inst{21-13} = 0b101110100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveUnary;
  let ElementSize = ZPR128.ElementSize;
}

class sve2_clamp<string asm, bits<2> sz, bit U, ZPRRegOp zpr_ty>
    : I<(outs zpr_ty:$Zd), (ins zpr_ty:$Zn, zpr_ty:$Zm, zpr_ty:$_Zd),
        asm, "\t$Zd, $Zn, $Zm", "", []>,
      Sched<[]> {
  bits<5> Zm;
  bits<5> Zn;
  bits<5> Zd;
  let Inst{31-24} = 0b01000100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15-11} = 0b11000;
  let Inst{10}    = U;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zpr_ty.ElementSize;
}

multiclass sve2_clamp<string asm, bit U> {
  def _B : sve2_clamp<asm, 0b00, U, ZPR8>;
  def _H : sve2_clamp<asm, 0b01, U, ZPR16>;
  def _S : sve2_clamp<asm, 0b10, U, ZPR32>;
  def _D : sve2_clamp<asm, 0b11, U, ZPR64>;
}

class sve2_int_perm_sel_p<string asm, PPRRegOp ppr_ty, Operand imm_ty>
    : I<(outs PPRAny:$Pd), (ins PPRAny:$Pn, ppr_ty:$Pm,
                            MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm),
        asm, "\t$Pd, $Pn, $Pm[$Rv, $imm]", "", []>,
      Sched<[]> {
  bits<2> Rv;
  bits<4> Pn;
  bits<4> Pm;
  bits<4> Pd;
  let Inst{31-24} = 0b00100101;
  let Inst{21}    = 0b1;
  let Inst{17-16} = Rv;
  let Inst{15-14} = 0b01;
  let Inst{13-10} = Pn;
  let Inst{9}     = 0b0;
  let Inst{8-5}   = Pm;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = Pd;
}

multiclass sve2_int_perm_sel_p<string asm> {
  def _B : sve2_int_perm_sel_p<asm, PPR8, sme_elm_idx0_15> {
    bits<4> imm;
    let Inst{23-22} = imm{3-2};
    let Inst{20-19} = imm{1-0};
    let Inst{18}    = 0b1;
  }
  def _H : sve2_int_perm_sel_p<asm, PPR16, sme_elm_idx0_7> {
    bits<3> imm;
    let Inst{23-22} = imm{2-1};
    let Inst{20}    = imm{0};
    let Inst{19-18} = 0b10;
  }
  def _S : sve2_int_perm_sel_p<asm, PPR32, sme_elm_idx0_3> {
    bits<2> imm;
    let Inst{23-22} = imm{1-0};
    let Inst{20-18} = 0b100;
  }
  def _D : sve2_int_perm_sel_p<asm, PPR64, sme_elm_idx0_1> {
    bits<1> imm;
    let Inst{23}    = imm;
    let Inst{22}    = 0b1;
    let Inst{20-18} = 0b000;
  }
}
