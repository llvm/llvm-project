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

def imm_to_tile8   : ComplexPattern<i32, 1, "ImmToTile<AArch64::ZAB0>", []>;
def imm_to_tile16  : ComplexPattern<i32, 1, "ImmToTile<AArch64::ZAH0>", []>;
def imm_to_tile32  : ComplexPattern<i32, 1, "ImmToTile<AArch64::ZAS0>", []>;
def imm_to_tile64  : ComplexPattern<i32, 1, "ImmToTile<AArch64::ZAD0>", []>;
def imm_to_tile128 : ComplexPattern<i32, 1, "ImmToTile<AArch64::ZAQ0>", []>;

def tileslice8   : ComplexPattern<i32 , 2, "SelectSMETileSlice<4>", []>;
def tileslice16  : ComplexPattern<i32 , 2, "SelectSMETileSlice<3>", []>;
def tileslice32  : ComplexPattern<i32 , 2, "SelectSMETileSlice<2>", []>;
def tileslice64  : ComplexPattern<i32 , 2, "SelectSMETileSlice<1>", []>;
def tileslice128 : ComplexPattern<i32 , 2, "SelectSMETileSlice<0>", []>; // nop

def am_sme_indexed_b4 :ComplexPattern<iPTR, 2, "SelectAddrModeIndexedSVE<0,15>", [], [SDNPWantRoot]>;

//===----------------------------------------------------------------------===//
// SME Outer Products
//===----------------------------------------------------------------------===//

class sme_outer_product_pseudo<ZPRRegOp zpr_ty>
    : Pseudo<(outs), (ins i32imm:$tile, PPR3bAny:$pn, PPR3bAny:$pm,
                          zpr_ty:$zn, zpr_ty:$zm), []>,
      Sched<[]> {
  // Translated to the actual instructions in AArch64ISelLowering.cpp
  let usesCustomInserter = 1;
}

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

multiclass sme_outer_product_fp32<bit S, string mnemonic, SDPatternOperator op> {
  def NAME : sme_fp_outer_product_inst<S, 0b0, TileOp32, ZPR32, mnemonic> {
    bits<2> ZAda;
    let Inst{1-0} = ZAda;
    let Inst{2}   = 0b0;
  }

  def NAME # _PSEUDO : sme_outer_product_pseudo<ZPR32>;

  def : Pat<(op timm32_0_3:$tile, (nxv4i1 PPR3bAny:$pn), (nxv4i1 PPR3bAny:$pm),
                (nxv4f32 ZPR32:$zn), (nxv4f32 ZPR32:$zm)),
            (!cast<Instruction>(NAME # _PSEUDO) timm32_0_3:$tile, $pn, $pm, $zn, $zm)>;
}

multiclass sme_outer_product_fp64<bit S, string mnemonic, SDPatternOperator op> {
  def NAME : sme_fp_outer_product_inst<S, 0b1, TileOp64, ZPR64, mnemonic> {
    bits<3> ZAda;
    let Inst{2-0} = ZAda;
  }

  def NAME # _PSEUDO : sme_outer_product_pseudo<ZPR64>;

  def : Pat<(op timm32_0_7:$tile, (nxv2i1 PPR3bAny:$pn), (nxv2i1 PPR3bAny:$pm),
                (nxv2f64 ZPR64:$zn), (nxv2f64 ZPR64:$zm)),
            (!cast<Instruction>(NAME # _PSEUDO) timm32_0_7:$tile, $pn, $pm, $zn, $zm)>;
}

class sme_int_outer_product_inst<bits<3> opc, bit sz, bit sme2,
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
  let Inst{24}    = opc{2}; // u0
  let Inst{23}    = 0b1;
  let Inst{22}    = sz;
  let Inst{21}    = opc{1}; // u1
  let Inst{20-16} = Zm;
  let Inst{15-13} = Pm;
  let Inst{12-10} = Pn;
  let Inst{9-5}   = Zn;
  let Inst{4}     = opc{0};  //S;
  let Inst{3}     = sme2;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme_int_outer_product_i32<bits<3> opc, string mnemonic,
                                     SDPatternOperator op> {
  def NAME : sme_int_outer_product_inst<opc, 0b0, 0b0,  TileOp32,
                                        ZPR8, mnemonic> {
    bits<2> ZAda;
    let Inst{1-0} = ZAda;
    let Inst{2}   = 0b0;
  }

  def NAME # _PSEUDO : sme_outer_product_pseudo<ZPR8>;

  def : Pat<(op timm32_0_3:$tile, (nxv16i1 PPR3bAny:$pn), (nxv16i1 PPR3bAny:$pm),
                (nxv16i8 ZPR8:$zn), (nxv16i8 ZPR8:$zm)),
            (!cast<Instruction>(NAME # _PSEUDO) timm32_0_3:$tile, $pn, $pm, $zn, $zm)>;
}

multiclass sme_int_outer_product_i64<bits<3> opc, string mnemonic,
                                     SDPatternOperator op> {
  def NAME : sme_int_outer_product_inst<opc, 0b1, 0b0, TileOp64,
                                        ZPR16, mnemonic> {
    bits<3> ZAda;
    let Inst{2-0} = ZAda;
  }

  def NAME # _PSEUDO : sme_outer_product_pseudo<ZPR16>;

  def : Pat<(op timm32_0_7:$tile, (nxv8i1 PPR3bAny:$pn), (nxv8i1 PPR3bAny:$pm),
                (nxv8i16 ZPR16:$zn), (nxv8i16 ZPR16:$zm)),
            (!cast<Instruction>(NAME # _PSEUDO) timm32_0_7:$tile, $pn, $pm, $zn, $zm)>;
}

class sme_outer_product_widening_inst<bits<3> opc, ZPRRegOp zpr_ty, string mnemonic>
    : I<(outs TileOp32:$ZAda),
        (ins  TileOp32:$_ZAda, PPR3bAny:$Pn, PPR3bAny:$Pm, zpr_ty:$Zn, zpr_ty:$Zm),
        mnemonic, "\t$ZAda, $Pn/m, $Pm/m, $Zn, $Zm",
        "", []>,
      Sched<[]> {
  bits<5> Zm;
  bits<3> Pm;
  bits<3> Pn;
  bits<5> Zn;
  bits<2> ZAda;
  let Inst{31-25} = 0b1000000;
  let Inst{24}    = !if(opc{2}, 0, 1);
  let Inst{23-22} = 0b10;
  let Inst{21}    = opc{1};
  let Inst{20-16} = Zm;
  let Inst{15-13} = Pm;
  let Inst{12-10} = Pn;
  let Inst{9-5}   = Zn;
  let Inst{4}     = opc{0};
  let Inst{3}     = opc{2};
  let Inst{2}     = 0b0;
  let Inst{1-0}   = ZAda;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme_bf16_outer_product<bits<3> opc, string mnemonic, SDPatternOperator op> {
  def NAME : sme_outer_product_widening_inst<opc, ZPR16, mnemonic>;

  def NAME # _PSEUDO : sme_outer_product_pseudo<ZPR16>;

  def : Pat<(op timm32_0_3:$tile, (nxv8i1 PPR3bAny:$pn), (nxv8i1 PPR3bAny:$pm),
                (nxv8bf16 ZPR16:$zn), (nxv8bf16 ZPR16:$zm)),
            (!cast<Instruction>(NAME # _PSEUDO) timm32_0_3:$tile, $pn, $pm, $zn, $zm)>;
}

multiclass sme_f16_outer_product<bits<3> opc, string mnemonic, SDPatternOperator op> {
  def NAME : sme_outer_product_widening_inst<opc, ZPR16, mnemonic>;

  def NAME # _PSEUDO : sme_outer_product_pseudo<ZPR16>;

  def : Pat<(op timm32_0_3:$tile, (nxv8i1 PPR3bAny:$pn), (nxv8i1 PPR3bAny:$pm),
                (nxv8f16 ZPR16:$zn), (nxv8f16 ZPR16:$zm)),
            (!cast<Instruction>(NAME # _PSEUDO) timm32_0_3:$tile, $pn, $pm, $zn, $zm)>;
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

class sme_add_vector_to_tile_pseudo<ZPRRegOp zpr_ty>
    : Pseudo<(outs),
             (ins i32imm:$tile, PPR3bAny:$Pn, PPR3bAny:$Pm, zpr_ty:$Zn), []>,
      Sched<[]> {
  // Translated to the actual instructions in AArch64ISelLowering.cpp
  let usesCustomInserter = 1;
}

def ADDHA_MPPZ_PSEUDO_S : sme_add_vector_to_tile_pseudo<ZPR32>;
def ADDVA_MPPZ_PSEUDO_S : sme_add_vector_to_tile_pseudo<ZPR32>;

def : Pat<(int_aarch64_sme_addha
            timm32_0_3:$tile, (nxv4i1 PPR3bAny:$pn), (nxv4i1 PPR3bAny:$pm),
            (nxv4i32 ZPR32:$zn)),
          (ADDHA_MPPZ_PSEUDO_S timm32_0_3:$tile, $pn, $pm, $zn)>;
def : Pat<(int_aarch64_sme_addva
            timm32_0_3:$tile, (nxv4i1 PPR3bAny:$pn), (nxv4i1 PPR3bAny:$pm),
            (nxv4i32 ZPR32:$zn)),
          (ADDVA_MPPZ_PSEUDO_S timm32_0_3:$tile, $pn, $pm, $zn)>;

let Predicates = [HasSMEI16I64] in {
def ADDHA_MPPZ_PSEUDO_D : sme_add_vector_to_tile_pseudo<ZPR64>;
def ADDVA_MPPZ_PSEUDO_D : sme_add_vector_to_tile_pseudo<ZPR64>;

def : Pat<(int_aarch64_sme_addha
            timm32_0_7:$tile, (nxv2i1 PPR3bAny:$pn), (nxv2i1 PPR3bAny:$pm),
            (nxv2i64 ZPR64:$zn)),
          (ADDHA_MPPZ_PSEUDO_D timm32_0_7:$tile, $pn, $pm, $zn)>;
def : Pat<(int_aarch64_sme_addva
            timm32_0_7:$tile, (nxv2i1 PPR3bAny:$pn), (nxv2i1 PPR3bAny:$pm),
            (nxv2i64 ZPR64:$zn)),
          (ADDVA_MPPZ_PSEUDO_D timm32_0_7:$tile, $pn, $pm, $zn)>;
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
  // base, tileslice
  def : Pat<(Load PPR3bAny:$pg, GPR64sp:$base, tile_ty:$tile,
                  (i32 (tileslice MatrixIndexGPR32Op12_15:$idx, offset_ty:$imm))),
            (Inst tile_ty:$tile, $idx, $imm, $pg, $base, XZR)>;

  // reg + reg, tileslice
  let AddedComplexity = 1 in {
    def : Pat<(Load PPR3bAny:$pg, (addr GPR64sp:$base, GPR64:$offset),
                    tile_ty:$tile, (i32 (tileslice MatrixIndexGPR32Op12_15:$idx,
                                              offset_ty:$imm))),
              (Inst tile_ty:$tile, $idx, $imm, $pg, $base, $offset)>;
  }
}

class sme_load_pseudo
    : Pseudo<(outs), (ins i32imm:$tile, MatrixIndexGPR32Op12_15:$idx,
                          i32imm:$imm, PPR3bAny:$pg, GPR64sp:$base, GPR64:$offset), []>,
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
                                sme_elm_idx0_0, timm32_0_15, am_sve_regreg_lsl0,
                                tileslice8>;
  defm : sme_mem_ld_ss_patterns<!cast<Instruction>(NAME # _PSEUDO_H),
                                !if(is_col, int_aarch64_sme_ld1h_vert,
                                            int_aarch64_sme_ld1h_horiz),
                                timm32_0_1, timm32_0_7, am_sve_regreg_lsl1,
                                tileslice16>;
  defm : sme_mem_ld_ss_patterns<!cast<Instruction>(NAME # _PSEUDO_S),
                                !if(is_col, int_aarch64_sme_ld1w_vert,
                                            int_aarch64_sme_ld1w_horiz),
                                timm32_0_3, timm32_0_3, am_sve_regreg_lsl2,
                                tileslice32>;
  defm : sme_mem_ld_ss_patterns<!cast<Instruction>(NAME # _PSEUDO_D),
                                !if(is_col, int_aarch64_sme_ld1d_vert,
                                            int_aarch64_sme_ld1d_horiz),
                                timm32_0_7, timm32_0_1, am_sve_regreg_lsl3,
                                tileslice64>;
  defm : sme_mem_ld_ss_patterns<!cast<Instruction>(NAME # _PSEUDO_Q),
                                !if(is_col, int_aarch64_sme_ld1q_vert,
                                            int_aarch64_sme_ld1q_horiz),
                                timm32_0_15, sme_elm_idx0_0, am_sve_regreg_lsl4,
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
  // base, tileslice
  def : Pat<(Store PPR3bAny:$pg, GPR64sp:$base, (imm2tile untyped:$tile),
                   (i32 (tileslice MatrixIndexGPR32Op12_15:$idx, offset_ty:$imm))),
            (Inst $tile, $idx, $imm, $pg, $base, XZR)>;

  // reg + reg, tileslice
  let AddedComplexity = 1 in {
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
                                timm32_0_15, imm_to_tile8, am_sve_regreg_lsl0,
                                tileslice8>;
  defm : sme_mem_st_ss_patterns<!cast<Instruction>(NAME # _H),
                                !if(is_col, int_aarch64_sme_st1h_vert,
                                            int_aarch64_sme_st1h_horiz),
                                timm32_0_7, imm_to_tile16, am_sve_regreg_lsl1,
                                tileslice16>;
  defm : sme_mem_st_ss_patterns<!cast<Instruction>(NAME # _S),
                                !if(is_col, int_aarch64_sme_st1w_vert,
                                            int_aarch64_sme_st1w_horiz),
                                timm32_0_3, imm_to_tile32, am_sve_regreg_lsl2,
                                tileslice32>;
  defm : sme_mem_st_ss_patterns<!cast<Instruction>(NAME # _D),
                                !if(is_col, int_aarch64_sme_st1d_vert,
                                            int_aarch64_sme_st1d_horiz),
                                timm32_0_1, imm_to_tile64, am_sve_regreg_lsl3,
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
    : Pseudo<(outs), (ins i32imm:$tile, MatrixIndexGPR32Op12_15:$idx,
                          i32imm:$imm, PPR3bAny:$pg, ZPRAny:$zn), []>,
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
                                     nxv16i8, nxv16i1, sme_elm_idx0_0, sme_elm_idx0_15,
                                     op, tileslice8>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_H),
                                     nxv8i16, nxv8i1, sme_elm_idx0_1, sme_elm_idx0_7,
                                     op, tileslice16>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_H),
                                     nxv8f16, nxv8i1, sme_elm_idx0_1, sme_elm_idx0_7,
                                     op, tileslice16>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_H),
                                     nxv8bf16, nxv8i1, sme_elm_idx0_1, sme_elm_idx0_7,
                                     op, tileslice16>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_S),
                                     nxv4i32, nxv4i1, sme_elm_idx0_3, sme_elm_idx0_3,
                                     op, tileslice32>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_S),
                                     nxv4f32, nxv4i1, sme_elm_idx0_3, sme_elm_idx0_3,
                                     op, tileslice32>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_D),
                                     nxv2i64, nxv2i1, sme_elm_idx0_7, sme_elm_idx0_1,
                                     op, tileslice64>;
  defm : sme_vector_to_tile_patterns<!cast<Instruction>(NAME # _PSEUDO_D),
                                     nxv2f64, nxv2i1, sme_elm_idx0_7, sme_elm_idx0_1,
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
                                     nxv16i8, nxv16i1, sme_elm_idx0_15,
                                     imm_to_tile8, tileslice8, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _H),
                                     nxv8i16, nxv8i1, sme_elm_idx0_7,
                                     imm_to_tile16, tileslice16, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _H),
                                     nxv8f16, nxv8i1, sme_elm_idx0_7,
                                     imm_to_tile16, tileslice16, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _H),
                                     nxv8bf16, nxv8i1, sme_elm_idx0_7,
                                     imm_to_tile16, tileslice16, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _S),
                                     nxv4i32, nxv4i1, sme_elm_idx0_3,
                                     imm_to_tile32, tileslice32, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _S),
                                     nxv4f32, nxv4i1, sme_elm_idx0_3,
                                     imm_to_tile32, tileslice32, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _D),
                                     nxv2i64, nxv2i1, sme_elm_idx0_1,
                                     imm_to_tile64, tileslice64, op>;
  defm : sme_tile_to_vector_patterns<!cast<Instruction>(NAME # _D),
                                     nxv2f64, nxv2i1, sme_elm_idx0_1,
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

  def NAME # _PSEUDO : Pseudo<(outs), (ins i32imm:$tilelist), []>,
      Sched<[]> {
    // Translated to the actual instructions in AArch64ISelLowering.cpp
    let usesCustomInserter = 1;
  }

  def : Pat<(int_aarch64_sme_zero timm32_0_255:$imm),
            (!cast<Instruction>(NAME # _PSEUDO) timm32_0_255:$imm)>;
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

multiclass sve2_int_perm_revd<string asm, SDPatternOperator op> {
  def NAME : sve2_int_perm_revd<asm>;

  def : SVE_1_Op_Passthru_Pat<nxv16i8, op, nxv16i1, nxv16i8, !cast<Instruction>(NAME)>;
  def : SVE_1_Op_Passthru_Pat<nxv8i16, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME)>;
  def : SVE_1_Op_Passthru_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME)>;
  def : SVE_1_Op_Passthru_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME)>;
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

multiclass sve2_clamp<string asm, bit U, SDPatternOperator op> {
  def _B : sve2_clamp<asm, 0b00, U, ZPR8>;
  def _H : sve2_clamp<asm, 0b01, U, ZPR16>;
  def _S : sve2_clamp<asm, 0b10, U, ZPR32>;
  def _D : sve2_clamp<asm, 0b11, U, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
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

multiclass sve2_int_perm_sel_p<string asm, SDPatternOperator op> {
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

  def : Pat<(nxv16i1 (op (nxv16i1 PPRAny:$Pn), (nxv16i1 PPRAny:$Pm),
             MatrixIndexGPR32Op12_15:$idx)),
            (!cast<Instruction>(NAME # _B) $Pn, $Pm, $idx, 0)>;
  def : Pat<(nxv8i1 (op (nxv8i1 PPRAny:$Pn), (nxv8i1 PPRAny:$Pm),
             MatrixIndexGPR32Op12_15:$idx)),
            (!cast<Instruction>(NAME # _H) $Pn, $Pm, $idx, 0)>;
  def : Pat<(nxv4i1 (op (nxv4i1 PPRAny:$Pn), (nxv4i1 PPRAny:$Pm),
             MatrixIndexGPR32Op12_15:$idx)),
            (!cast<Instruction>(NAME # _S) $Pn, $Pm, $idx, 0)>;
  def : Pat<(nxv2i1 (op (nxv2i1 PPRAny:$Pn), (nxv2i1 PPRAny:$Pm),
             MatrixIndexGPR32Op12_15:$idx)),
            (!cast<Instruction>(NAME # _D) $Pn, $Pm, $idx, 0)>;

  let AddedComplexity = 1 in {
    def : Pat<(nxv16i1 (op (nxv16i1 PPRAny:$Pn), (nxv16i1 PPRAny:$Pm),
               (i32 (tileslice8 MatrixIndexGPR32Op12_15:$idx, sme_elm_idx0_15:$imm)))),
              (!cast<Instruction>(NAME # _B) $Pn, $Pm, $idx, $imm)>;
    def : Pat<(nxv8i1 (op (nxv8i1 PPRAny:$Pn), (nxv8i1 PPRAny:$Pm),
               (i32 (tileslice16 MatrixIndexGPR32Op12_15:$idx, sme_elm_idx0_7:$imm)))),
              (!cast<Instruction>(NAME # _H) $Pn, $Pm, $idx, $imm)>;
    def : Pat<(nxv4i1 (op (nxv4i1 PPRAny:$Pn), (nxv4i1 PPRAny:$Pm),
               (i32 (tileslice32 MatrixIndexGPR32Op12_15:$idx, sme_elm_idx0_3:$imm)))),
              (!cast<Instruction>(NAME # _S) $Pn, $Pm, $idx, $imm)>;
    def : Pat<(nxv2i1 (op (nxv2i1 PPRAny:$Pn), (nxv2i1 PPRAny:$Pm),
               (i32 (tileslice64 MatrixIndexGPR32Op12_15:$idx, sme_elm_idx0_1:$imm)))),
              (!cast<Instruction>(NAME # _D) $Pn, $Pm, $idx, $imm)>;
  }
}

//===----------------------------------------------------------------------===//
// SME2 Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SME2 single-multi ternary int/fp, two/four registers

class sme2_dot_mla_add_sub_array_vg24_single<bits<7> op,
                                         MatrixOperand matrix_ty,
                                         RegisterOperand multi_vector_ty,
                                         ZPRRegOp zpr_ty,
                                         string mnemonic>
   : I<(outs matrix_ty:$ZAd),
       (ins  matrix_ty:$_ZAd, MatrixIndexGPR32Op8_11:$Rv,
       sme_elm_idx0_7:$imm3, multi_vector_ty:$Zn, zpr_ty:$Zm),
       mnemonic,"\t$ZAd[$Rv, $imm3, " # !if(op{5}, "vgx4", "vgx2") # "], $Zn, $Zm",
       "", []> , Sched<[]> {
  bits<4> Zm;
  bits<5> Zn;
  bits<2> Rv;
  bits<3> imm3;
  let Inst{31-23} = 0b110000010;
  let Inst{22}    = op{6}; //sz
  let Inst{21}    = 0b1;
  let Inst{20}    = op{5}; //vgx4
  let Inst{19-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-13} = Rv;
  let Inst{12-10} = op{4-2};
  let Inst{9-5}   = Zn;
  let Inst{4-3}   = op{1-0};
  let Inst{2-0}   = imm3;
  let Constraints = "$ZAd = $_ZAd";
}

multiclass sme2_dot_mla_add_sub_array_vg24_single<string mnemonic, bits<7> op,
                                              MatrixOperand matrix_ty,
                                              RegisterOperand multi_vector_ty,
                                              ZPRRegOp zpr_ty>{
  def NAME: sme2_dot_mla_add_sub_array_vg24_single<op, matrix_ty, multi_vector_ty, zpr_ty, mnemonic>;

  def : InstAlias<mnemonic # "\t$ZAd[$Rv, $imm3], $Zn, $Zm",
                 (!cast<Instruction>(NAME) matrix_ty:$ZAd,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3, multi_vector_ty:$Zn, zpr_ty:$Zm), 0>;
}

//===----------------------------------------------------------------------===//
// SME2 multiple vectors ternary INT/FP  two and four registers
class sme2_dot_mla_add_sub_array_vg2_multi<bits<6> op,
                                       MatrixOperand matrix_ty,
                                       RegisterOperand multi_vector_ty,
                                       string mnemonic>
   : I<(outs matrix_ty:$ZAd),
       (ins  matrix_ty:$_ZAd, MatrixIndexGPR32Op8_11:$Rv,
       sme_elm_idx0_7:$imm3, multi_vector_ty:$Zn, multi_vector_ty:$Zm),
       mnemonic, "\t$ZAd[$Rv, $imm3, vgx2], $Zn, $Zm",
       "", []>, Sched<[]>{
  bits<4> Zm;
  bits<4> Zn;
  bits<2> Rv;
  bits<3> imm3;
  let Inst{31-23} = 0b110000011;
  let Inst{22}    = op{5}; //sz
  let Inst{21}    = 0b1;
  let Inst{20-17} = Zm;
  let Inst{16-15} = 0b00;
  let Inst{14-13} = Rv;
  let Inst{12-10} = op{4-2};
  let Inst{9-6}   = Zn;
  let Inst{5}     = 0b0;
  let Inst{4-3}   = op{1-0};
  let Inst{2-0}   = imm3;
  let Constraints = "$ZAd = $_ZAd";
}

multiclass sme2_dot_mla_add_sub_array_vg2_multi<string mnemonic, bits<6> op,
                                            MatrixOperand  matrix_ty,
                                            RegisterOperand multi_vector_ty>{
 def NAME : sme2_dot_mla_add_sub_array_vg2_multi<op, matrix_ty, multi_vector_ty, mnemonic>;

   def : InstAlias<mnemonic # "\t$ZAd[$Rv, $imm3], $Zn, $Zm",
                 (!cast<Instruction>(NAME) matrix_ty:$ZAd,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3, multi_vector_ty:$Zn, multi_vector_ty:$Zm), 0>;

}

class sme2_dot_mla_add_sub_array_vg4_multi<bits<6> op,
                                            MatrixOperand matrix_ty,
                                            RegisterOperand multi_vector_ty,
                                            string mnemonic>
   : I<(outs matrix_ty:$ZAd),
       (ins  matrix_ty:$_ZAd, MatrixIndexGPR32Op8_11:$Rv,
        sme_elm_idx0_7:$imm3, multi_vector_ty:$Zn, multi_vector_ty:$Zm),
        mnemonic, "\t$ZAd[$Rv, $imm3, vgx4], $Zn, $Zm",
        "", []>, Sched<[]>{
  bits<3> Zm;
  bits<3> Zn;
  bits<2> Rv;
  bits<3> imm3;
  let Inst{31-23} = 0b110000011;
  let Inst{22}    = op{5}; //sz
  let Inst{21}    = 0b1;
  let Inst{20-18} = Zm;
  let Inst{17-15} = 0b010;
  let Inst{14-13} = Rv;
  let Inst{12-10} = op{4-2};
  let Inst{9-7}   = Zn;
  let Inst{6-5}   = 0b00;
  let Inst{4-3}   = op{1-0};
  let Inst{2-0}   = imm3;
  let Constraints = "$ZAd = $_ZAd";
}


multiclass sme2_dot_mla_add_sub_array_vg4_multi<string mnemonic, bits<6> op,
                                            MatrixOperand  matrix_ty,
                                            RegisterOperand multi_vector_ty>{
 def NAME : sme2_dot_mla_add_sub_array_vg4_multi<op, matrix_ty, multi_vector_ty, mnemonic>;

   def : InstAlias<mnemonic # "\t$ZAd[$Rv, $imm3], $Zn, $Zm",
                 (!cast<Instruction>(NAME) matrix_ty:$ZAd,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3, multi_vector_ty:$Zn, multi_vector_ty:$Zm), 0>;

}

//===----------------------------------------------------------------------===//
// SME2 multiple vectors binary two or four  registers

class sme2_multivec_accum_add_sub<string mnemonic, bit sz, bit is_int,
                                  bit s, MatrixOperand matrix_ty,
                                  RegisterOperand vector_ty,
                                  string vg_acronym>
    : I<(outs matrix_ty:$ZAdn),
        (ins matrix_ty:$_ZAdn, MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3, vector_ty:$Zm),
        mnemonic, "\t$ZAdn[$Rv, $imm3, " # vg_acronym # "], $Zm",
        "", []>, Sched<[]> {
  bits<2> Rv;
  bits<3> imm3;
  let Inst{31-23} = 0b110000011;
  let Inst{22}    = sz;
  let Inst{21-17} = 0b10000;
  let Inst{15}    = 0b0;
  let Inst{14-13} = Rv;
  let Inst{12-10} = 0b111;
  let Inst{5}     = 0b0;
  let Inst{4}     = is_int;
  let Inst{3}     = s;
  let Inst{2-0}   = imm3;

  let Constraints = "$ZAdn = $_ZAdn";
}

class sme2_multivec_accum_add_sub_vg2<string mnemonic, bit sz, bit is_int,
                                      bit s, MatrixOperand matrix_ty,
                                      RegisterOperand vector_ty>
    : sme2_multivec_accum_add_sub<mnemonic, sz, is_int, s,
                                  matrix_ty, vector_ty , "vgx2"> {
  bits<4> Zm;
  let Inst{16}  = 0b0;
  let Inst{9-6} = Zm;
}


multiclass sme2_multivec_accum_add_sub_vg2_S<string mnemonic, bits<2> op> {
  def NAME : sme2_multivec_accum_add_sub_vg2<mnemonic, 0b0, op{1}, op{0},
                                             MatrixOp32, ZZ_s_mul_r>;

  def : InstAlias<mnemonic # "\t$ZAdn[$Rv, $imm3], $Zm",
                 (!cast<Instruction>(NAME) MatrixOp32:$ZAdn,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3, ZZ_s_mul_r:$Zm), 0>;
}

multiclass sme2_multivec_accum_add_sub_vg2_D<string mnemonic, bits<2> op> {
  def NAME : sme2_multivec_accum_add_sub_vg2<mnemonic, 0b1, op{1}, op{0},
                                             MatrixOp64, ZZ_d_mul_r>;
  def : InstAlias<mnemonic # "\t$ZAdn[$Rv, $imm3], $Zm",
                 (!cast<Instruction>(NAME) MatrixOp64:$ZAdn,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3, ZZ_d_mul_r:$Zm), 0>;

}

class sme2_multivec_accum_add_sub_vg4<string mnemonic, bit sz, bit is_int,
                                      bit s, MatrixOperand matrix_ty,
                                      RegisterOperand vector_ty>
    : sme2_multivec_accum_add_sub<mnemonic, sz, is_int, s,
                                  matrix_ty, vector_ty , "vgx4"> {
  bits<3> Zm;
  let Inst{16}  = 0b1;
  let Inst{9-7} = Zm;
  let Inst{6}   = 0b0;
}

multiclass sme2_multivec_accum_add_sub_vg4_S<string mnemonic, bits<2> op> {
  def NAME : sme2_multivec_accum_add_sub_vg4<mnemonic, 0b0, op{1}, op{0},
                                             MatrixOp32, ZZZZ_s_mul_r>;

  def : InstAlias<mnemonic # "\t$ZAdn[$Rv, $imm3], $Zm",
                 (!cast<Instruction>(NAME) MatrixOp32:$ZAdn,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3, ZZZZ_s_mul_r:$Zm), 0>;
}

multiclass sme2_multivec_accum_add_sub_vg4_D<string mnemonic, bits<2> op> {
  def NAME : sme2_multivec_accum_add_sub_vg4<mnemonic, 0b1, op{1}, op{0},
                                             MatrixOp64, ZZZZ_d_mul_r>;
  def : InstAlias<mnemonic # "\t$ZAdn[$Rv, $imm3], $Zm",
                 (!cast<Instruction>(NAME) MatrixOp64:$ZAdn,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3, ZZZZ_d_mul_r:$Zm), 0>;
}

//===----------------------------------------------------------------------===//
// SME2 Multi-vector - Multiple and Single SVE Destructive
// Two and Four registers

class sme2_sve_destructive_vector_vg2_single<bits<2> sz, bits<7> op,
                                             RegisterOperand vector_ty,
                                             ZPRRegOp zpr_ty,
                                             string mnemonic>
    : I<(outs vector_ty:$Zdn), (ins vector_ty:$_Zdn, zpr_ty:$Zm),
        mnemonic, "\t$Zdn, $_Zdn, $Zm",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<4> Zdn;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = sz;
  let Inst{21-20} = 0b10;
  let Inst{19-16} = Zm;
  let Inst{15-11} = 0b10100;
  let Inst{10-5}  = op{6-1};
  let Inst{4-1}   = Zdn;
  let Inst{0}     = op{0};

  let Constraints = "$Zdn = $_Zdn";
}

multiclass sme2_fp_sve_destructive_vector_vg2_single<string mnemonic, bits<7> op> {
  def _H : sme2_sve_destructive_vector_vg2_single<0b01, op, ZZ_h_mul_r, ZPR4b16, mnemonic>;
  def _S : sme2_sve_destructive_vector_vg2_single<0b10, op, ZZ_s_mul_r, ZPR4b32, mnemonic>;
  def _D : sme2_sve_destructive_vector_vg2_single<0b11, op, ZZ_d_mul_r, ZPR4b64, mnemonic>;
}

multiclass sme2_int_sve_destructive_vector_vg2_single<string mnemonic, bits<7> op> {
  def _B : sme2_sve_destructive_vector_vg2_single<0b00, op, ZZ_b_mul_r, ZPR4b8, mnemonic>;
  def _H : sme2_sve_destructive_vector_vg2_single<0b01, op, ZZ_h_mul_r, ZPR4b16, mnemonic>;
  def _S : sme2_sve_destructive_vector_vg2_single<0b10, op, ZZ_s_mul_r, ZPR4b32, mnemonic>;
  def _D : sme2_sve_destructive_vector_vg2_single<0b11, op, ZZ_d_mul_r, ZPR4b64, mnemonic>;
}

class sme2_sve_destructive_vector_vg4_single<bits<2> sz, bits<7> op,
                                             RegisterOperand vector_ty,
                                             ZPRRegOp zpr_ty,
                                             string mnemonic>
    : I<(outs vector_ty:$Zdn), (ins vector_ty:$_Zdn, zpr_ty:$Zm),
        mnemonic, "\t$Zdn, $_Zdn, $Zm",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<3> Zdn;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = sz;
  let Inst{21-20} = 0b10;
  let Inst{19-16} = Zm;
  let Inst{15-11} = 0b10101;
  let Inst{10-5}  = op{6-1};
  let Inst{4-2}   = Zdn;
  let Inst{1}     = 0b0;
  let Inst{0}     = op{0};

  let Constraints = "$Zdn = $_Zdn";
}

multiclass sme2_fp_sve_destructive_vector_vg4_single<string mnemonic, bits<7> op> {
  def _H : sme2_sve_destructive_vector_vg4_single<0b01, op, ZZZZ_h_mul_r, ZPR4b16, mnemonic>;
  def _S : sme2_sve_destructive_vector_vg4_single<0b10, op, ZZZZ_s_mul_r, ZPR4b32, mnemonic>;
  def _D : sme2_sve_destructive_vector_vg4_single<0b11, op, ZZZZ_d_mul_r, ZPR4b64, mnemonic>;
}

multiclass sme2_int_sve_destructive_vector_vg4_single<string mnemonic, bits<7> op> {
  def _B : sme2_sve_destructive_vector_vg4_single<0b00, op, ZZZZ_b_mul_r, ZPR4b8, mnemonic>;
  def _H : sme2_sve_destructive_vector_vg4_single<0b01, op, ZZZZ_h_mul_r, ZPR4b16, mnemonic>;
  def _S : sme2_sve_destructive_vector_vg4_single<0b10, op, ZZZZ_s_mul_r, ZPR4b32, mnemonic>;
  def _D : sme2_sve_destructive_vector_vg4_single<0b11, op, ZZZZ_d_mul_r, ZPR4b64, mnemonic>;
}

class sme2_sve_destructive_vector_vg2_multi<bits<2> sz, bits<7> op,
                                            RegisterOperand vector_ty,
                                            string mnemonic>
    : I<(outs vector_ty:$Zdn), (ins vector_ty:$_Zdn, vector_ty:$Zm),
        mnemonic, "\t$Zdn, $_Zdn, $Zm",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<4> Zdn;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{20-17} = Zm;
  let Inst{16-11} = 0b010110;
  let Inst{10-5}  = op{6-1};
  let Inst{4-1}   = Zdn;
  let Inst{0}     = op{0};

  let Constraints = "$Zdn = $_Zdn";
}

multiclass sme2_fp_sve_destructive_vector_vg2_multi<string mnemonic, bits<7> op> {
  def _H : sme2_sve_destructive_vector_vg2_multi<0b01, op, ZZ_h_mul_r, mnemonic>;
  def _S : sme2_sve_destructive_vector_vg2_multi<0b10, op, ZZ_s_mul_r, mnemonic>;
  def _D : sme2_sve_destructive_vector_vg2_multi<0b11, op, ZZ_d_mul_r, mnemonic>;
}

multiclass sme2_int_sve_destructive_vector_vg2_multi<string mnemonic, bits<7> op> {
  def _B : sme2_sve_destructive_vector_vg2_multi<0b00, op, ZZ_b_mul_r, mnemonic>;
  def _H : sme2_sve_destructive_vector_vg2_multi<0b01, op, ZZ_h_mul_r, mnemonic>;
  def _S : sme2_sve_destructive_vector_vg2_multi<0b10, op, ZZ_s_mul_r, mnemonic>;
  def _D : sme2_sve_destructive_vector_vg2_multi<0b11, op, ZZ_d_mul_r, mnemonic>;
}

class sme2_sve_destructive_vector_vg4_multi<bits<2> sz, bits<7> op,
                                            RegisterOperand vector_ty,
                                            string mnemonic>
    : I<(outs vector_ty:$Zdn), (ins vector_ty:$_Zdn, vector_ty:$Zm),
        mnemonic, "\t$Zdn, $_Zdn, $Zm",
        "", []>, Sched<[]> {
  bits<3> Zm;
  bits<3> Zdn;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{20-18} = Zm;
  let Inst{17-11} = 0b0010111;
  let Inst{10-5}  = op{6-1};
  let Inst{4-2}   = Zdn;
  let Inst{1}     = 0b0;
  let Inst{0}     = op{0};

  let Constraints = "$Zdn = $_Zdn";
}

multiclass sme2_fp_sve_destructive_vector_vg4_multi<string mnemonic, bits<7> op> {
  def _H : sme2_sve_destructive_vector_vg4_multi<0b01, op, ZZZZ_h_mul_r, mnemonic>;
  def _S : sme2_sve_destructive_vector_vg4_multi<0b10, op, ZZZZ_s_mul_r, mnemonic>;
  def _D : sme2_sve_destructive_vector_vg4_multi<0b11, op, ZZZZ_d_mul_r, mnemonic>;
}

multiclass sme2_int_sve_destructive_vector_vg4_multi<string mnemonic, bits<7> op> {
  def _B : sme2_sve_destructive_vector_vg4_multi<0b00, op, ZZZZ_b_mul_r, mnemonic>;
  def _H : sme2_sve_destructive_vector_vg4_multi<0b01, op, ZZZZ_h_mul_r, mnemonic>;
  def _S : sme2_sve_destructive_vector_vg4_multi<0b10, op, ZZZZ_s_mul_r, mnemonic>;
  def _D : sme2_sve_destructive_vector_vg4_multi<0b11, op, ZZZZ_d_mul_r, mnemonic>;
}


//===----------------------------------------------------------------------===//
// SME2 Multi-vector - Index/Single/Multi Array Vectors FMA sources

class sme2_mla_long_array_index_base<bits<2> op0, bits<2> op, Operand index_ty,
                                     RegisterOperand multi_vector_ty,
                                     string mnemonic, string vg_acronym="">
    : I<(outs MatrixOp32:$ZAda),
        (ins MatrixOp32:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, index_ty:$imm, multi_vector_ty:$Zn, ZPR4b16:$Zm, VectorIndexH:$i3),
        mnemonic, "\t$ZAda[$Rv, $imm" # !if(!eq(vg_acronym, ""), "", ", " # vg_acronym) # "], $Zn, $Zm$i3",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = op0;
  let Inst{21}    = 0b0;
  let Inst{20}    = !if(!eq(vg_acronym, ""), 0, 1);
  let Inst{19-16} = Zm;
  let Inst{14-13} = Rv;
  let Inst{12}    = 0b1;
  let Inst{4-3}   = op;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme2_mla_long_array_index<string mnemonic, bits<2> op0, bits<2> op> {
  def _S : sme2_mla_long_array_index_base<op0, op, uimm3s2range, ZPR16,
                                          mnemonic> {
    bits<3> i3;
    bits<5> Zn;
    bits<3> imm;
    let Inst{15}    = i3{2};
    let Inst{11-10} = i3{1-0};
    let Inst{9-5}   = Zn;
    let Inst{2-0}   = imm;
  }
}

class sme2_mla_long_array_vg2_index<string mnemonic, bits<2> op0, bits<2> op>
    : sme2_mla_long_array_index_base<op0, op, uimm2s2range, ZZ_h_mul_r,
                                     mnemonic, "vgx2"> {
  bits<3> i3;
  bits<4> Zn;
  bits<2> imm;
  let Inst{15}    = 0b0;
  let Inst{11-10} = i3{2-1};
  let Inst{9-6}   = Zn;
  let Inst{5}     = 0b0;
  let Inst{2}     = i3{0};
  let Inst{1-0}   = imm;
}

multiclass sme2_fp_mla_long_array_vg2_index<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg2_index<mnemonic, 0b10, op>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm$i3",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZ_h_mul_r:$Zn, ZPR4b16:$Zm, VectorIndexH:$i3), 0>;
}

multiclass sme2_int_mla_long_array_vg2_index<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg2_index<mnemonic, 0b11, op>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm$i3",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZ_h_mul_r:$Zn, ZPR4b16:$Zm, VectorIndexH:$i3), 0>;
}

class sme2_mla_long_array_vg4_index<string mnemonic, bits<2> op0, bits<2> op>
    : sme2_mla_long_array_index_base<op0, op, uimm2s2range, ZZZZ_h_mul_r,
                                      mnemonic, "vgx4"> {
  bits<3> i3;
  bits<3> Zn;
  bits<2> imm;
  let Inst{15}    = 0b1;
  let Inst{11-10} = i3{2-1};
  let Inst{9-7}   = Zn;
  let Inst{6-5}   = 0b00;
  let Inst{2}     = i3{0};
  let Inst{1-0}   = imm;
}

multiclass sme2_fp_mla_long_array_vg4_index<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg4_index<mnemonic, 0b10, op>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm$i3",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZZZ_h_mul_r:$Zn, ZPR4b16:$Zm, VectorIndexH:$i3), 0>;
}

multiclass sme2_int_mla_long_array_vg4_index<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg4_index<mnemonic, 0b11, op>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm$i3",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZZZ_h_mul_r:$Zn, ZPR4b16:$Zm, VectorIndexH:$i3), 0>;
}

class sme2_mla_long_array<bits<2>op0, bits<2> op, Operand index_ty,
                          RegisterOperand first_vector_ty,
                          RegisterOperand second_vector_ty,
                          string mnemonic, string vg_acronym="">
   : I<(outs MatrixOp32:$ZAda),
       (ins  MatrixOp32:$_ZAda, MatrixIndexGPR32Op8_11:$Rv,
       index_ty:$imm, first_vector_ty:$Zn, second_vector_ty:$Zm),
       mnemonic,"\t$ZAda[$Rv, $imm" # !if(!eq(vg_acronym, ""), "", ", " # vg_acronym) # "], $Zn, $Zm",
       "", []> , Sched<[]> {
  bits<2> Rv;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = op0;
  let Inst{21}    = 0b1;
  let Inst{15}    = 0b0;
  let Inst{14-13} = Rv;
  let Inst{12-11} = 0b01;
  let Inst{10}    = !if(!eq(vg_acronym, ""), 1, 0);
  let Inst{4-3}   = op;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme2_mla_long_array_single<string mnemonic, bits<2> op0, bits<2> op> {
  def _S : sme2_mla_long_array<op0, op, uimm3s2range, ZPR16, ZPR4b16,
                               mnemonic> {
    bits<4> Zm;
    bits<5> Zn;
    bits<3> imm;
    let Inst{20}    = 0b0;
    let Inst{19-16} = Zm;
    let Inst{9-5}   = Zn;
    let Inst{2-0}   = imm;
  }
}

class sme2_mla_long_array_vg24_single<bits<2> op0, bit vg4, bits<2> op,
                                      RegisterOperand first_vector_ty,
                                      string mnemonic, string vg_acronym>
    : sme2_mla_long_array<op0, op, uimm2s2range, first_vector_ty, ZPR4b16,
                          mnemonic, vg_acronym> {
  bits<4> Zm;
  bits<5> Zn;
  bits<2> imm;
  let Inst{20}    = vg4;
  let Inst{19-16} = Zm;
  let Inst{9-5}   = Zn;
  let Inst{2}     = 0b0;
  let Inst{1-0}   = imm;
}

multiclass sme2_fp_mla_long_array_vg2_single<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg24_single<0b00, 0b0, op, ZZ_h, mnemonic,
                                           "vgx2">;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZ_h:$Zn, ZPR4b16:$Zm), 0>;
}

multiclass sme2_int_mla_long_array_vg2_single<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg24_single<0b01, 0b0, op, ZZ_h, mnemonic,
                                           "vgx2">;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZ_h:$Zn, ZPR4b16:$Zm), 0>;
}

multiclass sme2_fp_mla_long_array_vg4_single<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg24_single<0b00, 0b1, op, ZZZZ_h, mnemonic,
                                           "vgx4">;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZZZ_h:$Zn, ZPR4b16:$Zm), 0>;
}

multiclass sme2_int_mla_long_array_vg4_single<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg24_single<0b01, 0b1, op, ZZZZ_h, mnemonic,
                                           "vgx4">;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZZZ_h:$Zn, ZPR4b16:$Zm), 0>;
}
class sme2_mla_long_array_vg2_multi<string mnemonic, bits<2> op0, bits<2> op>
   : sme2_mla_long_array<op0, op, uimm2s2range, ZZ_h_mul_r, ZZ_h_mul_r,  mnemonic,
                         "vgx2"> {
  bits<4> Zm;
  bits<4> Zn;
  bits<2> imm;
  let Inst{20-17} = Zm;
  let Inst{16}    = 0b0;
  let Inst{9-6}   = Zn;
  let Inst{5}     = 0b0;
  let Inst{2}     = 0b0;
  let Inst{1-0}   = imm;
}

multiclass sme2_fp_mla_long_array_vg2_multi<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg2_multi<mnemonic, 0b10, op>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZ_h_mul_r:$Zn, ZZ_h_mul_r:$Zm), 0>;
}

multiclass sme2_int_mla_long_array_vg2_multi<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg2_multi<mnemonic, 0b11, op>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm2], $Zn, $Zm",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm2, ZZ_h_mul_r:$Zn, ZZ_h_mul_r:$Zm), 0>;
}

class sme2_mla_long_array_vg4_multi<string mnemonic, bits<2> op0, bits<2> op>
   : sme2_mla_long_array<op0, op, uimm2s2range, ZZZZ_h_mul_r, ZZZZ_h_mul_r, mnemonic,
                        "vgx4"> {
  bits<3> Zm;
  bits<3> Zn;
  bits<2> imm;
  let Inst{20-18} = Zm;
  let Inst{17}    = 0b0;
  let Inst{16}    = 0b1;
  let Inst{9-7}   = Zn;
  let Inst{6-5}   = 0b00;
  let Inst{2}     = 0b0;
  let Inst{1-0}   = imm;
}

multiclass sme2_fp_mla_long_array_vg4_multi<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg4_multi<mnemonic, 0b10, op>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm, ZZZZ_h_mul_r:$Zn, ZZZZ_h_mul_r:$Zm), 0>;
}

multiclass sme2_int_mla_long_array_vg4_multi<string mnemonic, bits<2> op> {
  def _S : sme2_mla_long_array_vg4_multi<mnemonic, 0b11, op>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm2], $Zn, $Zm",
                 (!cast<Instruction>(NAME #_S) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm2s2range:$imm2, ZZZZ_h_mul_r:$Zn, ZZZZ_h_mul_r:$Zm), 0>;
}

//===----------------------------------------------------------------------===//
class sme2_frint_cvt_vg2_multi<bits<2>sz, bits<5>op, RegisterOperand first_ty,
                               RegisterOperand second_ty, string mnemonic>
    : I<(outs first_ty:$Zd), (ins second_ty:$Zn),
        mnemonic, "\t$Zd, $Zn", "", []>, Sched<[]> {
  bits<4> Zn;
  bits<4> Zd;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = sz;
  let Inst{21-20} = 0b10;
  let Inst{19-16} = op{4-1};
  let Inst{15-10} = 0b111000;
  let Inst{9-6}   = Zn;
  let Inst{5}     = op{0};
  let Inst{4-1}   = Zd;
  let Inst{0}     = 0b0;
}

// SME2 multi-vec FP to int convert two registers
// SME2 multi-vec int to FP two registers
multiclass sme2_fp_cvt_vg2_multi<string mnemonic, bits<5> op> {
  def NAME : sme2_frint_cvt_vg2_multi<0b00, op, ZZ_s_mul_r, ZZ_s_mul_r, mnemonic>;
}

// SME2 multi-vec FRINT two registers
multiclass sme2_frint_vector_vg2_multi<string mnemonic, bits<5> op> {
  def _S : sme2_frint_cvt_vg2_multi<0b10, op, ZZ_s_mul_r, ZZ_s_mul_r, mnemonic>;
}

class sme2_frint_zip_cvt_vg4_multi<bits<2>sz, bits<7>op, RegisterOperand first_ty,
                                   RegisterOperand second_ty, string mnemonic>
    : I<(outs first_ty:$Zd), (ins second_ty:$Zn),
        mnemonic, "\t$Zd, $Zn", "", []>, Sched<[]> {
  bits<3> Zn;
  bits<3> Zd;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = sz;
  let Inst{21-20} = 0b11;
  let Inst{19-16} = op{6-3};
  let Inst{15-10} = 0b111000;
  let Inst{9-7}   = Zn;
  let Inst{6-5}   = op{2-1};
  let Inst{4-2}   = Zd;
  let Inst{1}     = op{0};
  let Inst{0}     = 0b0;
}

// SME2 multi-vec FP to int convert four registers
// SME2 multi-vec int to FP four registers
multiclass sme2_fp_cvt_vg4_multi<string mnemonic, bits<7> op> {
  def _S : sme2_frint_zip_cvt_vg4_multi<0b00, op, ZZZZ_s_mul_r, ZZZZ_s_mul_r, mnemonic>;
}

// SME2 multi-vec quadwords ZIP four registers
multiclass sme2_zip_vector_vg4<string mnemonic, bits<7> op> {
  def _B : sme2_frint_zip_cvt_vg4_multi<0b00, op, ZZZZ_b_mul_r, ZZZZ_b_mul_r,
                                        mnemonic>;
  def _H : sme2_frint_zip_cvt_vg4_multi<0b01, op, ZZZZ_h_mul_r, ZZZZ_h_mul_r,
                                        mnemonic>;
  def _S : sme2_frint_zip_cvt_vg4_multi<0b10, op, ZZZZ_s_mul_r, ZZZZ_s_mul_r,
                                        mnemonic>;
  def _D : sme2_frint_zip_cvt_vg4_multi<0b11, op, ZZZZ_d_mul_r, ZZZZ_d_mul_r,
                                         mnemonic>;
}

// SME2 multi-vec quadwords ZIP four registers
multiclass sme2_zip_vector_vg4_Q<string mnemonic, bits<7> op> {
  def NAME: sme2_frint_zip_cvt_vg4_multi<0b00, op, ZZZZ_q_mul_r, ZZZZ_q_mul_r,
                                         mnemonic>;
}

// SME2 multi-vec FRINT four registers
multiclass sme2_frint_vector_vg4_multi<string mnemonic, bits<7> op> {
  def _S :  sme2_frint_zip_cvt_vg4_multi<0b10, op, ZZZZ_s_mul_r, ZZZZ_s_mul_r,
                                         mnemonic>;
}

class sme2_cvt_vg2_single<string mnemonic, bits<4> op>
    : I<(outs ZPR16:$Zd), (ins ZZ_s_mul_r:$Zn),
        mnemonic, "\t$Zd, $Zn", "", []>, Sched<[]> {
  bits<4> Zn;
  bits<5> Zd;
  let Inst{31-23} = 0b110000010;
  let Inst{22}    = op{3};
  let Inst{21-18} = 0b1000;
  let Inst{17-16} = op{2-1};
  let Inst{15-10} = 0b111000;
  let Inst{9-6}   = Zn;
  let Inst{5}     = op{0};
  let Inst{4-0}   = Zd;
}

// SME2 multi-vec FP down convert two registers
// SME2 multi-vec int down convert two registers
multiclass sme2_cvt_vg2_single<string mnemonic, bits<4> op> {
  def NAME :  sme2_cvt_vg2_single<mnemonic, op>;
}


class sme2_unpk_vector_vg2<bits<2>sz, bit u, RegisterOperand first_ty,
                           RegisterOperand second_ty, string mnemonic>
    : I<(outs first_ty:$Zd), (ins second_ty:$Zn),
        mnemonic, "\t$Zd, $Zn", "", []>, Sched<[]> {
  bits<5> Zn;
  bits<4> Zd;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = sz;
  let Inst{21-10} = 0b100101111000;
  let Inst{9-5}   = Zn;
  let Inst{4-1}   = Zd;
  let Inst{0}     = u;
}

// SME2 multi-vec unpack two registers
multiclass sme2_unpk_vector_vg2<string mnemonic, bit u> {
  def _H : sme2_unpk_vector_vg2<0b01, u, ZZ_h_mul_r, ZPR8, mnemonic>;
  def _S : sme2_unpk_vector_vg2<0b10, u, ZZ_s_mul_r, ZPR16, mnemonic>;
  def _D : sme2_unpk_vector_vg2<0b11, u, ZZ_d_mul_r, ZPR32, mnemonic>;
}


class sme2_cvt_vg4_single<bit sz, bits<3> op, RegisterOperand first_ty,
                          RegisterOperand second_ty, string mnemonic>
    : I<(outs first_ty:$Zd), (ins second_ty:$Zn),
        mnemonic, "\t$Zd, $Zn", "", []>, Sched<[]> {
  bits<3> Zn;
  bits<5> Zd;
  let Inst{31-24} = 0b11000001;
  let Inst{23}    = sz;
  let Inst{22}    = op{2};
  let Inst{21-10} = 0b110011111000;
  let Inst{9-7}   = Zn;
  let Inst{6-5}   = op{1-0};
  let Inst{4-0}   = Zd;
}

// SME2 multi-vec int down convert four registers
multiclass sme2_int_cvt_vg4_single<string mnemonic, bits<3> op> {
def _StoB : sme2_cvt_vg4_single<0, op, ZPR8, ZZZZ_s_mul_r, mnemonic>;
def _DtoH : sme2_cvt_vg4_single<1, op, ZPR16, ZZZZ_d_mul_r, mnemonic>;
}

class sme2_unpk_vector_vg4<bits<2>sz, bit u, RegisterOperand first_ty,
                           RegisterOperand second_ty, string mnemonic>
    : I<(outs first_ty:$Zd), (ins second_ty:$Zn),
        mnemonic, "\t$Zd, $Zn", "", []>, Sched<[]> {
  bits<4> Zn;
  bits<3> Zd;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = sz;
  let Inst{21-10} = 0b110101111000;
  let Inst{9-6}   = Zn;
  let Inst{5}     = 0b0;
  let Inst{4-2}   = Zd;
  let Inst{1}     = 0b0;
  let Inst{0}     = u;
}

// SME2 multi-vec UNPK four registers
multiclass sme2_unpk_vector_vg4<string mnemonic, bit u> {
  def _H : sme2_unpk_vector_vg4<0b01, u, ZZZZ_h_mul_r, ZZ_b_mul_r, mnemonic>;
  def _S : sme2_unpk_vector_vg4<0b10, u, ZZZZ_s_mul_r, ZZ_h_mul_r, mnemonic>;
  def _D : sme2_unpk_vector_vg4<0b11, u, ZZZZ_d_mul_r, ZZ_s_mul_r, mnemonic>;
}

//===----------------------------------------------------------------------===//
// SME2 multi-vec CLAMP registers

class sme2_zip_clamp_vector_vg24_multi<bits<2> sz, bits<3> op1, bit u,
                                       RegisterOperand multi_vector_ty,
                                       ZPRRegOp vector_ty, string mnemonic>
    : I<(outs multi_vector_ty:$Zd),
        (ins  multi_vector_ty:$_Zd, vector_ty:$Zn, vector_ty:$Zm),
        mnemonic, "\t$Zd, $Zn, $Zm",
        "", []>, Sched<[]>{
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b11000001;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b110;
  let Inst{12-10} = op1;
  let Inst{9-5}   = Zn;
  let Inst{0}     = u;

  let Constraints = "$Zd = $_Zd";
}

class sme2_zip_clamp_vector_vg2_multi<bits<2> sz, bits<3> op1, bit u,
                                      RegisterOperand multi_vector_ty,
                                      ZPRRegOp vector_ty, string mnemonic>
    : sme2_zip_clamp_vector_vg24_multi<sz, op1, u, multi_vector_ty, vector_ty,
                                   mnemonic>{
  bits<4> Zd;
  let Inst{4-1} = Zd;
}

multiclass sme2_fp_clamp_vector_vg2_multi<string mnemonic>{
  def _H : sme2_zip_clamp_vector_vg2_multi<0b01, 0b000, 0b0, ZZ_h_mul_r, ZPR16, mnemonic>;
  def _S : sme2_zip_clamp_vector_vg2_multi<0b10, 0b000, 0b0, ZZ_s_mul_r, ZPR32, mnemonic>;
  def _D : sme2_zip_clamp_vector_vg2_multi<0b11, 0b000, 0b0, ZZ_d_mul_r, ZPR64, mnemonic>;
}

multiclass sme2_int_clamp_vector_vg2_multi<string mnemonic, bit u>{
  def _B : sme2_zip_clamp_vector_vg2_multi<0b00, 0b001, u, ZZ_b_mul_r, ZPR8, mnemonic>;
  def _H : sme2_zip_clamp_vector_vg2_multi<0b01, 0b001, u, ZZ_h_mul_r, ZPR16, mnemonic>;
  def _S : sme2_zip_clamp_vector_vg2_multi<0b10, 0b001, u, ZZ_s_mul_r, ZPR32, mnemonic>;
  def _D : sme2_zip_clamp_vector_vg2_multi<0b11, 0b001, u, ZZ_d_mul_r, ZPR64, mnemonic>;
}

multiclass sme2_zip_vector_vg2<string mnemonic, bit op> {
  def _B : sme2_zip_clamp_vector_vg2_multi<0b00, 0b100, op, ZZ_b_mul_r, ZPR8, mnemonic>;
  def _H : sme2_zip_clamp_vector_vg2_multi<0b01, 0b100, op, ZZ_h_mul_r, ZPR16, mnemonic>;
  def _S : sme2_zip_clamp_vector_vg2_multi<0b10, 0b100, op, ZZ_s_mul_r, ZPR32, mnemonic>;
  def _D : sme2_zip_clamp_vector_vg2_multi<0b11, 0b100, op, ZZ_d_mul_r, ZPR64, mnemonic>;
  def _Q : sme2_zip_clamp_vector_vg2_multi<0b00, 0b101, op, ZZ_q_mul_r, ZPR128, mnemonic>;
}

class sme2_clamp_vector_vg4_multi<bits<2> sz, bits<3> op1, bit u,
                                  RegisterOperand multi_vector_ty,
                                  ZPRRegOp vector_ty, string mnemonic>
    : sme2_zip_clamp_vector_vg24_multi<sz, op1, u,  multi_vector_ty, vector_ty,
                                   mnemonic>{
  bits<3> Zd;
  let Inst{4-2} = Zd;
  let Inst{1}   = 0b0;
}

multiclass sme2_fp_clamp_vector_vg4_multi<string mnemonic>{
  def _H : sme2_clamp_vector_vg4_multi<0b01, 0b010, 0b0, ZZZZ_h_mul_r, ZPR16, mnemonic>;
  def _S : sme2_clamp_vector_vg4_multi<0b10, 0b010, 0b0, ZZZZ_s_mul_r, ZPR32, mnemonic>;
  def _D : sme2_clamp_vector_vg4_multi<0b11, 0b010, 0b0, ZZZZ_d_mul_r, ZPR64, mnemonic>;
}

multiclass sme2_int_clamp_vector_vg4_multi<string mnemonic, bit u>{
  def _B : sme2_clamp_vector_vg4_multi<0b00, 0b011, u, ZZZZ_b_mul_r, ZPR8, mnemonic>;
  def _H : sme2_clamp_vector_vg4_multi<0b01, 0b011, u, ZZZZ_h_mul_r, ZPR16, mnemonic>;
  def _S : sme2_clamp_vector_vg4_multi<0b10, 0b011, u, ZZZZ_s_mul_r, ZPR32, mnemonic>;
  def _D : sme2_clamp_vector_vg4_multi<0b11, 0b011, u, ZZZZ_d_mul_r, ZPR64, mnemonic>;
}

//===----------------------------------------------------------------------===//
// SME2 Dot Products and MLA

// SME2 multi-vec ternary indexed two registers 32-bit
class sme2_multi_vec_array_vg2_index_32b<bits<4> op,
                                         RegisterOperand multi_vector_ty,
                                         ZPRRegOp vector_ty,
                                         string mnemonic>
    : I<(outs MatrixOp32:$ZAda),
        (ins MatrixOp32:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3,
         multi_vector_ty:$Zn, vector_ty:$Zm, VectorIndexS:$i2),
        mnemonic, "\t$ZAda[$Rv, $imm3, vgx2], $Zn, $Zm$i2",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<2> i2;
  bits<4> Zn;
  bits<3> imm3;
  let Inst{31-20} = 0b110000010101;
  let Inst{19-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-13} = Rv;
  let Inst{12}    = op{3};
  let Inst{11-10} = i2;
  let Inst{9-6}   = Zn;
  let Inst{5-3}   = op{2-0};
  let Inst{2-0}   = imm3;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme2_multi_vec_array_vg2_index_32b<string mnemonic, bits<4> op,
                                              RegisterOperand multi_vector_ty,
                                              ZPRRegOp vector_ty> {
  def NAME : sme2_multi_vec_array_vg2_index_32b<op, multi_vector_ty, vector_ty,
                                                mnemonic>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm3], $Zn, $Zm$i2",
        (!cast<Instruction>(NAME) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3,
        multi_vector_ty:$Zn, vector_ty:$Zm, VectorIndexS:$i2), 0>;
}

// SME2 multi-vec ternary indexed two registers 64-bit

class sme2_multi_vec_array_vg2_index_64b<bits<2> op,
                                         RegisterOperand multi_vector_ty,
                                         ZPRRegOp vector_ty,
                                         string mnemonic>
    : I<(outs MatrixOp64:$ZAda),
        (ins MatrixOp64:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3,
         multi_vector_ty:$Zn, vector_ty:$Zm, VectorIndexD:$i1),
        mnemonic, "\t$ZAda[$Rv, $imm3, vgx2], $Zn, $Zm$i1",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<1> i1;
  bits<4> Zn;
  bits<3> imm3;
  let Inst{31-20} = 0b110000011101;
  let Inst{19-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-13} = Rv;
  let Inst{12-11} = 0b00;
  let Inst{10}    = i1;
  let Inst{9-6}   = Zn;
  let Inst{5}     = 0b0;
  let Inst{4-3}   = op;
  let Inst{2-0}   = imm3;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme2_multi_vec_array_vg2_index_64b<string mnemonic, bits<2> op,
                                              RegisterOperand multi_vector_ty,
                                              ZPRRegOp vector_ty> {
  def NAME : sme2_multi_vec_array_vg2_index_64b<op, multi_vector_ty, vector_ty,
                                                mnemonic>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm3], $Zn, $Zm$i1",
        (!cast<Instruction>(NAME) MatrixOp64:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3,
        multi_vector_ty:$Zn, vector_ty:$Zm, VectorIndexD:$i1), 0>;
}

// SME2 multi-vec ternary indexed four registers 32-bit

class sme2_multi_vec_array_vg4_index_32b<bits<4> op,
                                         RegisterOperand multi_vector_ty,
                                         ZPRRegOp vector_ty,
                                         string mnemonic>
    : I<(outs MatrixOp32:$ZAda),
        (ins MatrixOp32:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3,
         multi_vector_ty:$Zn, vector_ty:$Zm, VectorIndexS:$i2),
        mnemonic, "\t$ZAda[$Rv, $imm3, vgx4], $Zn, $Zm$i2",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<2> i2;
  bits<3> Zn;
  bits<3> imm3;
  let Inst{31-20} = 0b110000010101;
  let Inst{19-16} = Zm;
  let Inst{15}    = 0b1;
  let Inst{14-13} = Rv;
  let Inst{12}    = op{3};
  let Inst{11-10} = i2;
  let Inst{9-7}   = Zn;
  let Inst{6}     = 0b0;
  let Inst{5-3}   = op{2-0};
  let Inst{2-0}   = imm3;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme2_multi_vec_array_vg4_index_32b<string mnemonic, bits<4> op,
                                              RegisterOperand multi_vector_ty,
                                              ZPRRegOp vector_ty> {
  def NAME : sme2_multi_vec_array_vg4_index_32b<op, multi_vector_ty, vector_ty,
                                                mnemonic>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm3], $Zn, $Zm$i2",
        (!cast<Instruction>(NAME) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3,
        multi_vector_ty:$Zn, vector_ty:$Zm, VectorIndexS:$i2), 0>;
}

// SME2 multi-vec ternary indexed four registers 64-bit
class sme2_multi_vec_array_vg4_index_64b<bits<3> op,
                                         RegisterOperand multi_vector_ty,
                                         ZPRRegOp vector_ty,
                                         string mnemonic>
    : I<(outs MatrixOp64:$ZAda),
        (ins MatrixOp64:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3,
         multi_vector_ty:$Zn, vector_ty:$Zm, VectorIndexD:$i1),
        mnemonic, "\t$ZAda[$Rv, $imm3, vgx4], $Zn, $Zm$i1",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<1> i1;
  bits<3> Zn;
  bits<3> imm3;
  let Inst{31-20} = 0b110000011101;
  let Inst{19-16} = Zm;
  let Inst{15}    = 0b1;
  let Inst{14-13} = Rv;
  let Inst{12}    = 0b0;
  let Inst{11}    = op{2};
  let Inst{10}    = i1;
  let Inst{9-7}   = Zn;
  let Inst{6-5}   = 0b00;
  let Inst{4-3}   = op{1-0};
  let Inst{2-0}   = imm3;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme2_multi_vec_array_vg4_index_64b<string mnemonic, bits<3> op,
                                              RegisterOperand multi_vector_ty,
                                              ZPRRegOp vector_ty> {
  def NAME : sme2_multi_vec_array_vg4_index_64b<op, multi_vector_ty, vector_ty,
                                                mnemonic>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm3], $Zn, $Zm$i1",
        (!cast<Instruction>(NAME) MatrixOp64:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, sme_elm_idx0_7:$imm3,
        multi_vector_ty:$Zn, vector_ty:$Zm, VectorIndexD:$i1), 0>;
}
//===----------------------------------------------------------------------===//
// SME2 multi-vec indexed long long MLA one source 32-bit
class sme2_mla_ll_array_index_32b<string mnemonic, bits<3> op>
    : I<(outs MatrixOp32:$ZAda),
        (ins MatrixOp32:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm2s4range:$imm2, ZPR8:$Zn, ZPR4b8:$Zm, VectorIndexB:$i),
        mnemonic, "\t$ZAda[$Rv, $imm2], $Zn, $Zm$i",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<4> i;
  bits<5> Zn;
  bits<2> imm2;
  let Inst{31-20} = 0b110000010000;
  let Inst{19-16} = Zm;
  let Inst{15}    = i{3};
  let Inst{14-13} = Rv;
  let Inst{12-10} = i{2-0};
  let Inst{9-5}   = Zn;
  let Inst{4-2}   = op;
  let Inst{1-0}   = imm2;

  let Constraints = "$ZAda = $_ZAda";
}

// SME2 multi-vec indexed long long MLA one source 64-bit

class sme2_mla_ll_array_index_64b<string mnemonic, bits<2> op>
    : I<(outs MatrixOp64:$ZAda),
        (ins MatrixOp64:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm2s4range:$imm2, ZPR16:$Zn, ZPR4b16:$Zm, VectorIndexH:$i),
        mnemonic, "\t$ZAda[$Rv, $imm2], $Zn, $Zm$i",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<3> i;
  bits<5> Zn;
  bits<2> imm2;
  let Inst{31-20} = 0b110000011000;
  let Inst{19-16} = Zm;
  let Inst{15}    = i{2};
  let Inst{14-13} = Rv;
  let Inst{12}    = 0b0;
  let Inst{11-10} = i{1-0};
  let Inst{9-5}   = Zn;
  let Inst{4-3}   = op;
  let Inst{2}     = 0b0;
  let Inst{1-0}   = imm2;

  let Constraints = "$ZAda = $_ZAda";
}

class sme2_mla_ll_array_vg24_index_32b<bit vg4, bits<3> op,
                                       RegisterOperand vector_ty,
                                       string mnemonic>
    : I<(outs MatrixOp32:$ZAda),
        (ins MatrixOp32:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm,
             vector_ty:$Zn, ZPR4b8:$Zm, VectorIndexB:$i),
        mnemonic, "\t$ZAda[$Rv, $imm, " # !if(vg4, "vgx4", "vgx2") # "], $Zn, $Zm$i",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<4> i;
  bit     imm;
  let Inst{31-20} = 0b110000010001;
  let Inst{19-16} = Zm;
  let Inst{15}    = vg4;
  let Inst{14-13} = Rv;
  let Inst{12}    = 0b0;
  let Inst{11-10} = i{3-2};
  let Inst{5-3}   = op;
  let Inst{2-1}   = i{1-0};
  let Inst{0}     = imm;

  let Constraints = "$ZAda = $_ZAda";
}

//SME2 multi-vec indexed long long MLA two sources 32-bit

multiclass sme2_mla_ll_array_vg2_index_32b<string mnemonic, bits<3> op> {
  def NAME: sme2_mla_ll_array_vg24_index_32b<0b0, op, ZZ_b_mul_r, mnemonic> {
   bits<4> Zn;
   let Inst{9-6} = Zn;
  }

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm$i",
                 (!cast<Instruction>(NAME) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm, ZZ_b_mul_r:$Zn, ZPR4b8:$Zm, VectorIndexB:$i), 0>;
}

// SME2 multi-vec indexed long long MLA four sources 32-bit

multiclass sme2_mla_ll_array_vg4_index_32b<string mnemonic, bits<3> op> {
  def NAME: sme2_mla_ll_array_vg24_index_32b<0b1, op, ZZZZ_b_mul_r, mnemonic> {
   bits<3> Zn;
   let Inst{9-7} = Zn;
   let Inst{6}   = 0b0;
  }

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm$i",
                 (!cast<Instruction>(NAME) MatrixOp32:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm, ZZZZ_b_mul_r:$Zn, ZPR4b8:$Zm, VectorIndexB:$i), 0>;
}
class sme2_mla_ll_array_vg24_index_64b<bit vg4,  bits<2> op,
                                       RegisterOperand vector_ty,
                                       string mnemonic>
    : I<(outs MatrixOp64:$ZAda),
        (ins MatrixOp64:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm,
             vector_ty:$Zn, ZPR4b16:$Zm, VectorIndexH:$i),
        mnemonic, "\t$ZAda[$Rv, $imm, " # !if(vg4, "vgx4", "vgx2") # "], $Zn, $Zm$i",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<3> i;
  bit     imm;
  let Inst{31-20} = 0b110000011001;
  let Inst{19-16} = Zm;
  let Inst{15}    = vg4;
  let Inst{14-13} = Rv;
  let Inst{12-11} = 0b00;
  let Inst{10}    = i{2};
  let Inst{5}     = 0b0;
  let Inst{4-3}   = op;
  let Inst{2-1}   = i{1-0};
  let Inst{0}     = imm;

  let Constraints = "$ZAda = $_ZAda";
}

// SME2 multi-vec indexed long long MLA two sources 64-bit

multiclass sme2_mla_ll_array_vg2_index_64b<string mnemonic, bits<2> op> {
  def NAME: sme2_mla_ll_array_vg24_index_64b<0b0, op, ZZ_h_mul_r, mnemonic>{
    bits<4> Zn;
    let Inst{9-6} = Zn;
  }

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm$i",
                 (!cast<Instruction>(NAME) MatrixOp64:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm, ZZ_h_mul_r:$Zn, ZPR4b16:$Zm, VectorIndexH:$i), 0>;
}

// SME2 multi-vec indexed long long MLA four sources 64-bit

multiclass sme2_mla_ll_array_vg4_index_64b<string mnemonic, bits<2> op> {
  def NAME: sme2_mla_ll_array_vg24_index_64b<0b1, op, ZZZZ_h_mul_r,  mnemonic>{
    bits<3> Zn;
    let Inst{9-7} = Zn;
    let Inst{6}   = 0b0;
  }

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm$i",
                 (!cast<Instruction>(NAME) MatrixOp64:$ZAda,  MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm, ZZZZ_h_mul_r:$Zn, ZPR4b16:$Zm, VectorIndexH:$i), 0>;
}


//SME2 multiple and single vector long long FMA one source

class sme2_mla_ll_array_single<string mnemonic, bits<4> op,
                               MatrixOperand matrix_ty, ZPRRegOp vector_ty,
                               ZPRRegOp zpr_ty>
    : I<(outs matrix_ty:$ZAda),
        (ins matrix_ty:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm2s4range:$imm,
             vector_ty:$Zn, zpr_ty:$Zm),
        mnemonic, "\t$ZAda[$Rv, $imm], $Zn, $Zm",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<5> Zn;
  bits<2> imm;
  let Inst{31-23} = 0b110000010;
  let Inst{22}    = op{3}; //sz
  let Inst{21-20} = 0b10;
  let Inst{19-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-13} = Rv;
  let Inst{12-10} = 0b001;
  let Inst{9-5}   = Zn;
  let Inst{4-2}   = op{2-0};
  let Inst{1-0}   = imm;

  let Constraints = "$ZAda = $_ZAda";
}

class sme2_mla_ll_array_vg24_single<bits<5> op, MatrixOperand matrix_ty,
                                    RegisterOperand vector_ty, ZPRRegOp zpr_ty,
                                    string mnemonic>
    : I<(outs matrix_ty:$ZAda),
        (ins matrix_ty:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm,
             vector_ty:$Zn, zpr_ty:$Zm),
        mnemonic, "\t$ZAda[$Rv, $imm,  " # !if(op{3}, "vgx4", "vgx2") # "], $Zn, $Zm",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<5> Zn;
  bit     imm;
  let Inst{31-23} = 0b110000010;
  let Inst{22}    = op{4}; //sz
  let Inst{21}    = 0b1;
  let Inst{20}    = op{3}; //vg4
  let Inst{19-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-13} = Rv;
  let Inst{12-10} = 0b000;
  let Inst{9-5}   = Zn;
  let Inst{4-2}   = op{2-0};
  let Inst{1}     = 0b0;
  let Inst{0}     = imm;

  let Constraints = "$ZAda = $_ZAda";
}

//SME2 single-multi long long MLA two and four sources

multiclass sme2_mla_ll_array_vg24_single<string mnemonic, bits<5> op,
                                          MatrixOperand matrix_ty,
                                          RegisterOperand multi_vector_ty,
                                          ZPRRegOp zpr_ty> {
  def NAME: sme2_mla_ll_array_vg24_single<op, matrix_ty, multi_vector_ty,
                                          zpr_ty, mnemonic>;

  def : InstAlias<mnemonic # "\t$ZAd[$Rv, $imm], $Zn, $Zm",
                 (!cast<Instruction>(NAME) matrix_ty:$ZAd,  MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm, multi_vector_ty:$Zn, zpr_ty:$Zm), 0>;
}

// SME2 multiple vectors long long MLA two sources

class sme2_mla_ll_array_vg2_multi<bits<4> op, MatrixOperand matrix_ty,
                                  RegisterOperand vector_ty,string mnemonic>
    : I<(outs matrix_ty:$ZAda),
        (ins matrix_ty:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm,
             vector_ty:$Zn, vector_ty:$Zm),
        mnemonic, "\t$ZAda[$Rv, $imm, vgx2], $Zn, $Zm",
        "", []>, Sched<[]> {
  bits<4> Zm;
  bits<2> Rv;
  bits<4> Zn;
  bit     imm;
  let Inst{31-23} = 0b110000011;
  let Inst{22}    = op{3};  // sz
  let Inst{21}    = 0b1;
  let Inst{20-17} = Zm;
  let Inst{16-15} = 0b00;
  let Inst{14-13} = Rv;
  let Inst{12-10} = 0b000;
  let Inst{9-6}   = Zn;
  let Inst{5}     = 0b0;
  let Inst{4-2}   = op{2-0};
  let Inst{1}     = 0b0;
  let Inst{0}     = imm;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme2_mla_ll_array_vg2_multi<string mnemonic, bits<4> op,
                                       MatrixOperand matrix_ty,
                                       RegisterOperand vector_ty> {
  def NAME : sme2_mla_ll_array_vg2_multi<op, matrix_ty, vector_ty, mnemonic>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm",
                 (!cast<Instruction>(NAME) matrix_ty:$ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm, vector_ty:$Zn, vector_ty:$Zm), 0>;
}

// SME2 multiple vectors long long MLA four sources

class sme2_mla_ll_array_vg4_multi<bits<4> op,MatrixOperand matrix_ty,
                                  RegisterOperand vector_ty,
                                  string mnemonic>
    : I<(outs matrix_ty:$ZAda),
        (ins matrix_ty:$_ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm,
             vector_ty:$Zn, vector_ty:$Zm),
        mnemonic, "\t$ZAda[$Rv, $imm, vgx4], $Zn, $Zm",
        "", []>, Sched<[]> {
  bits<3> Zm;
  bits<2> Rv;
  bits<3> Zn;
  bit     imm;
  let Inst{31-23} = 0b110000011;
  let Inst{22}    = op{3}; // sz
  let Inst{21}    = 0b1;
  let Inst{20-18} = Zm;
  let Inst{17-15} = 0b010;
  let Inst{14-13} = Rv;
  let Inst{12-10} = 0b000;
  let Inst{9-7}   = Zn;
  let Inst{6-5}   = 0b00;
  let Inst{4-2}   = op{2-0};
  let Inst{1}     = 0b0;
  let Inst{0}     = imm;

  let Constraints = "$ZAda = $_ZAda";
}

multiclass sme2_mla_ll_array_vg4_multi<string mnemonic, bits<4> op,
                                       MatrixOperand matrix_ty,
                                       RegisterOperand vector_ty> {
  def NAME : sme2_mla_ll_array_vg4_multi<op, matrix_ty, vector_ty, mnemonic>;

  def : InstAlias<mnemonic # "\t$ZAda[$Rv, $imm], $Zn, $Zm",
                 (!cast<Instruction>(NAME) matrix_ty:$ZAda, MatrixIndexGPR32Op8_11:$Rv, uimm1s4range:$imm, vector_ty:$Zn, vector_ty:$Zm), 0>;
}

//===----------------------------------------------------------------------===//
// SME2 Outer Product and Accumulate

multiclass sme2_int_mopx_tile<string mnemonic, bits<3> op> {
  def NAME : sme_int_outer_product_inst<op, 0b0, 0b1, TileOp32, ZPR16, mnemonic> {
    bits<2> ZAda;
    let Inst{1-0} = ZAda;
    let Inst{2}   = 0b0;
  }
}

multiclass  sme2_bfp_mopx_tile<string mnemonic, bits<3> op> {
  def NAME : sme_outer_product_widening_inst<op, ZPR32, mnemonic>;
}

//===----------------------------------------------------------------------===///
// SME2 Zero Lookup Table.
class sme2_zero_zt<string mnemonic, bits<4> opc>
    : I<(outs ZTR:$ZT), (ins ),
         mnemonic, "\t\\{ $ZT \\}",
         "", []>, Sched<[]> {
  let Inst{31-4} = 0b1100000001001000000000000000;
  let Inst{3-0}  = opc;
}

//===----------------------------------------------------------------------===//
// SME2 lookup table load/store
class sme2_spill_fill_vector<string mnemonic, bits<8> opc>
    : I<!if(opc{7}, (outs ), (outs ZTR:$ZTt)),
        !if(opc{7}, (ins ZTR:$ZTt, GPR64sp:$Rn), (ins GPR64sp:$Rn)),
        mnemonic, "\t$ZTt, [$Rn]",
        "", []>, Sched<[]> {
  bits<5> Rn;
  let Inst{31-22} = 0b1110000100;
  let Inst{21-16} = opc{7-2};
  let Inst{15-10} = 0b100000;
  let Inst{9-5}   = Rn;
  let Inst{4-2}   = 0b000;
  let Inst{1-0}   = opc{1-0};

  let mayLoad     = !not(opc{7});
  let mayStore    = opc{7};
}

//===----------------------------------------------------------------------===///
// SME2 move to/from lookup table
class sme2_movt_zt_to_scalar<string mnemonic, bits<7> opc>
    : I<(outs GPR64:$Rt), (ins ZTR:$ZTt, uimm3s8:$imm3),
         mnemonic, "\t$Rt, $ZTt$imm3",
         "", []>, Sched<[]> {
  bits<3> imm3;
  bits<5> Rt;
  let Inst{31-15} = 0b11000000010011000;
  let Inst{14-12} = imm3;
  let Inst{11-5}  = opc;
  let Inst{4-0}   = Rt;
}

class sme2_movt_scalar_to_zt<string mnemonic, bits<7> opc>
    : I<(outs ZTR:$ZTt), (ins uimm3s8:$imm3, GPR64:$Rt),
         mnemonic, "\t$ZTt$imm3, $Rt",
         "", []>, Sched<[]> {
  bits<3> imm3;
  bits<5> Rt;
  let Inst{31-15} = 0b11000000010011100;
  let Inst{14-12} = imm3;
  let Inst{11-5}  = opc;
  let Inst{4-0}   = Rt;
}

//===----------------------------------------------------------------------===//
// SME2 lookup table expand one register
class sme2_luti_vector_index<bits<2> sz, bits<7> opc, RegisterOperand vector_ty,
                             AsmVectorIndexOpnd index_ty, string mnemonic>
    : I<(outs vector_ty:$Zd),
        (ins ZTR:$ZTt, ZPRAny:$Zn, index_ty:$i),
        mnemonic, "\t$Zd, $ZTt, $Zn$i",
        "", []>, Sched<[]> {
  bits<5> Zn;
  bits<5> Zd;
  let Inst{31-19} = 0b1100000011001;
  let Inst{18-14} = opc{6-2};
  let Inst{13-12} = sz;
  let Inst{11-10} = opc{1-0};
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

class sme2_luti2_vector_index<bits<2> sz, RegisterOperand vector_ty,
                              string mnemonic>
    : sme2_luti_vector_index<sz, {1,?,?,?,?,0,0}, vector_ty, VectorIndexB, mnemonic> {
  bits<4> i;
  let Inst{17-14} = i;
}

multiclass sme2_luti2_vector_index<string mnemonic> {
  def _B : sme2_luti2_vector_index<0b00, ZPR8, mnemonic>;
  def _H : sme2_luti2_vector_index<0b01, ZPR16, mnemonic>;
  def _S : sme2_luti2_vector_index<0b10, ZPR32, mnemonic>;
}

class sme2_luti4_vector_index<bits<2> sz, RegisterOperand vector_ty,
                              string mnemonic>
    : sme2_luti_vector_index<sz, {0,1,?,?,?,0,0}, vector_ty, VectorIndexH, mnemonic> {
  bits<3> i;
  let Inst{16-14} = i;
}

multiclass sme2_luti4_vector_index<string mnemonic> {
  def _B : sme2_luti4_vector_index<0b00, ZPR8, mnemonic>;
  def _H : sme2_luti4_vector_index<0b01, ZPR16, mnemonic>;
  def _S : sme2_luti4_vector_index<0b10, ZPR32, mnemonic>;
}

// SME2 lookup table expand two contiguous registers
class sme2_luti_vector_vg2_index<bits<2> sz, bits<6> opc, RegisterOperand vector_ty,
                                 AsmVectorIndexOpnd index_ty, string mnemonic>
    : I<(outs vector_ty:$Zd),
        (ins ZTR:$ZTt, ZPRAny:$Zn, index_ty:$i),
        mnemonic, "\t$Zd, $ZTt, $Zn$i",
        "", []>, Sched<[]> {
  bits<5> Zn;
  bits<4> Zd;
  let Inst{31-19} = 0b1100000010001;
  let Inst{18-15} = opc{5-2};
  let Inst{14}    = 0b1;
  let Inst{13-12} = sz;
  let Inst{11-10} = opc{1-0};
  let Inst{9-5}   = Zn;
  let Inst{4-1}   = Zd;
  let Inst{0}     = 0b0;
}

class sme2_luti2_vector_vg2_index<bits<2> sz, RegisterOperand vector_ty,
                                  string mnemonic>
    : sme2_luti_vector_vg2_index<sz, {1,?,?,?,0,0}, vector_ty, VectorIndexH, mnemonic> {
  bits<3> i;
  let Inst{17-15} = i;
}

multiclass sme2_luti2_vector_vg2_index<string mnemonic> {
  def _B : sme2_luti2_vector_vg2_index<0b00, ZZ_b_mul_r, mnemonic>;
  def _H : sme2_luti2_vector_vg2_index<0b01, ZZ_h_mul_r, mnemonic>;
  def _S : sme2_luti2_vector_vg2_index<0b10, ZZ_s_mul_r, mnemonic>;
}

class sme2_luti4_vector_vg2_index<bits<2> sz, RegisterOperand vector_ty,
                                 string mnemonic>
    : sme2_luti_vector_vg2_index<sz, {0,1,?,?,0,0}, vector_ty, VectorIndexS, mnemonic> {
  bits<2> i;
  let Inst{16-15} = i;
}

multiclass sme2_luti4_vector_vg2_index<string mnemonic> {
  def _B : sme2_luti4_vector_vg2_index<0b00, ZZ_b_mul_r, mnemonic>;
  def _H : sme2_luti4_vector_vg2_index<0b01, ZZ_h_mul_r, mnemonic>;
  def _S : sme2_luti4_vector_vg2_index<0b10, ZZ_s_mul_r, mnemonic>;
}

// SME2 lookup table expand four contiguous registers
class sme2_luti_vector_vg4_index<bits<2> sz, bits<5>opc, RegisterOperand vector_ty,
                                 AsmVectorIndexOpnd index_ty, string mnemonic>
    : I<(outs vector_ty:$Zd),
        (ins ZTR:$ZTt, ZPRAny:$Zn, index_ty:$i),
        mnemonic, "\t$Zd, $ZTt, $Zn$i",
        "", []>, Sched<[]> {
  bits<5> Zn;
  bits<3> Zd;
  let Inst{31-19} = 0b1100000010001;
  let Inst{18-16} = opc{4-2};
  let Inst{15-14} = 0b10;
  let Inst{13-12} = sz;
  let Inst{11-10} = opc{1-0};
  let Inst{9-5}   = Zn;
  let Inst{4-2}   = Zd;
  let Inst{1-0}   = 0b00;
}

class sme2_luti2_vector_vg4_index<bits<2> sz, RegisterOperand vector_ty,
                                  string mnemonic>
    : sme2_luti_vector_vg4_index<sz, {1,?,?,0,0}, vector_ty, VectorIndexS, mnemonic> {
  bits<2> i;
  let Inst{17-16} = i;
}

multiclass sme2_luti2_vector_vg4_index<string mnemonic> {
  def _B : sme2_luti2_vector_vg4_index<0b00, ZZZZ_b_mul_r, mnemonic>;
  def _H : sme2_luti2_vector_vg4_index<0b01, ZZZZ_h_mul_r, mnemonic>;
  def _S : sme2_luti2_vector_vg4_index<0b10, ZZZZ_s_mul_r, mnemonic>;
}

class sme2_luti4_vector_vg4_index<bits<2> sz, RegisterOperand vector_ty,
                                  string mnemonic>
    : sme2_luti_vector_vg4_index<sz, {0,1,?,0,0}, vector_ty, VectorIndexD, mnemonic> {
  bits<1> i;
  let Inst{16}    = i;
}

multiclass sme2_luti4_vector_vg4_index<string mnemonic> {
  def _H : sme2_luti4_vector_vg4_index<0b01, ZZZZ_h_mul_r, mnemonic>;
  def _S : sme2_luti4_vector_vg4_index<0b10, ZZZZ_s_mul_r, mnemonic>;
}

