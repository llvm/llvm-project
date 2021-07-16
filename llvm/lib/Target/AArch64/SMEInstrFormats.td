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

//===----------------------------------------------------------------------===//
// SME Outer Products
//===----------------------------------------------------------------------===//

class sme_fp_outer_product_inst<bit S, bit sz, MatrixTileOperand za_ty,
                                ZPRRegOp zpr_ty, string mnemonic>
    : I<(outs za_ty:$ZAda),
        (ins PPR3bAny:$Pn, PPR3bAny:$Pm, zpr_ty:$Zn, zpr_ty:$Zm),
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
        (ins PPR3bAny:$Pn, PPR3bAny:$Pm, zpr_ty:$Zn, zpr_ty:$Zm),
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
        (ins PPR3bAny:$Pn, PPR3bAny:$Pm, ZPR16:$Zn, ZPR16:$Zm),
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
        (ins PPR3bAny:$Pn, PPR3bAny:$Pm, zpr_ty:$Zn),
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

class sme_mem_ld_ss_inst_BHSD<bits<2> msz, string mnemonic,
                              MatrixTileVectorOperand tile_ty, bit is_col,
                              Operand imm_ty, RegisterOperand gpr_ty>
    : sme_mem_ld_ss_base<
        0b0, is_col, msz, (outs tile_ty:$ZAt),
        (ins MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, GPR64sp:$Rn,
             gpr_ty:$Rm),
        mnemonic, "\t\\{$ZAt[$Rv, $imm]\\}, $Pg/z, [$Rn, $Rm]">;

class sme_mem_ld_ss_inst_Q<string mnemonic, MatrixTileVectorOperand tile_ty,
                           bit is_col>
    : sme_mem_ld_ss_base<
        0b1, is_col, 0b11, (outs tile_ty:$ZAt),
        (ins MatrixIndexGPR32Op12_15:$Rv, PPR3bAny:$Pg, GPR64sp:$Rn,
             GPR64shifted128:$Rm),
        mnemonic, "\t\\{$ZAt[$Rv]\\}, $Pg/z, [$Rn, $Rm]">;

multiclass sme_mem_ss_aliases_BHSD<string mnemonic, Instruction inst,
                                   MatrixTileVectorOperand tile_ty, Operand imm_ty,
                                   RegisterOperand gpr_ty,
                                   string pg_suffix=""> {
  def : InstAlias<mnemonic # "\t$ZAt[$Rv, $imm], $Pg" # pg_suffix # ", [$Rn, $Rm]",
                  (inst tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, GPR64sp:$Rn, gpr_ty:$Rm), 0>;
  // Default XZR offset aliases
  def : InstAlias<mnemonic # "\t\\{$ZAt[$Rv, $imm]\\}, $Pg" # pg_suffix # ", [$Rn]",
                  (inst tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, GPR64sp:$Rn, XZR), 1>;
  def : InstAlias<mnemonic # "\t$ZAt[$Rv, $imm], $Pg" # pg_suffix # ", [$Rn]",
                  (inst tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, GPR64sp:$Rn, XZR), 0>;
}

multiclass sme_mem_ss_aliases_Q<string mnemonic, Instruction inst,
                                MatrixTileVectorOperand tile_ty,
                                string pg_suffix=""> {
  def : InstAlias<mnemonic # "\t$ZAt[$Rv], $Pg" # pg_suffix # ", [$Rn, $Rm]",
                  (inst tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, PPR3bAny:$Pg, GPR64sp:$Rn, GPR64shifted128:$Rm), 0>;
  // Default XZR offset aliases
  def : InstAlias<mnemonic # "\t\\{$ZAt[$Rv]\\}, $Pg" # pg_suffix # ", [$Rn]",
                  (inst tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, PPR3bAny:$Pg, GPR64sp:$Rn, XZR), 2>;
  def : InstAlias<mnemonic # "\t$ZAt[$Rv], $Pg" # pg_suffix # ", [$Rn]",
                  (inst tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, PPR3bAny:$Pg, GPR64sp:$Rn, XZR), 0>;
}

multiclass sme_mem_ss_aliases<string mnemonic, string inst, bit is_col,
                              string pg_suffix=""> {
  defm : sme_mem_ss_aliases_BHSD<mnemonic # "b", !cast<Instruction>(inst # _B),
                                 !if(is_col, TileVectorOpV8, TileVectorOpH8),
                                 imm0_15, GPR64shifted8, pg_suffix>;
  defm : sme_mem_ss_aliases_BHSD<mnemonic # "h", !cast<Instruction>(inst # _H),
                                 !if(is_col, TileVectorOpV16, TileVectorOpH16),
                                 imm0_7, GPR64shifted16, pg_suffix>;
  defm : sme_mem_ss_aliases_BHSD<mnemonic # "w", !cast<Instruction>(inst # _S),
                                 !if(is_col, TileVectorOpV32, TileVectorOpH32),
                                 imm0_3, GPR64shifted32, pg_suffix>;
  defm : sme_mem_ss_aliases_BHSD<mnemonic # "d", !cast<Instruction>(inst # _D),
                                 !if(is_col, TileVectorOpV64, TileVectorOpH64),
                                 imm0_1, GPR64shifted64, pg_suffix>;
  defm : sme_mem_ss_aliases_Q   <mnemonic # "q", !cast<Instruction>(inst # _Q),
                                 !if(is_col, TileVectorOpV128, TileVectorOpH128),
                                 pg_suffix>;
}

multiclass sme_mem_ld_ss_aliases<string inst, bit is_col> {
  defm NAME : sme_mem_ss_aliases<"ld1", inst, is_col, "/z">;
}

multiclass sme_mem_ld_v_ss<string mnemonic, bit is_col> {
  def _B : sme_mem_ld_ss_inst_BHSD<0b00, mnemonic # "b",
                                   !if(is_col, TileVectorOpV8,
                                               TileVectorOpH8),
                                   is_col, imm0_15, GPR64shifted8> {
    bits<4> imm;
    let Inst{3-0} = imm;
  }
  def _H : sme_mem_ld_ss_inst_BHSD<0b01, mnemonic # "h",
                                   !if(is_col, TileVectorOpV16,
                                               TileVectorOpH16),
                                   is_col, imm0_7, GPR64shifted16> {
    bits<1> ZAt;
    bits<3> imm;
    let Inst{3}   = ZAt;
    let Inst{2-0} = imm;
  }
  def _S : sme_mem_ld_ss_inst_BHSD<0b10, mnemonic # "w",
                                   !if(is_col, TileVectorOpV32,
                                               TileVectorOpH32),
                                   is_col, imm0_3, GPR64shifted32> {
    bits<2> ZAt;
    bits<2> imm;
    let Inst{3-2} = ZAt;
    let Inst{1-0} = imm;
  }
  def _D : sme_mem_ld_ss_inst_BHSD<0b11, mnemonic # "d",
                                   !if(is_col, TileVectorOpV64,
                                               TileVectorOpH64),
                                   is_col, imm0_1, GPR64shifted64> {
    bits<3> ZAt;
    bits<1> imm;
    let Inst{3-1} = ZAt;
    let Inst{0}   = imm;
  }
  def _Q : sme_mem_ld_ss_inst_Q<mnemonic # "q",
                                !if(is_col, TileVectorOpV128,
                                            TileVectorOpH128),
                                is_col> {
    bits<4> ZAt;
    let Inst{3-0} = ZAt;
  }

  defm : sme_mem_ld_ss_aliases<NAME, is_col>;
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

class sme_mem_st_ss_inst_BHSD<bits<2> msz, string mnemonic,
                              MatrixTileVectorOperand tile_ty, bit is_col,
                              Operand imm_ty, RegisterOperand gpr_ty>
    : sme_mem_st_ss_base<
        0b0, is_col, msz,
        (ins tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg,
             GPR64sp:$Rn, gpr_ty:$Rm),
        mnemonic, "\t\\{$ZAt[$Rv, $imm]\\}, $Pg, [$Rn, $Rm]">;

class sme_mem_st_ss_inst_Q<string mnemonic, MatrixTileVectorOperand tile_ty,
                           bit is_col>
    : sme_mem_st_ss_base<
        0b1, is_col, 0b11,
        (ins tile_ty:$ZAt, MatrixIndexGPR32Op12_15:$Rv, PPR3bAny:$Pg,
             GPR64sp:$Rn, GPR64shifted128:$Rm),
        mnemonic, "\t\\{$ZAt[$Rv]\\}, $Pg, [$Rn, $Rm]">;

multiclass sme_mem_st_ss_aliases<string inst, bit is_col> {
  defm NAME : sme_mem_ss_aliases<"st1", inst, is_col>;
}

multiclass sme_mem_st_v_ss<string mnemonic, bit is_col> {
  def _B : sme_mem_st_ss_inst_BHSD<0b00, mnemonic # "b",
                                   !if(is_col, TileVectorOpV8,
                                               TileVectorOpH8),
                                   is_col, imm0_15, GPR64shifted8> {
    bits<4> imm;
    let Inst{3-0} = imm;
  }
  def _H : sme_mem_st_ss_inst_BHSD<0b01, mnemonic # "h",
                                   !if(is_col, TileVectorOpV16,
                                               TileVectorOpH16),
                                   is_col, imm0_7, GPR64shifted16> {
    bits<1> ZAt;
    bits<3> imm;
    let Inst{3}   = ZAt;
    let Inst{2-0} = imm;
  }
  def _S : sme_mem_st_ss_inst_BHSD<0b10, mnemonic # "w",
                                   !if(is_col, TileVectorOpV32,
                                               TileVectorOpH32),
                                   is_col, imm0_3, GPR64shifted32> {
    bits<2> ZAt;
    bits<2> imm;
    let Inst{3-2} = ZAt;
    let Inst{1-0} = imm;
  }
  def _D : sme_mem_st_ss_inst_BHSD<0b11, mnemonic # "d",
                                   !if(is_col, TileVectorOpV64,
                                               TileVectorOpH64),
                                   is_col, imm0_1, GPR64shifted64> {
    bits<3> ZAt;
    bits<1> imm;
    let Inst{3-1} = ZAt;
    let Inst{0}   = imm;
  }
  def _Q : sme_mem_st_ss_inst_Q<mnemonic # "q",
                                !if(is_col, TileVectorOpV128,
                                            TileVectorOpH128),
                                is_col> {
    bits<4> ZAt;
    let Inst{3-0} = ZAt;
  }

  defm : sme_mem_st_ss_aliases<NAME, is_col>;
}

multiclass sme_mem_st_ss<string mnemonic> {
  defm _H : sme_mem_st_v_ss<mnemonic, /*is_col=*/0b0>;
  defm _V : sme_mem_st_v_ss<mnemonic, /*is_col=*/0b1>;
}

