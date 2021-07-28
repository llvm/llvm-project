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

//===----------------------------------------------------------------------===//
// SME Save and Restore Array
//===----------------------------------------------------------------------===//

class sme_spill_fill_inst<bit isStore, dag outs, dag ins, string opcodestr>
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

  let mayLoad = !not(isStore);
  let mayStore = isStore;
}

multiclass sme_spill_fill<bit isStore, dag outs, dag ins, string opcodestr> {
  def NAME : sme_spill_fill_inst<isStore, outs, ins, opcodestr>;

  def : InstAlias<opcodestr # "\t$ZAt[$Rv, $imm4], [$Rn]",
                  (!cast<Instruction>(NAME) MatrixOp:$ZAt,
                   MatrixIndexGPR32Op12_15:$Rv, imm0_15:$imm4, GPR64sp:$Rn, 0), 1>;
}

multiclass sme_spill<string opcodestr> {
  defm NAME : sme_spill_fill<0b1, (outs),
                             (ins MatrixOp:$ZAt, MatrixIndexGPR32Op12_15:$Rv,
                                  imm0_15:$imm4, GPR64sp:$Rn,
                                  imm0_15:$offset),
                             opcodestr>;
}

multiclass sme_fill<string opcodestr> {
  defm NAME : sme_spill_fill<0b0, (outs MatrixOp:$ZAt),
                             (ins MatrixIndexGPR32Op12_15:$Rv,
                                  imm0_15:$imm4, GPR64sp:$Rn,
                                  imm0_15:$offset),
                             opcodestr>;
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

class sme_vector_to_tile_inst<bits<2> sz, MatrixTileVectorOperand tile_ty,
                              bit is_col, Operand imm_ty, ZPRRegOp zpr_ty,
                              string mnemonic>
    : sme_vector_to_tile_base<0b0, is_col, sz, (outs tile_ty:$ZAd),
        (ins MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm, PPR3bAny:$Pg, zpr_ty:$Zn),
        mnemonic, "\t$ZAd[$Rv, $imm], $Pg/m, $Zn">;

class sme_vector_to_tile_inst_Q<MatrixTileVectorOperand tile_ty,
                                bit is_col, string mnemonic>
    : sme_vector_to_tile_base<0b1, is_col, 0b11, (outs tile_ty:$ZAd),
        (ins MatrixIndexGPR32Op12_15:$Rv, PPR3bAny:$Pg, ZPR128:$Zn),
        mnemonic, "\t$ZAd[$Rv], $Pg/m, $Zn">;

multiclass sme_vector_to_tile_aliases<Instruction inst,
                                      MatrixTileVectorOperand tile_ty,
                                      ZPRRegOp zpr_ty, Operand imm_ty> {
  def : InstAlias<"mov\t$ZAd[$Rv, $imm], $Pg/m, $Zn",
                  (inst tile_ty:$ZAd, MatrixIndexGPR32Op12_15:$Rv, imm0_15:$imm, PPR3bAny:$Pg, zpr_ty:$Zn), 1>;
}

multiclass sme_vector_v_to_tile<string mnemonic, bit is_col> {
  def _B : sme_vector_to_tile_inst<0b00, !if(is_col, TileVectorOpV8,
                                                     TileVectorOpH8),
                                   is_col, imm0_15, ZPR8, mnemonic> {
    bits<4> imm;
    let Inst{3-0} = imm;
  }
  def _H : sme_vector_to_tile_inst<0b01, !if(is_col, TileVectorOpV16,
                                                     TileVectorOpH16),
                                   is_col, imm0_7, ZPR16, mnemonic> {
    bits<1> ZAd;
    bits<3> imm;
    let Inst{3}   = ZAd;
    let Inst{2-0} = imm;
  }
  def _S : sme_vector_to_tile_inst<0b10, !if(is_col, TileVectorOpV32,
                                                     TileVectorOpH32),
                                   is_col, imm0_3, ZPR32, mnemonic> {
    bits<2> ZAd;
    bits<2> imm;
    let Inst{3-2} = ZAd;
    let Inst{1-0} = imm;
  }
  def _D : sme_vector_to_tile_inst<0b11, !if(is_col, TileVectorOpV64,
                                                     TileVectorOpH64),
                                   is_col, imm0_1, ZPR64, mnemonic> {
    bits<3> ZAd;
    bits<1> imm;
    let Inst{3-1} = ZAd;
    let Inst{0}   = imm;
  }
  def _Q : sme_vector_to_tile_inst_Q<!if(is_col, TileVectorOpV128,
                                                 TileVectorOpH128),
                                     is_col, mnemonic> {
    bits<4> ZAd;
    bits<1> imm;
    let Inst{3-0} = ZAd;
  }

  defm : sme_vector_to_tile_aliases<!cast<Instruction>(NAME # _B),
                                    !if(is_col, TileVectorOpV8,
                                                TileVectorOpH8),
                                    ZPR8, imm0_15>;
  defm : sme_vector_to_tile_aliases<!cast<Instruction>(NAME # _H),
                                    !if(is_col, TileVectorOpV16,
                                                TileVectorOpH16),
                                    ZPR16, imm0_7>;
  defm : sme_vector_to_tile_aliases<!cast<Instruction>(NAME # _S),
                                    !if(is_col, TileVectorOpV32,
                                                TileVectorOpH32),
                                    ZPR32, imm0_3>;
  defm : sme_vector_to_tile_aliases<!cast<Instruction>(NAME # _D),
                                    !if(is_col, TileVectorOpV64,
                                                TileVectorOpH64),
                                    ZPR64, imm0_1>;

  def : InstAlias<"mov\t$ZAd[$Rv], $Pg/m, $Zn",
                  (!cast<Instruction>(NAME # _Q) !if(is_col,
                                                      TileVectorOpV128,
                                                      TileVectorOpH128):$ZAd,
                                                 MatrixIndexGPR32Op12_15:$Rv,
                                                 PPR3bAny:$Pg, ZPR128:$Zn), 1>;
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

class sme_tile_to_vector_inst<bits<2> sz, ZPRRegOp zpr_ty,
                              MatrixTileVectorOperand tile_ty,
                              bit is_col, Operand imm_ty, string mnemonic>
    : sme_tile_to_vector_base<0b0, is_col, sz, (outs zpr_ty:$Zd),
        (ins PPR3bAny:$Pg, tile_ty:$ZAn, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm),
        mnemonic, "\t$Zd, $Pg/m, $ZAn[$Rv, $imm]">;

class sme_tile_to_vector_inst_Q<MatrixTileVectorOperand tile_ty,
                                bit is_col, string mnemonic>
    : sme_tile_to_vector_base<0b1, is_col, 0b11, (outs ZPR128:$Zd),
        (ins PPR3bAny:$Pg, tile_ty:$ZAn, MatrixIndexGPR32Op12_15:$Rv),
        mnemonic, "\t$Zd, $Pg/m, $ZAn[$Rv]">;

multiclass sme_tile_to_vector_aliases<Instruction inst, ZPRRegOp zpr_ty,
                                      MatrixTileVectorOperand tile_ty,
                                      Operand imm_ty > {
  def : InstAlias<"mov\t$Zd, $Pg/m, $ZAn[$Rv, $imm]",
                  (inst zpr_ty:$Zd, PPR3bAny:$Pg, tile_ty:$ZAn, MatrixIndexGPR32Op12_15:$Rv, imm_ty:$imm), 1>;
}

multiclass sme_tile_to_vector_v<string mnemonic, bit is_col> {
  def _B : sme_tile_to_vector_inst<0b00, ZPR8, !if(is_col, TileVectorOpV8,
                                                           TileVectorOpH8),
                                   is_col, imm0_15, mnemonic> {
    bits<4> imm;
    let Inst{8-5} = imm;
  }
  def _H : sme_tile_to_vector_inst<0b01, ZPR16, !if(is_col, TileVectorOpV16,
                                                            TileVectorOpH16),
                                   is_col, imm0_7, mnemonic> {
    bits<1> ZAn;
    bits<3> imm;
    let Inst{8}   = ZAn;
    let Inst{7-5} = imm;
  }
  def _S : sme_tile_to_vector_inst<0b10, ZPR32, !if(is_col, TileVectorOpV32,
                                                            TileVectorOpH32),
                                   is_col, imm0_3, mnemonic> {
    bits<2> ZAn;
    bits<2> imm;
    let Inst{8-7} = ZAn;
    let Inst{6-5} = imm;
  }
  def _D : sme_tile_to_vector_inst<0b11, ZPR64, !if(is_col, TileVectorOpV64,
                                                            TileVectorOpH64),
                                   is_col, imm0_1, mnemonic> {
    bits<3> ZAn;
    bits<1> imm;
    let Inst{8-6} = ZAn;
    let Inst{5}   = imm;
  }
  def _Q : sme_tile_to_vector_inst_Q<!if(is_col, TileVectorOpV128,
                                                 TileVectorOpH128),
                                     is_col, mnemonic> {
    bits<4> ZAn;
    let Inst{8-5} = ZAn;
  }

  defm : sme_tile_to_vector_aliases<!cast<Instruction>(NAME # _B), ZPR8,
                                    !if(is_col, TileVectorOpV8,
                                                TileVectorOpH8), imm0_15>;
  defm : sme_tile_to_vector_aliases<!cast<Instruction>(NAME # _H), ZPR16,
                                    !if(is_col, TileVectorOpV16,
                                                TileVectorOpH16), imm0_7>;
  defm : sme_tile_to_vector_aliases<!cast<Instruction>(NAME # _S), ZPR32,
                                    !if(is_col, TileVectorOpV32,
                                                TileVectorOpH32), imm0_3>;
  defm : sme_tile_to_vector_aliases<!cast<Instruction>(NAME # _D), ZPR64,
                                    !if(is_col, TileVectorOpV64,
                                                TileVectorOpH64), imm0_1>;

  def : InstAlias<"mov\t$Zd, $Pg/m, $ZAn[$Rv]",
                  (!cast<Instruction>(NAME # _Q) ZPR128:$Zd, PPR3bAny:$Pg,
                                                 !if(is_col,
                                                      TileVectorOpV128,
                                                      TileVectorOpH128):$ZAn,
                                                 MatrixIndexGPR32Op12_15:$Rv), 1>;
}

multiclass sme_tile_to_vector<string mnemonic> {
  defm _H : sme_tile_to_vector_v<mnemonic, /*is_col=*/0b0>;
  defm _V : sme_tile_to_vector_v<mnemonic, /*is_col=*/0b1>;
}

//===----------------------------------------------------------------------===//
// SME Zero
//===----------------------------------------------------------------------===//

class sme_zero_inst<string mnemonic>
    : I<(outs MatrixTileList:$imm), (ins),
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

class sve2_int_perm_dup_p<string asm, PPRRegOp ppr_ty, Operand imm_ty>
    : I<(outs ppr_ty:$Pd), (ins PPRAny:$Pg, ppr_ty:$Pn,
                                MatrixIndexGPR32Op12_15:$Rm, imm_ty:$imm),
        asm, "\t$Pd, $Pg/z, $Pn[$Rm, $imm]", "", []>,
      Sched<[]> {
  bits<2> Rm;
  bits<4> Pg;
  bits<4> Pn;
  bits<4> Pd;
  let Inst{31-24} = 0b00100101;
  let Inst{21}    = 0b1;
  let Inst{17-16} = Rm;
  let Inst{15-14} = 0b01;
  let Inst{13-10} = Pg;
  let Inst{9}     = 0b0;
  let Inst{8-5}   = Pn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = Pd;
}

multiclass sve2_int_perm_dup_p<string asm> {
  def _B : sve2_int_perm_dup_p<asm, PPR8, imm0_15> {
    bits<4> imm;
    let Inst{23-22} = imm{3-2};
    let Inst{20-19} = imm{1-0};
    let Inst{18}    = 0b1;
  }
  def _H : sve2_int_perm_dup_p<asm, PPR16, imm0_7> {
    bits<3> imm;
    let Inst{23-22} = imm{2-1};
    let Inst{20}    = imm{0};
    let Inst{19-18} = 0b10;
  }
  def _S : sve2_int_perm_dup_p<asm, PPR32, imm0_3> {
    bits<2> imm;
    let Inst{23-22} = imm{1-0};
    let Inst{20-18} = 0b100;
  }
  def _D : sve2_int_perm_dup_p<asm, PPR64, imm0_1> {
    bits<1> imm;
    let Inst{23}    = imm;
    let Inst{22}    = 0b1;
    let Inst{20-18} = 0b000;
  }

  def : InstAlias<"dup\t$Pd, $Pg/z, $Pn[$Rm]",
                  (!cast<Instruction>(NAME # _B) PPR8:$Pd, PPRAny:$Pg, PPR8:$Pn, MatrixIndexGPR32Op12_15:$Rm, 0), 1>;
  def : InstAlias<"dup\t$Pd, $Pg/z, $Pn[$Rm]",
                  (!cast<Instruction>(NAME # _H) PPR16:$Pd, PPRAny:$Pg, PPR16:$Pn, MatrixIndexGPR32Op12_15:$Rm, 0), 1>;
  def : InstAlias<"dup\t$Pd, $Pg/z, $Pn[$Rm]",
                  (!cast<Instruction>(NAME # _S) PPR32:$Pd, PPRAny:$Pg, PPR32:$Pn, MatrixIndexGPR32Op12_15:$Rm, 0), 1>;
  def : InstAlias<"dup\t$Pd, $Pg/z, $Pn[$Rm]",
                  (!cast<Instruction>(NAME # _D) PPR64:$Pd, PPRAny:$Pg, PPR64:$Pn, MatrixIndexGPR32Op12_15:$Rm, 0), 1>;
}
