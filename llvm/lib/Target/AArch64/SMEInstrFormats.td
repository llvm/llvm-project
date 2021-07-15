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
