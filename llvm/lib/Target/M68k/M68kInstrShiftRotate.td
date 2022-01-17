//===-- M68kInstrShiftRotate.td - Logical Instrs -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the logical instructions in the M68k architecture.
/// Here is the current status of the file:
///
///  Machine:
///
///    SHL     [~]   ASR     [~]   LSR      [~]   SWAP     [ ]
///    ROL     [~]   ROR     [~]   ROXL     [ ]   ROXR     [ ]
///
///  Map:
///
///   [ ] - was not touched at all
///   [!] - requires extarnal stuff implemented
///   [~] - in progress but usable
///   [x] - done
///
//===----------------------------------------------------------------------===//

def MxRODI_R : MxBead1Bit<0>;
def MxRODI_L : MxBead1Bit<1>;

def MxROOP_AS  : MxBead2Bits<0b00>;
def MxROOP_LS  : MxBead2Bits<0b01>;
def MxROOP_ROX : MxBead2Bits<0b10>;
def MxROOP_RO  : MxBead2Bits<0b11>;

/// ------------+---------+---+------+---+------+---------
///  F  E  D  C | B  A  9 | 8 | 7  6 | 5 | 4  3 | 2  1  0
/// ------------+---------+---+------+---+------+---------
///  1  1  1  0 | REG/IMM | D | SIZE |R/I|  OP  |   REG
/// ------------+---------+---+------+---+------+---------
class MxSREncoding_R<MxBead1Bit DIRECTION, MxBead2Bits ROOP, MxEncSize SIZE>
    : MxEncoding<MxBeadDReg<0>, ROOP, MxBead1Bit<1>, SIZE, DIRECTION,
                 MxBeadDReg<2>, MxBead4Bits<0b1110>>;

class MxSREncoding_I<MxBead1Bit DIRECTION, MxBead2Bits ROOP, MxEncSize SIZE>
    : MxEncoding<MxBeadDReg<0>, ROOP, MxBead1Bit<0>, SIZE, DIRECTION,
                 MxBead3Imm<2, 1>, MxBead4Bits<0b1110>>;

// $reg <- $reg op $reg
class MxSR_DD<string MN, MxType TYPE, SDNode NODE,
              MxBead1Bit RODI, MxBead2Bits ROOP>
    : MxInst<(outs TYPE.ROp:$dst), (ins TYPE.ROp:$src, TYPE.ROp:$opd),
             MN#"."#TYPE.Prefix#"\t$opd, $dst",
             [(set TYPE.VT:$dst, (NODE TYPE.VT:$src, TYPE.VT:$opd))],
             MxSREncoding_R<RODI, ROOP,
                            !cast<MxEncSize>("MxEncSize"#TYPE.Size)>>;

// $reg <- $reg op $imm
class MxSR_DI<string MN, MxType TYPE, SDNode NODE,
              MxBead1Bit RODI, MxBead2Bits ROOP>
    : MxInst<(outs TYPE.ROp:$dst),
             (ins TYPE.ROp:$src, !cast<Operand>("Mxi"#TYPE.Size#"imm"):$opd),
             MN#"."#TYPE.Prefix#"\t$opd, $dst",
             [(set TYPE.VT:$dst,
                   (NODE TYPE.VT:$src,
                         !cast<ImmLeaf>("Mximm"#TYPE.Size#"_1to8"):$opd))],
             MxSREncoding_I<RODI, ROOP,
                            !cast<MxEncSize>("MxEncSize"#TYPE.Size)>>;

multiclass MxSROp<string MN, SDNode NODE, MxBead1Bit RODI, MxBead2Bits ROOP> {

  let Defs = [CCR] in {
  let Constraints = "$src = $dst" in {

  def NAME#"8dd"  : MxSR_DD<MN, MxType8d,  NODE, RODI, ROOP>;
  def NAME#"16dd" : MxSR_DD<MN, MxType16d, NODE, RODI, ROOP>;
  def NAME#"32dd" : MxSR_DD<MN, MxType32d, NODE, RODI, ROOP>;

  def NAME#"8di"  : MxSR_DI<MN, MxType8d,  NODE, RODI, ROOP>;
  def NAME#"16di" : MxSR_DI<MN, MxType16d, NODE, RODI, ROOP>;
  def NAME#"32di" : MxSR_DI<MN, MxType32d, NODE, RODI, ROOP>;

  } // $src = $dst
  } // Defs = [CCR]

} // MxBiArOp_RF

defm SHL : MxSROp<"lsl", shl, MxRODI_L, MxROOP_LS>;
defm LSR : MxSROp<"lsr", srl, MxRODI_R, MxROOP_LS>;
defm ASR : MxSROp<"asr", sra, MxRODI_R, MxROOP_AS>;

defm ROL : MxSROp<"rol", rotl, MxRODI_L, MxROOP_RO>;
defm ROR : MxSROp<"ror", rotr, MxRODI_R, MxROOP_RO>;
