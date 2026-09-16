//===-- NanoMipsInstrFormats.td - nanoMIPS Instr Formats --*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the formats of the nanoMIPS instruction set.
//
//===----------------------------------------------------------------------===//

class InstNM_Enc : Instruction, PredicateControl, ISA_NANOMIPS {
  let Namespace = "Mips";
  let DecoderNamespace = "NanoMips";

  let EncodingPredicates = [HasNanoMips];
  string Arch = "nanomips";
  let AsmVariantName = "nanomips";
}

class InstNM<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             InstrItinClass itin = NoItinerary,
             bit FastISelIgnore = 1>
  : InstNM_Enc {
  let OutOperandList = outs;
  let InOperandList  = ins;
  let AsmString = asmstr;
  let Pattern   = pattern;
  let FastISelShouldIgnore = FastISelIgnore;
  bit isCTI = 0; // Any form of Control Transfer Instruction.

  let Itinerary = itin;
}

class InstSize16 {
  bits<16> Inst;
  int Size = 2;
  bits<16> SoftFail = 0;
}

class InstSize32 {
  bits<32> Inst;
  int Size = 4;
  bits<32> SoftFail = 0;
}

class InstSize48 {
  bits<48> Inst;
  int Size = 6;
  bits<48> SoftFail = 0;
}

class RegConstraint<string C> {
  string Constraints = C;
}

class NoEncode<string E> {
  string DisableEncoding = E;
}

//===----------------------------------------------------------------------===//
// Opcode Pools, as specified in the ISA document.
//===----------------------------------------------------------------------===//

class _Pool_P32<bits<5> psel> : InstSize32 {
  let Inst{31-26} = {psel{4-2}, 0b0, psel{1-0}};
}

class _Pool_P_ADDIU<bits<5> psel> : _Pool_P32<0b00000> {
  let Inst{25-21} = psel;
}

class _Pool_P32A<bits<3> psel> : _Pool_P32<0b00100> {
  let Inst{2-0} = psel;
}

class _Pool_P_GP_W<bits<2> psel> : _Pool_P32<0b01000> {
  let Inst{1-0} = psel;
}

class _Pool_P_U12<bits<4> psel> : _Pool_P32<0b10000> {
  let Inst{15-12} = psel;
}

class _Pool_P_LUI<bits<1> psel> : _Pool_P32<0b11100> {
  let Inst{1} = psel;
}

class _Pool_P48I<bits<5> psel> : InstSize48 {
  let Inst{47-42} = 0b011000;
  let Inst{36-32} = psel;
}

class _Pool_P_GP_BH<bits<3> psel> : _Pool_P32<0b01001> {
  let Inst{20-18} = psel;
}

class _Pool_P_LS_U12<bits<4> psel> : _Pool_P32<0b10001> {
  let Inst{15-12} = psel;
}

class _Pool_P_LS_S9<bits<3> psel> : _Pool_P32<0b10101> {
  let Inst{10-8} = psel;
}

class _Pool_P_BAL<bits<1> psel> : _Pool_P32<0b00110> {
  let Inst{25} = psel;
}

class _Pool_P_J<bits<4> psel> : _Pool_P32<0b01010> {
  let Inst{15-12} = psel;
}

// Covers both the P.BR1 and P.BR2 pools.
class _Pool_P_BR12<bits<3> psel> : _Pool_P32<0b10010> {
  let Inst{29} = psel{2};
  let Inst{15-14} = psel{1-0};
}

class _Pool_P_BRI<bits<3> psel> : _Pool_P32<0b11010> {
  let Inst{20-18} = psel;
}

class _Pool_P_SHIFT<bits<4> psel> : _Pool_P_U12<0b1100> {
  bits<5> shift;
  bits<5> rs;
  bits<5> rt;

  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{8-5} = psel;
  let Inst{4-0} = shift;
}

class _Pool_P_EXT_INS<bits<4> psel> : _Pool_P_U12<psel> {
  let Inst{11} = 0;
  let Inst{5} = 0;
}

class _Pool_P_LS_E0<bits<4> psel> : _Pool_P_LS_S9<0b010> {
  let Inst{14-11} = psel;
}

class _Pool_P16<bits<5> psel> : InstSize16 {
  let Inst{15-10} = {psel{4-2}, 0b1, psel{1-0}};
}

class _Pool_P16_MV : _Pool_P16<0b00000>;

class _Pool_P16_SR<bits<1> psel> : _Pool_P16<0b00011> {
  let Inst{8} = psel;
}

class _Pool_P16_SHIFT<bits<1> isel> : _Pool_P16<0b00100> {
  let Inst{3} = isel;
}

class _Pool_P16_4X4<bits<2> psel> : _Pool_P16<0b00111> {
  let Inst{8} = psel{1};
  let Inst{3} = psel{0};
}

class _Pool_P16C<bits<1> psel> : _Pool_P16<0b01000> {
  let Inst{0} = psel;
}

class _Pool_POOL16C_00<bits<2> psel> : _Pool_P16<0b01000> {
  let Inst{3-2} = psel;
  let Inst{1-0} = 0b00;
}

class _Pool_P16_LB<bits<2> psel> : _Pool_P16<0b01011> {
  let Inst{3-2} = psel;
}

class _Pool_P16_A1<bits<1> psel> : _Pool_P16<0b01100> {
  let Inst{6} = psel;
}

class _Pool_P16_LH<bits<2> psel> : _Pool_P16<0b01111> {
  let Inst{3} = psel{1};
  let Inst{0} = psel{0};
}

class _Pool_P16_A2<bits<1> psel> : _Pool_P16<0b10000> {
  let Inst{3} = psel;
}

class _Pool_P16_ADDU<bits<1> psel> : _Pool_P16<0b10100> {
  let Inst{0} = psel;
}

class _Pool_P16_BR<bits<4> psel> : _Pool_P16<0b11010> {
  let Inst{3-0} = psel;
}

class _Pool_P16_RI<bits<2> psel> : _Pool_P16_MV {
  let Inst{4-3} = psel;
  let Inst{9-5} = 0b00000;
}

class _Pool_P_RI<bits<2> psel> : _Pool_P_ADDIU<0b00000> {
  let Inst{20-19} = psel;
}

class _Pool_POOL32A0<bits<1> psel> : _Pool_P32A<0b000> {
  let Inst{5} = psel;
}

class _Pool_POOL32A0_0<bits<6> psel> : _Pool_POOL32A0<0b0> {
  let Inst{9-6} = psel{5-2};
  let Inst{4-3} = psel{1-0};
}

class _Pool_P_TRAP<bits<1> isel> : _Pool_POOL32A0_0<0b000000> {
  let Inst{10} = isel;
}

class _Pool_P_CMOVE<bits<1> isel> : _Pool_POOL32A0_0<0b100010> {
  let Inst{10} = isel;
}

class _Pool_POOL32A7<bits<3> psel> : _Pool_P32A<0b111> {
  let Inst{5-3} = psel;
}

class _Pool_POOL32Axf_4<bits<7> isel> : _Pool_POOL32A7<0b111> {
  let Inst{15-9} = isel;
  let Inst{8-6} = 0b100;
}

class _Pool_POOL32Axf_5<bits<2> psel> : _Pool_POOL32A7<0b111> {
  let Inst{8-6} = 0b101;
  let Inst{15-14} = psel;
}

class _Pool_POOL32Axf_5_group0<bits<5> isel> : _Pool_POOL32Axf_5<0b00> {
  let Inst{13-9} = isel;
}

class _Pool_POOL32Axf_5_group1<bits<5> isel> : _Pool_POOL32Axf_5<0b01> {
  let Inst{13-9} = isel;
}

class _Pool_POOL32Axf_5_group3<bits<5> isel> : _Pool_POOL32Axf_5<0b11> {
  let Inst{13-9} = isel;
}

class _Pool_PP_LSX<bit scaled, bits<4> isel> : _Pool_POOL32A7<0b000> {
  let Inst{6} = scaled;
  let Inst{10-7} = isel;
}

class _Pool_POOL32A0_1<bits<6> psel> : _Pool_POOL32A0<0b1> {
  let Inst{9-6} = psel{5-2};
  let Inst{4-3} = psel{1-0};
}

class _Pool_PP_SR<bits<2> isel> : _Pool_P_U12<0b0011> {
  let Inst{20} = 0;
  let Inst{1-0} = isel;
}

//===----------------------------------------------------------------------===//
// nanoMIPS Instruction Formats
//===----------------------------------------------------------------------===//

class Arith4x4_Enc<bits<2> isel> :  _Pool_P16_4X4<isel> {
  bits<5> dst;
  bits<5> rs;
  bits<5> rt;

  let Inst{9} = dst{4};
  let Inst{7-5} = dst{2-0};
  let Inst{4} = rs{4};
  let Inst{2-0} = rs{2-0};
}

class ArithLogicR32_Enc<bits<6> psel> : _Pool_POOL32A0_0<psel> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
}

class ArithLogicR16_Enc<bits<2> isel> : _Pool_POOL16C_00<isel> {
  bits<3> rs;
  bits<3> rd;
  let Inst{9-7} = rd;
  let Inst{6-4} = rs;
}

class CLZO_Enc<bits<7> isel> : _Pool_POOL32Axf_4<isel> {
  bits<5> rs;
  bits<5> rt;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
}

class LogicU32_Enc<bits<6> isel> : _Pool_POOL32A0_0<isel> {
  bits<5> rs;
  bits<5> rt;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
}

class IndirectBranch32_Enc : _Pool_P_J<0b1000> {
  bits<5> rs;
  let Inst{25-21} = 0b00000;
  let Inst{20-16} = rs;
}

class IndirectBranch16_Enc<bits<1> isel> : _Pool_P16_BR<0b0000> {
  bits<5> rs;
  let Inst{9-5} = rs;
  let Inst{4} = isel;
}

class IndirectCall32_Enc<bits<4> isel> : _Pool_P_J<isel> {
  bits<5> rt;
  bits<5> rs;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
}

class IndirectCall16_Enc : _Pool_P16_BR<0b0000> {
  bits<5> rt;
  bits<5> rs;
  let Inst{9-5} = rs;
  let Inst{4} = rt{0}; // always 1, forced for disassembly of JALRC[16]
}

class LI16_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<3> rt;
  bits<7> eu;
  let Inst{9-7} = rt;
  let Inst{6-0} = eu;
}

class BC16_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<11> addr;
  let Inst{9-1} = addr{9-1};
  let Inst{0} = addr{10};
}

class BC32_Enc<bits<1> isel> : _Pool_P_BAL<isel> {
  bits<26> addr;
  let Inst{24-1} = addr{24-1};
  let Inst{0} = addr{25};
}

class AddSubR16_Enc<bits<1> isel> : _Pool_P16_ADDU<isel> {
  bits<3> rd;
  bits<3> rs;
  bits<3> rt;
  let Inst{15-10} = 0b101100;
  let Inst{9-7} = rt;
  let Inst{6-4} = rs;
  let Inst{3-1} = rd;
}

class ANDI16_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<3> rt;
  bits<3> rs;
  bits<4> imm;
  let Inst{9-7} = rt;
  let Inst{6-4} = rs;
  let Inst{3-0} = imm;
}

class SaveRestore32_Enc<bits<2> isel> : _Pool_PP_SR<isel> {
  bits<12> adj;
  bits<10> regs;
  let Inst{11-3} = adj{11-3};
  let Inst{25-21} = regs{9-5};
  let Inst{19-16} = regs{4-1};
  let Inst{2} = regs{0};
}

class SaveRestore16_Enc<bits<1> isel> : _Pool_P16_SR<isel> {
  bits<8> adj;
  bits<5> regs;
  let Inst{7-4} = adj{7-4};
  let Inst{9} = regs{4};
  let Inst{3-0} = regs{3-0};
}

class CondLogicI32_Enc<bits<4> psel> : _Pool_P_U12<psel> {
  bits<5> rt;
  bits<5> rs;
  bits<12> imm;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{11-0} = imm;
}

class ShiftI16_Enc<bits<1> isel> : _Pool_P16_SHIFT<isel> {
  bits<3> rt;
  bits<3> rs;
  bits<3> imm;
  let Inst{9-7} = rt;
  let Inst{6-4} = rs;
  let Inst{2-0} = imm;
}

class ShiftI32_Enc<bits<4> psel> : _Pool_P_SHIFT<psel> {
  bits<5> rt;
  bits<5> rs;
  bits<5> imm;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{4-0} = imm;
}

class ExtInsBase_Enc<bits<4> isel> : _Pool_P_EXT_INS<isel> {
  bits<5> rt;
  bits<5> rs;
  bits<5> pos;
  bits<5> size;

  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{4-0} = pos;
  let Inst{10-6} = size;
}

class AddIU_Enc : _Pool_P32<0b00000>  {
  bits<5> rt;
  bits<5> rs;
  bits<16> imm;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-0} = imm;
}

class ADDIUR2_Enc : _Pool_P16_A2<0b0> {
  bits<3> rt;
  bits<3> rs;
  bits<5> imm;
  let Inst{9-7} = rt;
  let Inst{6-4} = rs;
  let Inst{2-0} = imm{4-2};
}

class ADDIURS5_Enc : _Pool_P16_A2<0b1>  {
  bits<5> rt;
  bits<4> imm;
  let Inst{9-5} = rt;
  let Inst{4} = imm{3};
  let Inst{2-0} = imm{2-0};
}

class ADDIUR1SP_Enc : _Pool_P16_A1<0b1> {
  bits<3> rt;
  bits<8> imm;
  let Inst{9-7} = rt;
  let Inst{5-0} = imm{7-2};
}

class ADDIUGPB_Enc : _Pool_P_GP_BH<0b011> {
  bits<5> rt;
  bits<18> offset;
  let Inst{25-21} = rt;
  let Inst{17-0} = offset;
 }

class ADDIUGPW_Enc : _Pool_P_GP_W<0b00> {
  bits<5> rt;
  bits<21> offset;
  let Inst{25-21} = rt;
  let Inst{20-2} = offset{20-2};
}

class Op48_Imm_Enc<bits<5> isel> : _Pool_P48I<isel> {
  bits<5> rt;
  bits<32> imm;
  let Inst{41-37} = rt;
  let Inst{31-16} = imm{15-0};
  let Inst{15-0} = imm{31-16};
}

class Op32_Addr_Enc<bits<5> isel> : _Pool_P32<isel> {
  bits<5> rt;
  bits<22> addr;
  let Inst{25-21} = rt;
  let Inst{20-1} = addr{20-1};
  let Inst{0} = addr{21};
}

class Op48_Addr_Enc<bits<5> isel> : _Pool_P48I<isel> {
  bits<5> rt;
  bits<32> addr;
  let Inst{41-37} = rt;
  let Inst{31-16} = addr{15-0};
  let Inst{15-0} = addr{31-16};
}

class LoadScaledAddress_Enc: _Pool_POOL32A7<0b001> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<2> shift;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10-9} = shift;
}

class Trap_Enc<bits<1> isel> : _Pool_P_TRAP<isel> {
  bits<5> rt;
  bits<5> rs;
  bits<5> imm;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = imm;
}

class LUI_Enc<bits<1> isel> : _Pool_P_LUI<isel> {
  bits<5> rt;
  bits<20> imm;
  let Inst{25-21} = rt;
  let Inst{20-12} = imm{8-0};
  let Inst{11-2} = imm{18-9};
  let Inst{0} = imm{19};
}

class RDHWR_Enc : _Pool_POOL32A0_0<0b011100> {
  bits<5> rt;
  bits<5> hs;
  bits<5> sel;
  let Inst{25-21} = rt;
  let Inst{20-16} = hs;
  let Inst{15-11} = sel;
}

class B14_Enc<bits<3> isel> : _Pool_P_BR12<isel> {
  bits<5> rt;
  bits<5> rs;
  bits<15> offset;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{13-1} = offset{13-1};
  let Inst{0} =  offset{14};
}

class B4_Enc : _Pool_P16<0b11010> {
  bits<3> rs;
  bits<3> rt;
  bits<4> offset;
  let Inst{9-7} = rt;
  let Inst{6-4} = rs;
  let Inst{3-0} = offset;
}

class BZ16_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<3> rs;
  bits<8> offset;
  let Inst{9-7} = rs;
  let Inst{6-1} = offset{6-1};
  let Inst{0} =  offset{7};
}

class BRI_Enc<bits<3> isel> : _Pool_P_BRI<isel> {
  bits<5> rt;
  bits<7> u;
  bits<12> offset;
  let Inst{25-21} = rt;
  let Inst{17-11} = u;
  let Inst{10-1} = offset{10-1};
  let Inst{0} =  offset{11};
}

class BBit_Enc<bits<3> isel> : _Pool_P_BRI<isel> {
  bits<5> rt;
  bits<6> u;
  bits<12> offset;
  let Inst{25-21} = rt;
  let Inst{16-11} = u;
  let Inst{10-1} = offset{10-1};
  let Inst{0} =  offset{11};
}

class LSSPRegMem16_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<5> rt;
  bits<7> addr;
  bits<5> offset = addr{6-2};
  let Inst{9-5} = rt;
  let Inst{4-0} = offset;
}

class LSGPRegMem16_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<3> rt;
  bits<9> addr;
  bits<7> offset = addr{8-2};
  let Inst{9-7} = rt;
  let Inst{6-0} = offset;
}

class LSMem16_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<3> rt;
  bits<9> addr;
  bits<3> base = addr{8-6};
  bits<4> offset = addr{5-2};
  let Inst{9-7} = rt;
  let Inst{6-4} = base;
  let Inst{3-0} = offset;
}

class LHMem16_Enc<bits<2> isel> : _Pool_P16_LH<isel> {
  bits<3> rt;
  bits<6> addr;
  bits<3> base = addr{5-3};
  bits<3> offset = addr{2-0};
  let Inst{9-7} = rt;
  let Inst{6-4} = base;
  let Inst{2-1} = offset{2-1};
}

class LBMem16_Enc<bits<2> isel> : _Pool_P16_LB<isel> {
  bits<3> rt;
  bits<5> addr;
  bits<3> base = addr{4-2};
  bits<2> offset = addr{1-0};
  let Inst{9-7} = rt;
  let Inst{6-4} = base;
  let Inst{1-0} = offset{1-0};
}

class LSMemGP19S2_Enc<bits<2> isel> : _Pool_P_GP_W<isel> {
  bits<5> rt;
  bits<21> addr;
  let Inst{25-21} = rt;
  let Inst{20-2} = addr{20-2};
}

class LSMemGP18_Enc<bits<3> isel> : _Pool_P_GP_BH<isel> {
  bits<5> rt;
  bits<18> addr;
  let Inst{25-21} = rt;
  let Inst{17-0} = addr;
}

class LSMemGP17S1_Enc<bits<4> isel> : _Pool_P_GP_BH<isel{3-1}> {
  bits<5> rt;
  bits<18> addr;
  let Inst{25-21} = rt;
  let Inst{17-1} = addr{17-1};
  let Inst{0} = isel{0};
}

class LSMem_U12_Enc<bits<4> isel> : _Pool_P_LS_U12<isel> {
  bits<5> rt;
  bits<17> addr;
  bits<5> base = addr{16-12};
  bits<12> offset = addr{11-0};
  let Inst{25-21} = rt;
  let Inst{20-16} = base;
  let Inst{11-0} = offset;
}

class LSMemX_Enc<bit scaled, bits<4> isel> : _Pool_PP_LSX<scaled, isel> {
  bits<5> rt;
  bits<10> addr;
  let Inst{25-21} = addr{9-5};
  let Inst{20-16} = addr{4-0};
  let Inst{15-11} = rt;
}

class LSMemXS16_Enc : _Pool_P16C<0b1> {
  bits<3> rt;
  // Uses same encoding method as LWXS to get 5+5 register bits
  // but only needs the lower 3 bits of each register.
  bits<10> addr;
  let Inst{9-7} = addr{7-5}; // Discard {9-8}
  let Inst{6-4} = addr{2-0}; // Discard {4-3}
  let Inst{3-1} = rt;
}

class LSMem_S9_Enc<bits<3> psel, bits<4> isel> : _Pool_P_LS_S9<psel> {
  bits<5> rt;
  bits<14> addr;
  bits<5> base = addr{13-9};
  bits<9> offset = addr{8-0};
  let Inst{25-21} = rt;
  let Inst{20-16} = base;
  let Inst{14-11} = isel;
  let Inst{15} = offset{8};
  let Inst{7-0} = offset{7-0};
}

class LSMem_S9_Mult_Enc<bits<4> isel> : _Pool_P_LS_S9<isel{2-0}> {
  bits<5> rt;
  bits<14> addr;
  bits<5> base = addr{13-9};
  bits<9> offset = addr{8-0};
  bits<3> rcount;
  let Inst{25-21} = rt;
  let Inst{20-16} = base;
  let Inst{11} = isel{3};
  let Inst{15} = offset{8};
  let Inst{7-0} = offset{7-0};
  let Inst{14-12} = rcount;
}

class LSMem4x4_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<5> rt;
  bits<9> addr;
  bits<5> rs = addr{8-4};
  bits<4> offset = addr{3-0};
  let Inst{9} = rt{4};
  let Inst{7-5} = rt{2-0};
  let Inst{4} = rs{4};
  let Inst{2-0} = rs{2-0};
  let Inst{3} = offset{3};
  let Inst{8} = offset{2};
}

class MOVE_Enc : _Pool_P16_MV {
  bits<5> rt;
  bits<5> rs;
  let Inst{9-5} = rt;
  let Inst{4-0} = rs;
}

class MOVEP_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<2> dst1;
  bits<5> src1;
  bits<5> src2;

  let Inst{9} = src2{4};
  let Inst{7-5} = src2{2-0};
  let Inst{4} = src1{4};
  let Inst{2-0} = src1{2-0};
  let Inst{8} = dst1{0};
  let Inst{3} = dst1{1};
}

class MOVEPREV_Enc<bits<5> isel> : _Pool_P16<isel> {
  bits<2> src1;
  bits<5> dst1;
  bits<5> dst2;

  let Inst{9} = dst2{4};
  let Inst{7-5} = dst2{2-0};
  let Inst{4} = dst1{4};
  let Inst{2-0} = dst1{2-0};
  let Inst{8} = src1{0};
  let Inst{3} = src1{1};
}

class MOVEBALC_Enc : _Pool_P32<0b00010> {
  bits<5> rt;
  bits<1> rd;
  bits<22> addr;
  let Inst{25} = rt{4};
  let Inst{24} = rd;
  let Inst{23-21} = rt{2-0};
  let Inst{20-1} = addr{20-1};
  let Inst{0} = addr{21};
}

class LLSC_Enc<bits<3> psel, bits<4>isel> : _Pool_P_LS_S9<psel> {
  bits<5> rt;
  bits<14> addr;
  bits<5> base = addr{13-9};
  bits<9> offset = addr{8-0};
  let Inst{25-21} = rt;
  let Inst{20-16} = base;
  let Inst{15} = offset{8};
  let Inst{14-11} = isel;
  let Inst{7-2} = offset{7-2};
  let Inst{1-0} = 0b00;
}

class LLSCWP_Enc<bits<3> psel, bits<4> isel> : _Pool_P_LS_S9<psel> {
  bits<5> rt;
  bits<5> ru;
  bits<5> addr;
  let Inst{25-21} = rt;
  let Inst{20-16} = addr;
  let Inst{14-11} = isel;
  let Inst{7-3} = ru;
  let Inst{1-0} = 0b01;
}

class CondMove_Enc<bits<1> isel> : _Pool_P_CMOVE<isel> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
}

class ToggleInterrupts_Enc<bits<5> isel> : _Pool_POOL32Axf_5_group1<isel> {
  bits<5> rt;
  let Inst{25-21} = rt;
}

class ERET_Enc<bits<5> psel, bits<1> isel> : _Pool_POOL32Axf_5_group3<psel> {
  let Inst{16} = isel;
}

class SLL_VAR_Enc<bits<5> sh> : _Pool_P_SHIFT<0b0000> {
  let Inst{25-21} = 0b00000;
  let Inst{20-16} = 0b00000;
  let Inst{4-0} = sh;
}

class NOP_Enc : _Pool_P16_A2<0b1> {
  let Inst{9-5} = 0b00000;
  let Inst{4} = 0b0;
  let Inst{2-0} = 0b000;
}

class IntExcpFlow_Enc<bits<2> isel, int immsize = 19> : _Pool_P_RI<isel> {
  bits<19> imm;
  let Inst{18} = !if(!eq(immsize, 19), imm{18}, 0b0);
  let Inst{17-0} = imm{17-0};
}

class IntExcpFlow16_Enc<bits<2> isel, int immsize = 3> : _Pool_P16_RI<isel> {
  bits<3> imm;
  let Inst{2} = !if(!eq(immsize, 3), imm{2}, 0b0);
  let Inst{1-0} = imm{1-0};
}

class ShadowRW_Enc<bits<5> isel> : _Pool_POOL32Axf_5_group3<isel> {
  bits<5> rs;
  bits<5> rt;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
}

class EXTW_Enc : _Pool_POOL32A7<0b011> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<5> shift;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10-6} = shift;
}

class RotxAliasBase_Enc : _Pool_P_U12<0b1101> {
  bits<5> rt;
  bits<5> rs;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{11} = 0;
  let Inst{5} = 0;
}

class RotxAlias_Enc<bits<5> shift, bits<5> shiftx, bit stripe>
  : RotxAliasBase_Enc {
  let Inst{10-7} = shiftx{4-1};
  let Inst{4-0} = shift;
  let Inst{6} = stripe;
}

class ROTX_Enc : RotxAliasBase_Enc {
  bits<5> shift;
  bits<5> shiftx;
  bits<1> stripe;
  let Inst{10-7} = shiftx{4-1};
  let Inst{4-0} = shift;
  let Inst{6} = stripe;
}

class CACHE_PREFs9_Enc<bits<3> psel, bits<4> isel> : _Pool_P_LS_S9<psel> {
  bits<5> op;
  bits<14> addr;
  bits<5> base = addr{13-9};
  bits<9> offset = addr{8-0};
  let Inst{25-21} = op;
  let Inst{20-16} = base;
  let Inst{15} = offset{8};
  let Inst{14-11} = isel;
  let Inst{7-0} = offset{7-0};
}

class PREF_Enc : _Pool_P_LS_U12<0b0011> {
  bits<5> op;
  bits<17> addr;
  bits<5> base = addr{16-12};
  bits<12> offset = addr{11-0};
  let Inst{25-21} = op;
  let Inst{20-16} = base;
  let Inst{11-0} = offset{11-0};
}

class SYNC_Enc : _Pool_P_SHIFT<0b0000> {
  bits<5> stype;

  let Inst{25-21} = 0b00000;
  let Inst{20-16} = stype;
  let Inst{4-0} = 0b00110;
}

class SYNCIs9_SYNCIE_Enc<bits<3> psel, bits<4> isel> : _Pool_P_LS_S9<psel> {
  bits<14> addr;
  bits<5> base = addr{13-9};
  bits<9> offset = addr{8-0};
  let Inst{25-21} = 0b11111;
  let Inst{20-16} = base;
  let Inst{15} = offset{8};
  let Inst{14-11} = isel;
  let Inst{7-0} = offset{7-0};
}

class SYNCI_Enc : _Pool_P_LS_U12<0b0011> {
  bits<17> addr;
  bits<5> base = addr{16-12};
  bits<12> offset = addr{11-0};
  let Inst{25-21} = 0b11111;
  let Inst{20-16} = base;
  let Inst{11-0} = offset;
}

class WAIT_Enc : _Pool_POOL32Axf_5_group3<0b00001> {
  bits<10> cd;
  let Inst{25-16} = cd;
}

class ReadWriteCP0_Enc<bits<6> isel> : _Pool_POOL32A0_1<isel> {
  bits<5> rt;
  bits<5> c0s;
  bits<5> sel;

  let Inst{25-21} = rt;
  let Inst{20-16} = c0s;
  let Inst{15-11} = sel;
}

class ReadWriteCP0Sel_Enc<bits<6> isel> : _Pool_POOL32A0_1<isel> {
  bits<5> rt;
  bits<10> c0s;

  let Inst{25-21} = rt;
  let Inst{20-11} = c0s;
}

class GINVI_Enc<bits<5> isel> : _Pool_POOL32Axf_5_group0<isel> {
  bits<5> rs;

  let Inst{20-16} = rs;
}

class GINVT_Enc<bits<5> isel> : GINVI_Enc<isel> {
  bits<2> type;

  let Inst{22-21} = type;
}

class CRC_Enc<bits<3> isel> : _Pool_POOL32A0_1<0b111101> {
  bits<5> rt;
  bits<5> rs;

  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{12-10} = isel;
}

class VP_Enc<bits<1> p> : _Pool_POOL32A0_0<0b111010> {
  bits<5> rt;

  let Inst{25-21} = rt;
  let Inst{20-16} = 0b00000;
  let Inst{15-11} = 0b00000;
  let Inst{10} = p;
}
