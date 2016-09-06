//===-- VIInstrFormats.td - VI Instruction Encodings ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// VI Instruction format definitions.
//
//===----------------------------------------------------------------------===//

class MUBUFe_vi <bits<7> op> : Enc64 {
  bits<12> offset;
  bits<1> offen;
  bits<1> idxen;
  bits<1> glc;
  bits<1> lds;
  bits<8> vaddr;
  bits<8> vdata;
  bits<7> srsrc;
  bits<1> slc;
  bits<1> tfe;
  bits<8> soffset;

  let Inst{11-0} = offset;
  let Inst{12} = offen;
  let Inst{13} = idxen;
  let Inst{14} = glc;
  let Inst{16} = lds;
  let Inst{17} = slc;
  let Inst{24-18} = op;
  let Inst{31-26} = 0x38; //encoding
  let Inst{39-32} = vaddr;
  let Inst{47-40} = vdata;
  let Inst{52-48} = srsrc{6-2};
  let Inst{55} = tfe;
  let Inst{63-56} = soffset;
}

class MTBUFe_vi <bits<4> op> : Enc64 {
  bits<12> offset;
  bits<1>  offen;
  bits<1>  idxen;
  bits<1>  glc;
  bits<4>  dfmt;
  bits<3>  nfmt;
  bits<8>  vaddr;
  bits<8>  vdata;
  bits<7>  srsrc;
  bits<1>  slc;
  bits<1>  tfe;
  bits<8>  soffset;

  let Inst{11-0}  = offset;
  let Inst{12}    = offen;
  let Inst{13}    = idxen;
  let Inst{14}    = glc;
  let Inst{18-15} = op;
  let Inst{22-19} = dfmt;
  let Inst{25-23} = nfmt;
  let Inst{31-26} = 0x3a; //encoding
  let Inst{39-32} = vaddr;
  let Inst{47-40} = vdata;
  let Inst{52-48} = srsrc{6-2};
  let Inst{54}    = slc;
  let Inst{55}    = tfe;
  let Inst{63-56} = soffset;
}

class SMEMe_vi <bits<8> op, bit imm> : Enc64 {
  bits<7>  sbase;
  bits<7>  sdst;
  bits<1>  glc;

  let Inst{5-0}   = sbase{6-1};
  let Inst{12-6}  = sdst;
  let Inst{16}    = glc;
  let Inst{17}    = imm;
  let Inst{25-18} = op;
  let Inst{31-26} = 0x30; //encoding
}

class SMEM_IMMe_vi <bits<8> op> : SMEMe_vi<op, 1> {
  bits<20> offset;
  let Inst{51-32} = offset;
}

class SMEM_SOFFe_vi <bits<8> op> : SMEMe_vi<op, 0> {
  bits<20> soff;
  let Inst{51-32} = soff;
}

class VOP3a_vi <bits<10> op> : Enc64 {
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;
  bits<2> omod;

  let Inst{8}     = src0_modifiers{1};
  let Inst{9}     = src1_modifiers{1};
  let Inst{10}    = src2_modifiers{1};
  let Inst{15}    = clamp;
  let Inst{25-16} = op;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = src0;
  let Inst{49-41} = src1;
  let Inst{58-50} = src2;
  let Inst{60-59} = omod;
  let Inst{61} = src0_modifiers{0};
  let Inst{62} = src1_modifiers{0};
  let Inst{63} = src2_modifiers{0};
}

class VOP3e_vi <bits<10> op> : VOP3a_vi <op> {
  bits<8> vdst;

  let Inst{7-0} = vdst;
}

// Encoding used for VOPC instructions encoded as VOP3
// Differs from VOP3e by destination name (sdst) as VOPC doesn't have vector dst
class VOP3ce_vi <bits<10> op> : VOP3a_vi <op> {
  bits<8> sdst;

  let Inst{7-0} = sdst;
}

class VOP3be_vi <bits<10> op> : Enc64 {
  bits<8> vdst;
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<7> sdst;
  bits<2> omod;
  bits<1> clamp;

  let Inst{7-0} = vdst;
  let Inst{14-8} = sdst;
  let Inst{15} = clamp;
  let Inst{25-16} = op;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = src0;
  let Inst{49-41} = src1;
  let Inst{58-50} = src2;
  let Inst{60-59} = omod;
  let Inst{61} = src0_modifiers{0};
  let Inst{62} = src1_modifiers{0};
  let Inst{63} = src2_modifiers{0};
}

class VOP_DPP <dag outs, dag ins, string asm, list<dag> pattern, bit HasMods = 0> :
    VOPAnyCommon <outs, ins, asm, pattern> {
  let DPP = 1;
  let Size = 8;

  let AsmMatchConverter = !if(!eq(HasMods,1), "cvtDPP", "");
}

class VOP_DPPe : Enc64 {
  bits<2> src0_modifiers;
  bits<8> src0;
  bits<2> src1_modifiers;
  bits<9> dpp_ctrl;
  bits<1> bound_ctrl;
  bits<4> bank_mask;
  bits<4> row_mask;

  let Inst{39-32} = src0;
  let Inst{48-40} = dpp_ctrl;
  let Inst{51}    = bound_ctrl;
  let Inst{52}    = src0_modifiers{0}; // src0_neg
  let Inst{53}    = src0_modifiers{1}; // src0_abs
  let Inst{54}    = src1_modifiers{0}; // src1_neg
  let Inst{55}    = src1_modifiers{1}; // src1_abs
  let Inst{59-56} = bank_mask;
  let Inst{63-60} = row_mask;
}

class VOP1_DPPe <bits<8> op> : VOP_DPPe {
  bits<8> vdst;

  let Inst{8-0} = 0xfa; // dpp
  let Inst{16-9} = op;
  let Inst{24-17} = vdst;
  let Inst{31-25} = 0x3f; //encoding
}

class VOP2_DPPe <bits<6> op> : VOP_DPPe {
  bits<8> vdst;
  bits<8> src1;

  let Inst{8-0} = 0xfa; //dpp
  let Inst{16-9} = src1;
  let Inst{24-17} = vdst;
  let Inst{30-25} = op;
  let Inst{31} = 0x0; //encoding
}

class VOP_SDWA <dag outs, dag ins, string asm, list<dag> pattern, bit HasMods = 0> :
    VOPAnyCommon <outs, ins, asm, pattern> {
  let SDWA = 1;
  let Size = 8;
}

class VOP_SDWAe : Enc64 {
  bits<8> src0;
  bits<3> src0_sel;
  bits<2> src0_fmodifiers; // {abs,neg}
  bits<1> src0_imodifiers; // sext
  bits<3> src1_sel;
  bits<2> src1_fmodifiers;
  bits<1> src1_imodifiers;
  bits<3> dst_sel;
  bits<2> dst_unused;
  bits<1> clamp;

  let Inst{39-32} = src0;
  let Inst{42-40} = dst_sel;
  let Inst{44-43} = dst_unused;
  let Inst{45} = clamp;
  let Inst{50-48} = src0_sel;
  let Inst{53-52} = src0_fmodifiers;
  let Inst{51} = src0_imodifiers;
  let Inst{58-56} = src1_sel;
  let Inst{61-60} = src1_fmodifiers;
  let Inst{59} = src1_imodifiers;
}

class VOP1_SDWAe <bits<8> op> : VOP_SDWAe {
  bits<8> vdst;

  let Inst{8-0} = 0xf9; // sdwa
  let Inst{16-9} = op;
  let Inst{24-17} = vdst;
  let Inst{31-25} = 0x3f; // encoding
}

class VOP2_SDWAe <bits<6> op> : VOP_SDWAe {
  bits<8> vdst;
  bits<8> src1;

  let Inst{8-0} = 0xf9; // sdwa
  let Inst{16-9} = src1;
  let Inst{24-17} = vdst;
  let Inst{30-25} = op;
  let Inst{31} = 0x0; // encoding
}

class VOPC_SDWAe <bits<8> op> : VOP_SDWAe {
  bits<8> src1;

  let Inst{8-0} = 0xf9; // sdwa
  let Inst{16-9} = src1;
  let Inst{24-17} = op;
  let Inst{31-25} = 0x3e; // encoding

  // VOPC disallows dst_sel and dst_unused as they have no effect on destination
  let Inst{42-40} = 0x6;
  let Inst{44-43} = 0x2;
}

class EXPe_vi : EXPe {
  let Inst{31-26} = 0x31; //encoding
}

class VINTRPe_vi <bits<2> op> : VINTRPe <op> {
  let Inst{31-26} = 0x35; // encoding
}
