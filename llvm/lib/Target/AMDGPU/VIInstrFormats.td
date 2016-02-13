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

class DSe_vi <bits<8> op> : Enc64 {
  bits<8> vdst;
  bits<1> gds;
  bits<8> addr;
  bits<8> data0;
  bits<8> data1;
  bits<8> offset0;
  bits<8> offset1;

  let Inst{7-0} = offset0;
  let Inst{15-8} = offset1;
  let Inst{16} = gds;
  let Inst{24-17} = op;
  let Inst{31-26} = 0x36; //encoding
  let Inst{39-32} = addr;
  let Inst{47-40} = data0;
  let Inst{55-48} = data1;
  let Inst{63-56} = vdst;
}

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
  bits<7>  sdata;
  bits<1>  glc;
  bits<20> offset;

  let Inst{5-0}   = sbase{6-1};
  let Inst{12-6}  = sdata;
  let Inst{16}    = glc;
  let Inst{17}    = imm;
  let Inst{25-18} = op;
  let Inst{31-26} = 0x30; //encoding
  let Inst{51-32} = offset;
}

class VOP3e_vi <bits<10> op> : Enc64 {
  bits<8> vdst;
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;
  bits<2> omod;

  let Inst{7-0}   = vdst;
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

class VOP_DPP <dag outs, dag ins, string asm, list<dag> pattern> :
    VOPAnyCommon <outs, ins, asm, pattern> {
  let Size = 8;
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
  bits<8> dst;

  let Inst{8-0} = 0xfa; // dpp
  let Inst{16-9} = op;
  let Inst{24-17} = dst;
  let Inst{31-25} = 0x3f; //encoding
}

class VOP2_DPPe <bits<6> op> : Enc32 {
  bits<8> dst;
  bits<8> src1;

  let Inst{8-0} = 0xfa; //dpp
  let Inst{16-9} = src1;
  let Inst{24-17} = dst;
  let Inst{30-25} = op;
  let Inst{31} = 0x0; //encoding
}

class EXPe_vi : EXPe {
  let Inst{31-26} = 0x31; //encoding
}

class VINTRPe_vi <bits<2> op> : VINTRPe <op> {
  let Inst{31-26} = 0x35; // encoding
}
