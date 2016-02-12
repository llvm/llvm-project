//===-- SIInstrFormats.td - SI Instruction Encodings ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SI Instruction format definitions.
//
//===----------------------------------------------------------------------===//

class InstSI <dag outs, dag ins, string asm, list<dag> pattern> :
    AMDGPUInst<outs, ins, asm, pattern>, PredicateControl {

  field bits<1> VM_CNT = 0;
  field bits<1> EXP_CNT = 0;
  field bits<1> LGKM_CNT = 0;

  field bits<1> SALU = 0;
  field bits<1> VALU = 0;

  field bits<1> SOP1 = 0;
  field bits<1> SOP2 = 0;
  field bits<1> SOPC = 0;
  field bits<1> SOPK = 0;
  field bits<1> SOPP = 0;

  field bits<1> VOP1 = 0;
  field bits<1> VOP2 = 0;
  field bits<1> VOP3 = 0;
  field bits<1> VOPC = 0;

  field bits<1> MUBUF = 0;
  field bits<1> MTBUF = 0;
  field bits<1> SMRD = 0;
  field bits<1> DS = 0;
  field bits<1> MIMG = 0;
  field bits<1> FLAT = 0;
  field bits<1> WQM = 0;
  field bits<1> VGPRSpill = 0;

  // This bit tells the assembler to use the 32-bit encoding in case it
  // is unable to infer the encoding from the operands.
  field bits<1> VOPAsmPrefer32Bit = 0;

  // These need to be kept in sync with the enum in SIInstrFlags.
  let TSFlags{0} = VM_CNT;
  let TSFlags{1} = EXP_CNT;
  let TSFlags{2} = LGKM_CNT;

  let TSFlags{3} = SALU;
  let TSFlags{4} = VALU;

  let TSFlags{5} = SOP1;
  let TSFlags{6} = SOP2;
  let TSFlags{7} = SOPC;
  let TSFlags{8} = SOPK;
  let TSFlags{9} = SOPP;

  let TSFlags{10} = VOP1;
  let TSFlags{11} = VOP2;
  let TSFlags{12} = VOP3;
  let TSFlags{13} = VOPC;

  let TSFlags{14} = MUBUF;
  let TSFlags{15} = MTBUF;
  let TSFlags{16} = SMRD;
  let TSFlags{17} = DS;
  let TSFlags{18} = MIMG;
  let TSFlags{19} = FLAT;
  let TSFlags{20} = WQM;
  let TSFlags{21} = VGPRSpill;
  let TSFlags{22} = VOPAsmPrefer32Bit;

  let SchedRW = [Write32Bit];
}

class Enc32 {
  field bits<32> Inst;
  int Size = 4;
}

class Enc64 {
  field bits<64> Inst;
  int Size = 8;
}

class VOPDstOperand <RegisterClass rc> : RegisterOperand <rc, "printVOPDst">;

let Uses = [EXEC] in {

class VOPAnyCommon <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let VALU = 1;
}

class VOPCCommon <dag ins, string asm, list<dag> pattern> :
    VOPAnyCommon <(outs), ins, asm, pattern> {

  let VOPC = 1;
  let Size = 4;
  let Defs = [VCC];
}

class VOP1Common <dag outs, dag ins, string asm, list<dag> pattern> :
    VOPAnyCommon <outs, ins, asm, pattern> {

  let VOP1 = 1;
  let Size = 4;
}

class VOP2Common <dag outs, dag ins, string asm, list<dag> pattern> :
    VOPAnyCommon <outs, ins, asm, pattern> {

  let VOP2 = 1;
  let Size = 4;
}

class VOP3Common <dag outs, dag ins, string asm, list<dag> pattern, bit HasMods = 0, bit VOP3Only = 0> :
    VOPAnyCommon <outs, ins, asm, pattern> {

  // Using complex patterns gives VOP3 patterns a very high complexity rating,
  // but standalone patterns are almost always prefered, so we need to adjust the
  // priority lower.  The goal is to use a high number to reduce complexity to
  // zero (or less than zero).
  let AddedComplexity = -1000;

  let VOP3 = 1;
  let VALU = 1;

  let AsmMatchConverter =
    !if(!eq(VOP3Only,1),
        "cvtVOP3_only",
        !if(!eq(HasMods,1), "cvtVOP3_2_mod", "cvtVOP3_2_nomod"));
  let isCodeGenOnly = 0;

  int Size = 8;

  // Because SGPRs may be allowed if there are multiple operands, we
  // need a post-isel hook to insert copies in order to avoid
  // violating constant bus requirements.
  let hasPostISelHook = 1;
}

} // End Uses = [EXEC]

//===----------------------------------------------------------------------===//
// Scalar operations
//===----------------------------------------------------------------------===//

class SOP1e <bits<8> op> : Enc32 {
  bits<7> sdst;
  bits<8> ssrc0;

  let Inst{7-0} = ssrc0;
  let Inst{15-8} = op;
  let Inst{22-16} = sdst;
  let Inst{31-23} = 0x17d; //encoding;
}

class SOP2e <bits<7> op> : Enc32 {
  bits<7> sdst;
  bits<8> ssrc0;
  bits<8> ssrc1;

  let Inst{7-0} = ssrc0;
  let Inst{15-8} = ssrc1;
  let Inst{22-16} = sdst;
  let Inst{29-23} = op;
  let Inst{31-30} = 0x2; // encoding
}

class SOPCe <bits<7> op> : Enc32 {
  bits<8> ssrc0;
  bits<8> ssrc1;

  let Inst{7-0} = ssrc0;
  let Inst{15-8} = ssrc1;
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17e;
}

class SOPKe <bits<5> op> : Enc32 {
  bits <7> sdst;
  bits <16> simm16;

  let Inst{15-0} = simm16;
  let Inst{22-16} = sdst;
  let Inst{27-23} = op;
  let Inst{31-28} = 0xb; //encoding
}

class SOPK64e <bits<5> op> : Enc64 {
  bits <7> sdst = 0;
  bits <16> simm16;
  bits <32> imm;

  let Inst{15-0} = simm16;
  let Inst{22-16} = sdst;
  let Inst{27-23} = op;
  let Inst{31-28} = 0xb;

  let Inst{63-32} = imm;
}

class SOPPe <bits<7> op> : Enc32 {
  bits <16> simm16;

  let Inst{15-0} = simm16;
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17f; // encoding
}

class SMRDe <bits<5> op, bits<1> imm> : Enc32 {
  bits<7> sdst;
  bits<7> sbase;
  bits<8> offset;

  let Inst{7-0} = offset;
  let Inst{8} = imm;
  let Inst{14-9} = sbase{6-1};
  let Inst{21-15} = sdst;
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
}

class SMRD_IMMe_ci <bits<5> op> : Enc64 {
  bits<7> sdst;
  bits<7> sbase;
  bits<32> offset;

  let Inst{7-0}   = 0xff;
  let Inst{8}     = 0;
  let Inst{14-9}  = sbase{6-1};
  let Inst{21-15} = sdst;
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
  let Inst{63-32} = offset;
}

let SchedRW = [WriteSALU] in {
class SOP1 <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI<outs, ins, asm, pattern> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let isCodeGenOnly = 0;
  let SALU = 1;
  let SOP1 = 1;
}

class SOP2 <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let isCodeGenOnly = 0;
  let SALU = 1;
  let SOP2 = 1;

  let UseNamedOperandTable = 1;
}

class SOPC <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
  InstSI<outs, ins, asm, pattern>, SOPCe <op> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPC = 1;
  let isCodeGenOnly = 0;
  let Defs = [SCC];

  let UseNamedOperandTable = 1;
}

class SOPK <dag outs, dag ins, string asm, list<dag> pattern> :
   InstSI <outs, ins , asm, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPK = 1;

  let UseNamedOperandTable = 1;
}

class SOPP <bits<7> op, dag ins, string asm, list<dag> pattern = []> :
		InstSI <(outs), ins, asm, pattern >, SOPPe <op> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPP = 1;

  let UseNamedOperandTable = 1;
}

} // let SchedRW = [WriteSALU]

class SMRD <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI<outs, ins, asm, pattern> {

  let LGKM_CNT = 1;
  let SMRD = 1;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let SchedRW = [WriteSMEM];
}

//===----------------------------------------------------------------------===//
// Vector ALU operations
//===----------------------------------------------------------------------===//

class VOP1e <bits<8> op> : Enc32 {
  bits<8> vdst;
  bits<9> src0;

  let Inst{8-0} = src0;
  let Inst{16-9} = op;
  let Inst{24-17} = vdst;
  let Inst{31-25} = 0x3f; //encoding
}

class VOP2e <bits<6> op> : Enc32 {
  bits<8> vdst;
  bits<9> src0;
  bits<8> src1;

  let Inst{8-0} = src0;
  let Inst{16-9} = src1;
  let Inst{24-17} = vdst;
  let Inst{30-25} = op;
  let Inst{31} = 0x0; //encoding
}

class VOP2_MADKe <bits<6> op> : Enc64 {

  bits<8>  vdst;
  bits<9>  src0;
  bits<8>  vsrc1;
  bits<32> src2;

  let Inst{8-0} = src0;
  let Inst{16-9} = vsrc1;
  let Inst{24-17} = vdst;
  let Inst{30-25} = op;
  let Inst{31} = 0x0; // encoding
  let Inst{63-32} = src2;
}

class VOP3e <bits<9> op> : Enc64 {
  bits<8> vdst;
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;
  bits<2> omod;

  let Inst{7-0} = vdst;
  let Inst{8} = src0_modifiers{1};
  let Inst{9} = src1_modifiers{1};
  let Inst{10} = src2_modifiers{1};
  let Inst{11} = clamp;
  let Inst{25-17} = op;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = src0;
  let Inst{49-41} = src1;
  let Inst{58-50} = src2;
  let Inst{60-59} = omod;
  let Inst{61} = src0_modifiers{0};
  let Inst{62} = src1_modifiers{0};
  let Inst{63} = src2_modifiers{0};
}

class VOP3be <bits<9> op> : Enc64 {
  bits<8> vdst;
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<7> sdst;
  bits<2> omod;

  let Inst{7-0} = vdst;
  let Inst{14-8} = sdst;
  let Inst{25-17} = op;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = src0;
  let Inst{49-41} = src1;
  let Inst{58-50} = src2;
  let Inst{60-59} = omod;
  let Inst{61} = src0_modifiers{0};
  let Inst{62} = src1_modifiers{0};
  let Inst{63} = src2_modifiers{0};
}

class VOPCe <bits<8> op> : Enc32 {
  bits<9> src0;
  bits<8> vsrc1;

  let Inst{8-0} = src0;
  let Inst{16-9} = vsrc1;
  let Inst{24-17} = op;
  let Inst{31-25} = 0x3e;
}

class VINTRPe <bits<2> op> : Enc32 {
  bits<8> vdst;
  bits<8> vsrc;
  bits<2> attrchan;
  bits<6> attr;

  let Inst{7-0} = vsrc;
  let Inst{9-8} = attrchan;
  let Inst{15-10} = attr;
  let Inst{17-16} = op;
  let Inst{25-18} = vdst;
  let Inst{31-26} = 0x32; // encoding
}

class DSe <bits<8> op> : Enc64 {
  bits<8> vdst;
  bits<1> gds;
  bits<8> addr;
  bits<8> data0;
  bits<8> data1;
  bits<8> offset0;
  bits<8> offset1;

  let Inst{7-0} = offset0;
  let Inst{15-8} = offset1;
  let Inst{17} = gds;
  let Inst{25-18} = op;
  let Inst{31-26} = 0x36; //encoding
  let Inst{39-32} = addr;
  let Inst{47-40} = data0;
  let Inst{55-48} = data1;
  let Inst{63-56} = vdst;
}

class MUBUFe <bits<7> op> : Enc64 {
  bits<12> offset;
  bits<1> offen;
  bits<1> idxen;
  bits<1> glc;
  bits<1> addr64;
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
  let Inst{15} = addr64;
  let Inst{16} = lds;
  let Inst{24-18} = op;
  let Inst{31-26} = 0x38; //encoding
  let Inst{39-32} = vaddr;
  let Inst{47-40} = vdata;
  let Inst{52-48} = srsrc{6-2};
  let Inst{54} = slc;
  let Inst{55} = tfe;
  let Inst{63-56} = soffset;
}

class MTBUFe <bits<3> op> : Enc64 {
  bits<8> vdata;
  bits<12> offset;
  bits<1> offen;
  bits<1> idxen;
  bits<1> glc;
  bits<1> addr64;
  bits<4> dfmt;
  bits<3> nfmt;
  bits<8> vaddr;
  bits<7> srsrc;
  bits<1> slc;
  bits<1> tfe;
  bits<8> soffset;

  let Inst{11-0} = offset;
  let Inst{12} = offen;
  let Inst{13} = idxen;
  let Inst{14} = glc;
  let Inst{15} = addr64;
  let Inst{18-16} = op;
  let Inst{22-19} = dfmt;
  let Inst{25-23} = nfmt;
  let Inst{31-26} = 0x3a; //encoding
  let Inst{39-32} = vaddr;
  let Inst{47-40} = vdata;
  let Inst{52-48} = srsrc{6-2};
  let Inst{54} = slc;
  let Inst{55} = tfe;
  let Inst{63-56} = soffset;
}

class MIMGe <bits<7> op> : Enc64 {
  bits<8> vdata;
  bits<4> dmask;
  bits<1> unorm;
  bits<1> glc;
  bits<1> da;
  bits<1> r128;
  bits<1> tfe;
  bits<1> lwe;
  bits<1> slc;
  bits<8> vaddr;
  bits<7> srsrc;
  bits<7> ssamp;

  let Inst{11-8} = dmask;
  let Inst{12} = unorm;
  let Inst{13} = glc;
  let Inst{14} = da;
  let Inst{15} = r128;
  let Inst{16} = tfe;
  let Inst{17} = lwe;
  let Inst{24-18} = op;
  let Inst{25} = slc;
  let Inst{31-26} = 0x3c;
  let Inst{39-32} = vaddr;
  let Inst{47-40} = vdata;
  let Inst{52-48} = srsrc{6-2};
  let Inst{57-53} = ssamp{6-2};
}

class FLATe<bits<7> op> : Enc64 {
  bits<8> addr;
  bits<8> data;
  bits<8> vdst;
  bits<1> slc;
  bits<1> glc;
  bits<1> tfe;

  // 15-0 is reserved.
  let Inst{16} = glc;
  let Inst{17} = slc;
  let Inst{24-18} = op;
  let Inst{31-26} = 0x37; // Encoding.
  let Inst{39-32} = addr;
  let Inst{47-40} = data;
  // 54-48 is reserved.
  let Inst{55} = tfe;
  let Inst{63-56} = vdst;
}

class EXPe : Enc64 {
  bits<4> en;
  bits<6> tgt;
  bits<1> compr;
  bits<1> done;
  bits<1> vm;
  bits<8> vsrc0;
  bits<8> vsrc1;
  bits<8> vsrc2;
  bits<8> vsrc3;

  let Inst{3-0} = en;
  let Inst{9-4} = tgt;
  let Inst{10} = compr;
  let Inst{11} = done;
  let Inst{12} = vm;
  let Inst{31-26} = 0x3e;
  let Inst{39-32} = vsrc0;
  let Inst{47-40} = vsrc1;
  let Inst{55-48} = vsrc2;
  let Inst{63-56} = vsrc3;
}

let Uses = [EXEC] in {

class VOP1 <bits<8> op, dag outs, dag ins, string asm, list<dag> pattern> :
    VOP1Common <outs, ins, asm, pattern>,
    VOP1e<op> {
  let isCodeGenOnly = 0;
}

class VOP2 <bits<6> op, dag outs, dag ins, string asm, list<dag> pattern> :
    VOP2Common <outs, ins, asm, pattern>, VOP2e<op> {
  let isCodeGenOnly = 0;
}

class VOPC <bits<8> op, dag ins, string asm, list<dag> pattern> :
    VOPCCommon <ins, asm, pattern>, VOPCe <op>;

class VINTRPCommon <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
}

} // End Uses = [EXEC]

//===----------------------------------------------------------------------===//
// Vector I/O operations
//===----------------------------------------------------------------------===//

class DS <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {

  let LGKM_CNT = 1;
  let DS = 1;
  let UseNamedOperandTable = 1;
  let Uses = [M0, EXEC];

  // Most instruction load and store data, so set this as the default.
  let mayLoad = 1;
  let mayStore = 1;

  let hasSideEffects = 0;
  let AsmMatchConverter = "cvtDS";
  let SchedRW = [WriteLDS];
}

class MUBUF <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI<outs, ins, asm, pattern> {

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MUBUF = 1;
  let Uses = [EXEC];

  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let AsmMatchConverter = "cvtMubuf";
  let SchedRW = [WriteVMEM];
}

class MTBUF <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI<outs, ins, asm, pattern> {

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MTBUF = 1;
  let Uses = [EXEC];

  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let SchedRW = [WriteVMEM];
}

class FLAT <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI<outs, ins, asm, pattern>, FLATe <op> {
  let FLAT = 1;
  // Internally, FLAT instruction are executed as both an LDS and a
  // Buffer instruction; so, they increment both VM_CNT and LGKM_CNT
  // and are not considered done until both have been decremented.
  let VM_CNT = 1;
  let LGKM_CNT = 1;

  let Uses = [EXEC, FLAT_SCR]; // M0

  let UseNamedOperandTable = 1;
  let hasSideEffects = 0;
  let AsmMatchConverter = "cvtFlat";
  let SchedRW = [WriteVMEM];
}

class MIMG <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern>, MIMGe <op> {

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MIMG = 1;
  let Uses = [EXEC];

  let UseNamedOperandTable = 1;
  let hasSideEffects = 0; // XXX ????
}
