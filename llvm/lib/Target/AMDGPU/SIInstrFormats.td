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

class InstSI <dag outs, dag ins, string asm = "",
              list<dag> pattern = []> :
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
  field bits<1> SDWA = 0;
  field bits<1> DPP = 0;

  field bits<1> MUBUF = 0;
  field bits<1> MTBUF = 0;
  field bits<1> SMRD = 0;
  field bits<1> DS = 0;
  field bits<1> MIMG = 0;
  field bits<1> FLAT = 0;

  // Whether WQM _must_ be enabled for this instruction.
  field bits<1> WQM = 0;
  field bits<1> VGPRSpill = 0;
  field bits<1> SGPRSpill = 0;

  // This bit tells the assembler to use the 32-bit encoding in case it
  // is unable to infer the encoding from the operands.
  field bits<1> VOPAsmPrefer32Bit = 0;

  field bits<1> Gather4 = 0;

  // Whether WQM _must_ be disabled for this instruction.
  field bits<1> DisableWQM = 0;

  // Most sopk treat the immediate as a signed 16-bit, however some
  // use it as unsigned.
  field bits<1> SOPKZext = 0;

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
  let TSFlags{14} = SDWA;
  let TSFlags{15} = DPP;

  let TSFlags{16} = MUBUF;
  let TSFlags{17} = MTBUF;
  let TSFlags{18} = SMRD;
  let TSFlags{19} = DS;
  let TSFlags{20} = MIMG;
  let TSFlags{21} = FLAT;
  let TSFlags{22} = WQM;
  let TSFlags{23} = VGPRSpill;
  let TSFlags{24} = SGPRSpill;
  let TSFlags{25} = VOPAsmPrefer32Bit;
  let TSFlags{26} = Gather4;
  let TSFlags{27} = DisableWQM;
  let TSFlags{28} = SOPKZext;

  let SchedRW = [Write32Bit];

  field bits<1> DisableSIDecoder = 0;
  field bits<1> DisableVIDecoder = 0;
  field bits<1> DisableDecoder = 0;

  let isAsmParserOnly = !if(!eq(DisableDecoder{0}, {0}), 0, 1);
  let AsmVariantName = AMDGPUAsmVariants.Default;
}

class PseudoInstSI<dag outs, dag ins, list<dag> pattern = []>
  : InstSI<outs, ins, "", pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class SPseudoInstSI<dag outs, dag ins, list<dag> pattern = []>
  : PseudoInstSI<outs, ins, pattern> {
  let SALU = 1;
}

class VPseudoInstSI<dag outs, dag ins, list<dag> pattern = []>
  : PseudoInstSI<outs, ins, pattern> {
  let VALU = 1;
  let Uses = [EXEC];
}

class CFPseudoInstSI<dag outs, dag ins, list<dag> pattern = [],
  bit UseExec = 0, bit DefExec = 0> :
  SPseudoInstSI<outs, ins, pattern> {

  let Uses = !if(UseExec, [EXEC], []);
  let Defs = !if(DefExec, [EXEC, SCC], [SCC]);
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
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

class VOP3Common <dag outs, dag ins, string asm = "",
                  list<dag> pattern = [], bit HasMods = 0,
                  bit VOP3Only = 0> :
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
        "cvtVOP3",
        !if(!eq(HasMods,1), "cvtVOP3_2_mod", ""));

  let AsmVariantName = AMDGPUAsmVariants.VOP3;

  let isCodeGenOnly = 0;

  int Size = 8;

  // Because SGPRs may be allowed if there are multiple operands, we
  // need a post-isel hook to insert copies in order to avoid
  // violating constant bus requirements.
  let hasPostISelHook = 1;
}

} // End Uses = [EXEC]

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
  bits<8>  src1;
  bits<32> imm;

  let Inst{8-0} = src0;
  let Inst{16-9} = src1;
  let Inst{24-17} = vdst;
  let Inst{30-25} = op;
  let Inst{31} = 0x0; // encoding
  let Inst{63-32} = imm;
}

class VOP3a <bits<9> op> : Enc64 {
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;
  bits<2> omod;

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

class VOP3e <bits<9> op> : VOP3a <op> {
  bits<8> vdst;

  let Inst{7-0} = vdst;
}

// Encoding used for VOPC instructions encoded as VOP3
// Differs from VOP3e by destination name (sdst) as VOPC doesn't have vector dst
class VOP3ce <bits<9> op> : VOP3a <op> {
  bits<8> sdst;

  let Inst{7-0} = sdst;
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
  bits<8> src1;

  let Inst{8-0} = src0;
  let Inst{16-9} = src1;
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

class MIMG <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MIMG = 1;
  let Uses = [EXEC];

  let UseNamedOperandTable = 1;
  let hasSideEffects = 0; // XXX ????
}
