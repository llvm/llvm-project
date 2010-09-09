//===- ARMInstrFormats.td - ARM Instruction Formats --*- tablegen -*---------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// ARM Instruction Format Definitions.
//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<6> val> {
  bits<6> Value = val;
}

def Pseudo        : Format<0>;
def MulFrm        : Format<1>;
def BrFrm         : Format<2>;
def BrMiscFrm     : Format<3>;

def DPFrm         : Format<4>;
def DPSoRegFrm    : Format<5>;

def LdFrm         : Format<6>;
def StFrm         : Format<7>;
def LdMiscFrm     : Format<8>;
def StMiscFrm     : Format<9>;
def LdStMulFrm    : Format<10>;

def LdStExFrm     : Format<11>;

def ArithMiscFrm  : Format<12>;
def SatFrm        : Format<13>;
def ExtFrm        : Format<14>;

def VFPUnaryFrm   : Format<15>;
def VFPBinaryFrm  : Format<16>;
def VFPConv1Frm   : Format<17>;
def VFPConv2Frm   : Format<18>;
def VFPConv3Frm   : Format<19>;
def VFPConv4Frm   : Format<20>;
def VFPConv5Frm   : Format<21>;
def VFPLdStFrm    : Format<22>;
def VFPLdStMulFrm : Format<23>;
def VFPMiscFrm    : Format<24>;

def ThumbFrm      : Format<25>;
def MiscFrm       : Format<26>;

def NGetLnFrm     : Format<27>;
def NSetLnFrm     : Format<28>;
def NDupFrm       : Format<29>;
def NLdStFrm      : Format<30>;
def N1RegModImmFrm: Format<31>;
def N2RegFrm      : Format<32>;
def NVCVTFrm      : Format<33>;
def NVDupLnFrm    : Format<34>;
def N2RegVShLFrm  : Format<35>;
def N2RegVShRFrm  : Format<36>;
def N3RegFrm      : Format<37>;
def N3RegVShFrm   : Format<38>;
def NVExtFrm      : Format<39>;
def NVMulSLFrm    : Format<40>;
def NVTBLFrm      : Format<41>;

// Misc flags.

// the instruction has a Rn register operand.
// UnaryDP - Indicates this is a unary data processing instruction, i.e.
// it doesn't have a Rn operand.
class UnaryDP    { bit isUnaryDataProc = 1; }

// Xform16Bit - Indicates this Thumb2 instruction may be transformed into
// a 16-bit Thumb instruction if certain conditions are met.
class Xform16Bit { bit canXformTo16Bit = 1; }

//===----------------------------------------------------------------------===//
// ARM Instruction flags.  These need to match ARMBaseInstrInfo.h.
//

// Addressing mode.
class AddrMode<bits<4> val> {
  bits<4> Value = val;
}
def AddrModeNone    : AddrMode<0>;
def AddrMode1       : AddrMode<1>;
def AddrMode2       : AddrMode<2>;
def AddrMode3       : AddrMode<3>;
def AddrMode4       : AddrMode<4>;
def AddrMode5       : AddrMode<5>;
def AddrMode6       : AddrMode<6>;
def AddrModeT1_1    : AddrMode<7>;
def AddrModeT1_2    : AddrMode<8>;
def AddrModeT1_4    : AddrMode<9>;
def AddrModeT1_s    : AddrMode<10>;
def AddrModeT2_i12  : AddrMode<11>;
def AddrModeT2_i8   : AddrMode<12>;
def AddrModeT2_so   : AddrMode<13>;
def AddrModeT2_pc   : AddrMode<14>;
def AddrModeT2_i8s4 : AddrMode<15>;

// Instruction size.
class SizeFlagVal<bits<3> val> {
  bits<3> Value = val;
}
def SizeInvalid  : SizeFlagVal<0>;  // Unset.
def SizeSpecial  : SizeFlagVal<1>;  // Pseudo or special.
def Size8Bytes   : SizeFlagVal<2>;
def Size4Bytes   : SizeFlagVal<3>;
def Size2Bytes   : SizeFlagVal<4>;

// Load / store index mode.
class IndexMode<bits<2> val> {
  bits<2> Value = val;
}
def IndexModeNone : IndexMode<0>;
def IndexModePre  : IndexMode<1>;
def IndexModePost : IndexMode<2>;
def IndexModeUpd  : IndexMode<3>;

// Instruction execution domain.
class Domain<bits<2> val> {
  bits<2> Value = val;
}
def GenericDomain : Domain<0>;
def VFPDomain     : Domain<1>; // Instructions in VFP domain only
def NeonDomain    : Domain<2>; // Instructions in Neon domain only
def VFPNeonDomain : Domain<3>; // Instructions in both VFP & Neon domains

//===----------------------------------------------------------------------===//

// ARM special operands.
//

def CondCodeOperand : AsmOperandClass {
  let Name = "CondCode";
  let SuperClasses = [];
}

// ARM Predicate operand. Default to 14 = always (AL). Second part is CC
// register whose default is 0 (no register).
def pred : PredicateOperand<OtherVT, (ops i32imm, CCR),
                                     (ops (i32 14), (i32 zero_reg))> {
  let PrintMethod = "printPredicateOperand";
  let ParserMatchClass = CondCodeOperand;
}

// Conditional code result for instructions whose 's' bit is set, e.g. subs.
def cc_out : OptionalDefOperand<OtherVT, (ops CCR), (ops (i32 zero_reg))> {
  let PrintMethod = "printSBitModifierOperand";
}

// Same as cc_out except it defaults to setting CPSR.
def s_cc_out : OptionalDefOperand<OtherVT, (ops CCR), (ops (i32 CPSR))> {
  let PrintMethod = "printSBitModifierOperand";
}

// ARM special operands for disassembly only.
//

def cps_opt : Operand<i32> {
  let PrintMethod = "printCPSOptionOperand";
}

def msr_mask : Operand<i32> {
  let PrintMethod = "printMSRMaskOperand";
}

// A8.6.117, A8.6.118.  Different instructions are generated for #0 and #-0.
// The neg_zero operand translates -0 to -1, -1 to -2, ..., etc.
def neg_zero : Operand<i32> {
  let PrintMethod = "printNegZeroOperand";
}

//===----------------------------------------------------------------------===//

// ARM Instruction templates.
//

class InstTemplate<AddrMode am, SizeFlagVal sz, IndexMode im,
                   Format f, Domain d, string cstr, InstrItinClass itin>
  : Instruction {
  let Namespace = "ARM";

  AddrMode AM = am;
  SizeFlagVal SZ = sz;
  IndexMode IM = im;
  bits<2> IndexModeBits = IM.Value;
  Format F = f;
  bits<6> Form = F.Value;
  Domain D = d;
  bit isUnaryDataProc = 0;
  bit canXformTo16Bit = 0;

  // The layout of TSFlags should be kept in sync with ARMBaseInstrInfo.h.
  let TSFlags{3-0}   = AM.Value;
  let TSFlags{6-4}   = SZ.Value;
  let TSFlags{8-7}   = IndexModeBits;
  let TSFlags{14-9}  = Form;
  let TSFlags{15}    = isUnaryDataProc;
  let TSFlags{16}    = canXformTo16Bit;
  let TSFlags{18-17} = D.Value;

  let Constraints = cstr;
  let Itinerary = itin;
}

class Encoding {
  field bits<32> Inst;
}

class InstARM<AddrMode am, SizeFlagVal sz, IndexMode im,
              Format f, Domain d, string cstr, InstrItinClass itin>
  : InstTemplate<am, sz, im, f, d, cstr, itin>, Encoding;

// This Encoding-less class is used by Thumb1 to specify the encoding bits later
// on by adding flavors to specific instructions.
class InstThumb<AddrMode am, SizeFlagVal sz, IndexMode im,
                Format f, Domain d, string cstr, InstrItinClass itin>
  : InstTemplate<am, sz, im, f, d, cstr, itin>;

class PseudoInst<dag oops, dag iops, InstrItinClass itin,
                 string asm, list<dag> pattern>
  : InstARM<AddrModeNone, SizeSpecial, IndexModeNone, Pseudo, GenericDomain,
            "", itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString = asm;
  let Pattern = pattern;
}

// Almost all ARM instructions are predicable.
class I<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
        IndexMode im, Format f, InstrItinClass itin,
        string opc, string asm, string cstr,
        list<dag> pattern>
  : InstARM<am, sz, im, f, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p));
  let AsmString = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

// A few are not predicable
class InoP<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
           IndexMode im, Format f, InstrItinClass itin,
           string opc, string asm, string cstr,
           list<dag> pattern>
  : InstARM<am, sz, im, f, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString = !strconcat(opc, asm);
  let Pattern = pattern;
  let isPredicable = 0;
  list<Predicate> Predicates = [IsARM];
}

// Same as I except it can optionally modify CPSR. Note it's modeled as an input
// operand since by default it's a zero register. It will become an implicit def
// once it's "flipped".
class sI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
         IndexMode im, Format f, InstrItinClass itin,
         string opc, string asm, string cstr,
         list<dag> pattern>
  : InstARM<am, sz, im, f, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p, cc_out:$s));
  let AsmString = !strconcat(opc, !strconcat("${p}${s}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

// Special cases
class XI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
         IndexMode im, Format f, InstrItinClass itin,
         string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, im, f, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

class AI<dag oops, dag iops, Format f, InstrItinClass itin,
         string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern>;
class AsI<dag oops, dag iops, Format f, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : sI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, itin,
       opc, asm, "", pattern>;
class AXI<dag oops, dag iops, Format f, InstrItinClass itin,
          string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern>;
class AInoP<dag oops, dag iops, Format f, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : InoP<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, itin,
         opc, asm, "", pattern>;

// Ctrl flow instructions
class ABI<bits<4> opcod, dag oops, dag iops, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, BrFrm, itin,
      opc, asm, "", pattern> {
  let Inst{27-24} = opcod;
}
class ABXI<bits<4> opcod, dag oops, dag iops, InstrItinClass itin,
           string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, BrFrm, itin,
       asm, "", pattern> {
  let Inst{27-24} = opcod;
}
class ABXIx2<dag oops, dag iops, InstrItinClass itin,
             string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, Size8Bytes, IndexModeNone, Pseudo, itin,
       asm, "", pattern>;

// BR_JT instructions
class JTI<dag oops, dag iops, InstrItinClass itin,
          string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, SizeSpecial, IndexModeNone, BrMiscFrm, itin,
       asm, "", pattern>;

// Atomic load/store instructions
class AIldrex<bits<2> opcod, dag oops, dag iops, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, LdStExFrm, itin,
      opc, asm, "", pattern> {
  let Inst{27-23} = 0b00011;
  let Inst{22-21} = opcod;
  let Inst{20}    = 1;
  let Inst{11-0}  = 0b111110011111;
}
class AIstrex<bits<2> opcod, dag oops, dag iops, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, LdStExFrm, itin,
      opc, asm, "", pattern> {
  let Inst{27-23} = 0b00011;
  let Inst{22-21} = opcod;
  let Inst{20}    = 0;
  let Inst{11-4}  = 0b11111001;
}

// addrmode1 instructions
class AI1<bits<4> opcod, dag oops, dag iops, Format f, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{24-21} = opcod;
  let Inst{27-26} = 0b00;
}
class AsI1<bits<4> opcod, dag oops, dag iops, Format f, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : sI<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, itin,
       opc, asm, "", pattern> {
  let Inst{24-21} = opcod;
  let Inst{27-26} = 0b00;
}
class AXI1<bits<4> opcod, dag oops, dag iops, Format f, InstrItinClass itin,
           string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{24-21} = opcod;
  let Inst{27-26} = 0b00;
}
class AI1x2<dag oops, dag iops, Format f, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode1, Size8Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern>;


// addrmode2 loads and stores
class AI2<dag oops, dag iops, Format f, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{27-26} = 0b01;
}

// loads
class AI2ldw<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}
class AXI2ldw<dag oops, dag iops, Format f, InstrItinClass itin,
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}
class AI2ldb<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}
class AXI2ldb<dag oops, dag iops, Format f, InstrItinClass itin,
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}

// stores
class AI2stw<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}
class AXI2stw<dag oops, dag iops, Format f, InstrItinClass itin,
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}
class AI2stb<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}
class AXI2stb<dag oops, dag iops, Format f, InstrItinClass itin,
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}

// Pre-indexed loads
class AI2ldwpr<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}
class AI2ldbpr<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}

// Pre-indexed stores
class AI2stwpr<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}
class AI2stbpr<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = 0b01;
}

// Post-indexed loads
class AI2ldwpo<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = 0b01;
}
class AI2ldbpo<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = 0b01;
}

// Post-indexed stores
class AI2stwpo<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = 0b01;
}
class AI2stbpo<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = 0b01;
}

// addrmode3 instructions
class AI3<dag oops, dag iops, Format f, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern>;
class AXI3<dag oops, dag iops, Format f, InstrItinClass itin,
           string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern>;

// loads
class AI3ldh<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}
class AXI3ldh<dag oops, dag iops, Format f, InstrItinClass itin,
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3ldsh<dag oops, dag iops, Format f, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}
class AXI3ldsh<dag oops, dag iops, Format f, InstrItinClass itin,
               string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3ldsb<dag oops, dag iops, Format f, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}
class AXI3ldsb<dag oops, dag iops, Format f, InstrItinClass itin,
               string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3ldd<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}

// stores
class AI3sth<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}
class AXI3sth<dag oops, dag iops, Format f, InstrItinClass itin,
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3std<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}

// Pre-indexed loads
class AI3ldhpr<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}
class AI3ldshpr<dag oops, dag iops, Format f, InstrItinClass itin,
                string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}
class AI3ldsbpr<dag oops, dag iops, Format f, InstrItinClass itin,
                string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}
class AI3lddpr<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}


// Pre-indexed stores
class AI3sthpr<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}
class AI3stdpr<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
  let Inst{27-25} = 0b000;
}

// Post-indexed loads
class AI3ldhpo<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 0; // P bit
  let Inst{27-25} = 0b000;
}
class AI3ldshpo<dag oops, dag iops, Format f, InstrItinClass itin,
                string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 0; // P bit
  let Inst{27-25} = 0b000;
}
class AI3ldsbpo<dag oops, dag iops, Format f, InstrItinClass itin,
                string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 0; // P bit
  let Inst{27-25} = 0b000;
}
class AI3lddpo<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 0; // P bit
  let Inst{27-25} = 0b000;
}

// Post-indexed stores
class AI3sthpo<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 0; // P bit
  let Inst{27-25} = 0b000;
}
class AI3stdpo<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 0; // P bit
  let Inst{27-25} = 0b000;
}

// addrmode4 instructions
class AXI4ld<dag oops, dag iops, IndexMode im, Format f, InstrItinClass itin,
             string asm, string cstr, list<dag> pattern>
  : XI<oops, iops, AddrMode4, Size4Bytes, im, f, itin,
       asm, cstr, pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{22}    = 0; // S bit
  let Inst{27-25} = 0b100;
}
class AXI4st<dag oops, dag iops, IndexMode im, Format f, InstrItinClass itin,
             string asm, string cstr, list<dag> pattern>
  : XI<oops, iops, AddrMode4, Size4Bytes, im, f, itin,
       asm, cstr, pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{22}    = 0; // S bit
  let Inst{27-25} = 0b100;
}

// Unsigned multiply, multiply-accumulate instructions.
class AMul1I<bits<7> opcod, dag oops, dag iops, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, MulFrm, itin,
      opc, asm, "", pattern> {
  let Inst{7-4}   = 0b1001;
  let Inst{20}    = 0; // S bit
  let Inst{27-21} = opcod;
}
class AsMul1I<bits<7> opcod, dag oops, dag iops, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
  : sI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, MulFrm, itin,
       opc, asm, "", pattern> {
  let Inst{7-4}   = 0b1001;
  let Inst{27-21} = opcod;
}

// Most significant word multiply
class AMul2I<bits<7> opcod, dag oops, dag iops, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, MulFrm, itin,
      opc, asm, "", pattern> {
  let Inst{7-4}   = 0b1001;
  let Inst{20}    = 1;
  let Inst{27-21} = opcod;
}

// SMUL<x><y> / SMULW<y> / SMLA<x><y> / SMLAW<x><y>
class AMulxyI<bits<7> opcod, dag oops, dag iops, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, MulFrm, itin,
      opc, asm, "", pattern> {
  let Inst{4}     = 0;
  let Inst{7}     = 1;
  let Inst{20}    = 0;
  let Inst{27-21} = opcod;
}

// Extend instructions.
class AExtI<bits<8> opcod, dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, ExtFrm, itin,
      opc, asm, "", pattern> {
  let Inst{7-4}   = 0b0111;
  let Inst{27-20} = opcod;
}

// Misc Arithmetic instructions.
class AMiscA1I<bits<8> opcod, dag oops, dag iops, InstrItinClass itin,
               string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, ArithMiscFrm, itin,
      opc, asm, "", pattern> {
  let Inst{27-20} = opcod;
}

//===----------------------------------------------------------------------===//

// ARMPat - Same as Pat<>, but requires that the compiler be in ARM mode.
class ARMPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsARM];
}
class ARMV5TEPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsARM, HasV5TE];
}
class ARMV6Pat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsARM, HasV6];
}

//===----------------------------------------------------------------------===//
//
// Thumb Instruction Format Definitions.
//

// TI - Thumb instruction.

class ThumbI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
             InstrItinClass itin, string asm, string cstr, list<dag> pattern>
  : InstThumb<am, sz, IndexModeNone, ThumbFrm, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb];
}

class TI<dag oops, dag iops, InstrItinClass itin, string asm, list<dag> pattern>
  : ThumbI<oops, iops, AddrModeNone, Size2Bytes, itin, asm, "", pattern>;

// Two-address instructions
class TIt<dag oops, dag iops, InstrItinClass itin, string asm,
          list<dag> pattern>
  : ThumbI<oops, iops, AddrModeNone, Size2Bytes, itin, asm, "$lhs = $dst",
           pattern>;

// tBL, tBX 32-bit instructions
class TIx2<bits<5> opcod1, bits<2> opcod2, bit opcod3,
           dag oops, dag iops, InstrItinClass itin, string asm,
           list<dag> pattern>
    : ThumbI<oops, iops, AddrModeNone, Size4Bytes, itin, asm, "", pattern>,
      Encoding {
  let Inst{31-27} = opcod1;
  let Inst{15-14} = opcod2;
  let Inst{12}    = opcod3;
}

// BR_JT instructions
class TJTI<dag oops, dag iops, InstrItinClass itin, string asm,
           list<dag> pattern>
  : ThumbI<oops, iops, AddrModeNone, SizeSpecial, itin, asm, "", pattern>;

// Thumb1 only
class Thumb1I<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
              InstrItinClass itin, string asm, string cstr, list<dag> pattern>
  : InstThumb<am, sz, IndexModeNone, ThumbFrm, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb1Only];
}

class T1I<dag oops, dag iops, InstrItinClass itin,
          string asm, list<dag> pattern>
  : Thumb1I<oops, iops, AddrModeNone, Size2Bytes, itin, asm, "", pattern>;
class T1Ix2<dag oops, dag iops, InstrItinClass itin,
            string asm, list<dag> pattern>
  : Thumb1I<oops, iops, AddrModeNone, Size4Bytes, itin, asm, "", pattern>;
class T1JTI<dag oops, dag iops, InstrItinClass itin,
            string asm, list<dag> pattern>
  : Thumb1I<oops, iops, AddrModeNone, SizeSpecial, itin, asm, "", pattern>;

// Two-address instructions
class T1It<dag oops, dag iops, InstrItinClass itin,
           string asm, string cstr, list<dag> pattern>
  : Thumb1I<oops, iops, AddrModeNone, Size2Bytes, itin,
            asm, cstr, pattern>;

// Thumb1 instruction that can either be predicated or set CPSR.
class Thumb1sI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
               InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : InstThumb<am, sz, IndexModeNone, ThumbFrm, GenericDomain, cstr, itin> {
  let OutOperandList = !con(oops, (outs s_cc_out:$s));
  let InOperandList = !con(iops, (ins pred:$p));
  let AsmString = !strconcat(opc, !strconcat("${s}${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb1Only];
}

class T1sI<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : Thumb1sI<oops, iops, AddrModeNone, Size2Bytes, itin, opc, asm, "", pattern>;

// Two-address instructions
class T1sIt<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb1sI<oops, iops, AddrModeNone, Size2Bytes, itin, opc, asm,
             "$lhs = $dst", pattern>;

// Thumb1 instruction that can be predicated.
class Thumb1pI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
               InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : InstThumb<am, sz, IndexModeNone, ThumbFrm, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p));
  let AsmString = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb1Only];
}

class T1pI<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : Thumb1pI<oops, iops, AddrModeNone, Size2Bytes, itin, opc, asm, "", pattern>;

// Two-address instructions
class T1pIt<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb1pI<oops, iops, AddrModeNone, Size2Bytes, itin, opc, asm,
             "$lhs = $dst", pattern>;

class T1pI1<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb1pI<oops, iops, AddrModeT1_1, Size2Bytes, itin, opc, asm, "", pattern>;
class T1pI2<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb1pI<oops, iops, AddrModeT1_2, Size2Bytes, itin, opc, asm, "", pattern>;
class T1pI4<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb1pI<oops, iops, AddrModeT1_4, Size2Bytes, itin, opc, asm, "", pattern>;
class T1pIs<dag oops, dag iops,
            InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : Thumb1pI<oops, iops, AddrModeT1_s, Size2Bytes, itin, opc, asm, "", pattern>;

class Encoding16 : Encoding {
  let Inst{31-16} = 0x0000;
}

// A6.2 16-bit Thumb instruction encoding
class T1Encoding<bits<6> opcode> : Encoding16 {
  let Inst{15-10} = opcode;
}

// A6.2.1 Shift (immediate), add, subtract, move, and compare encoding.
class T1General<bits<5> opcode> : Encoding16 {
  let Inst{15-14} = 0b00;
  let Inst{13-9} = opcode;
}

// A6.2.2 Data-processing encoding.
class T1DataProcessing<bits<4> opcode> : Encoding16 {
  let Inst{15-10} = 0b010000;
  let Inst{9-6} = opcode;
}

// A6.2.3 Special data instructions and branch and exchange encoding.
class T1Special<bits<4> opcode> : Encoding16 {
  let Inst{15-10} = 0b010001;
  let Inst{9-6} = opcode;
}

// A6.2.4 Load/store single data item encoding.
class T1LoadStore<bits<4> opA, bits<3> opB> : Encoding16 {
  let Inst{15-12} = opA;
  let Inst{11-9}  = opB;
}
class T1LdSt<bits<3> opB>     : T1LoadStore<0b0101, opB>;
class T1LdSt4Imm<bits<3> opB> : T1LoadStore<0b0110, opB>; // Immediate, 4 bytes
class T1LdSt1Imm<bits<3> opB> : T1LoadStore<0b0111, opB>; // Immediate, 1 byte
class T1LdSt2Imm<bits<3> opB> : T1LoadStore<0b1000, opB>; // Immediate, 2 bytes
class T1LdStSP<bits<3> opB>   : T1LoadStore<0b1001, opB>; // SP relative

// A6.2.5 Miscellaneous 16-bit instructions encoding.
class T1Misc<bits<7> opcode> : Encoding16 {
  let Inst{15-12} = 0b1011;
  let Inst{11-5} = opcode;
}

// Thumb2I - Thumb2 instruction. Almost all Thumb2 instructions are predicable.
class Thumb2I<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
              InstrItinClass itin,
              string opc, string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p));
  let AsmString = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb2];
}

// Same as Thumb2I except it can optionally modify CPSR. Note it's modeled as an
// input operand since by default it's a zero register. It will become an
// implicit def once it's "flipped".
// 
// FIXME: This uses unified syntax so {s} comes before {p}. We should make it
// more consistent.
class Thumb2sI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
               InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p, cc_out:$s));
  let AsmString = !strconcat(opc, !strconcat("${s}${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb2];
}

// Special cases
class Thumb2XI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
               InstrItinClass itin,
               string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb2];
}

class ThumbXI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
              InstrItinClass itin,
              string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb1Only];
}

class T2I<dag oops, dag iops, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeNone, Size4Bytes, itin, opc, asm, "", pattern>;
class T2Ii12<dag oops, dag iops, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_i12, Size4Bytes, itin, opc, asm, "",pattern>;
class T2Ii8<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_i8, Size4Bytes, itin, opc, asm, "", pattern>;
class T2Iso<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_so, Size4Bytes, itin, opc, asm, "", pattern>;
class T2Ipc<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_pc, Size4Bytes, itin, opc, asm, "", pattern>;
class T2Ii8s4<bit P, bit W, bit load, dag oops, dag iops, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_i8s4, Size4Bytes, itin, opc, asm, "",
            pattern> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b00;
  let Inst{24}    = P;
  let Inst{23}    = ?; // The U bit.
  let Inst{22}    = 1;
  let Inst{21}    = W;
  let Inst{20}    = load;
}

class T2sI<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : Thumb2sI<oops, iops, AddrModeNone, Size4Bytes, itin, opc, asm, "", pattern>;

class T2XI<dag oops, dag iops, InstrItinClass itin,
           string asm, list<dag> pattern>
  : Thumb2XI<oops, iops, AddrModeNone, Size4Bytes, itin, asm, "", pattern>;
class T2JTI<dag oops, dag iops, InstrItinClass itin,
            string asm, list<dag> pattern>
  : Thumb2XI<oops, iops, AddrModeNone, SizeSpecial, itin, asm, "", pattern>;

class T2Ix2<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeNone, Size8Bytes, itin, opc, asm, "", pattern>;

// Two-address instructions
class T2XIt<dag oops, dag iops, InstrItinClass itin,
            string asm, string cstr, list<dag> pattern>
  : Thumb2XI<oops, iops, AddrModeNone, Size4Bytes, itin, asm, cstr, pattern>;

// T2Iidxldst - Thumb2 indexed load / store instructions.
class T2Iidxldst<bit signed, bits<2> opcod, bit load, bit pre,
                 dag oops, dag iops,
                 AddrMode am, IndexMode im, InstrItinClass itin,
                 string opc, string asm, string cstr, list<dag> pattern>
  : InstARM<am, Size4Bytes, im, ThumbFrm, GenericDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p));
  let AsmString = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb2];
  let Inst{31-27} = 0b11111;
  let Inst{26-25} = 0b00;
  let Inst{24}    = signed;
  let Inst{23}    = 0;
  let Inst{22-21} = opcod;
  let Inst{20}    = load;
  let Inst{11}    = 1;
  // (P, W) = (1, 1) Pre-indexed or (0, 1) Post-indexed
  let Inst{10}    = pre; // The P bit.
  let Inst{8}     = 1; // The W bit.
}

// Helper class for disassembly only
// A6.3.16 & A6.3.17
// T2Imac - Thumb2 multiply [accumulate, and absolute difference] instructions.
class T2I_mac<bit long, bits<3> op22_20, bits<4> op7_4, dag oops, dag iops,
             InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-27} = 0b11111;
  let Inst{26-24} = 0b011;
  let Inst{23}    = long;
  let Inst{22-20} = op22_20;
  let Inst{7-4}   = op7_4;
}

// Tv5Pat - Same as Pat<>, but requires V5T Thumb mode.
class Tv5Pat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsThumb1Only, HasV5T];
}

// T1Pat - Same as Pat<>, but requires that the compiler be in Thumb1 mode.
class T1Pat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsThumb1Only];
}

// T2Pat - Same as Pat<>, but requires that the compiler be in Thumb2 mode.
class T2Pat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsThumb2];
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM VFP Instruction templates.
//

// Almost all VFP instructions are predicable.
class VFPI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
           IndexMode im, Format f, InstrItinClass itin,
           string opc, string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, im, f, VFPDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p));
  let AsmString = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [HasVFP2];
}

// Special cases
class VFPXI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
            IndexMode im, Format f, InstrItinClass itin,
            string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, im, f, VFPDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [HasVFP2];
}

class VFPAI<dag oops, dag iops, Format f, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : VFPI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, itin,
         opc, asm, "", pattern>;

// ARM VFP addrmode5 loads and stores
class ADI5<bits<4> opcod1, bits<2> opcod2, dag oops, dag iops,
           InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : VFPI<oops, iops, AddrMode5, Size4Bytes, IndexModeNone,
         VFPLdStFrm, itin, opc, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-24} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{11-8}  = 0b1011;

  // 64-bit loads & stores operate on both NEON and VFP pipelines.
  let D = VFPNeonDomain;
}

class ASI5<bits<4> opcod1, bits<2> opcod2, dag oops, dag iops,
           InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : VFPI<oops, iops, AddrMode5, Size4Bytes, IndexModeNone,
         VFPLdStFrm, itin, opc, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-24} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{11-8}  = 0b1010;
}

// Load / store multiple
class AXDI4<dag oops, dag iops, IndexMode im, InstrItinClass itin,
            string asm, string cstr, list<dag> pattern>
  : VFPXI<oops, iops, AddrMode4, Size4Bytes, im,
          VFPLdStMulFrm, itin, asm, cstr, pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-25} = 0b110;
  let Inst{11-8}  = 0b1011;

  // 64-bit loads & stores operate on both NEON and VFP pipelines.
  let D = VFPNeonDomain;
}

class AXSI4<dag oops, dag iops, IndexMode im, InstrItinClass itin,
            string asm, string cstr, list<dag> pattern>
  : VFPXI<oops, iops, AddrMode4, Size4Bytes, im,
          VFPLdStMulFrm, itin, asm, cstr, pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-25} = 0b110;
  let Inst{11-8}  = 0b1010;
}

// Double precision, unary
class ADuI<bits<5> opcod1, bits<2> opcod2, bits<4> opcod3, bits<2> opcod4,
           bit opcod5, dag oops, dag iops, InstrItinClass itin, string opc,
           string asm, list<dag> pattern>
  : VFPAI<oops, iops, VFPUnaryFrm, itin, opc, asm, pattern> {
  let Inst{27-23} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{19-16} = opcod3;
  let Inst{11-8}  = 0b1011;
  let Inst{7-6}   = opcod4;
  let Inst{4}     = opcod5;
}

// Double precision, binary
class ADbI<bits<5> opcod1, bits<2> opcod2, bit op6, bit op4, dag oops,
           dag iops, InstrItinClass itin, string opc, string asm,
           list<dag> pattern>
  : VFPAI<oops, iops, VFPBinaryFrm, itin, opc, asm, pattern> {
  let Inst{27-23} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{11-8}  = 0b1011;
  let Inst{6}     = op6;
  let Inst{4}     = op4;
}

// Double precision, binary, VML[AS] (for additional predicate)
class ADbI_vmlX<bits<5> opcod1, bits<2> opcod2, bit op6, bit op4, dag oops,
           dag iops, InstrItinClass itin, string opc, string asm,
           list<dag> pattern>
  : VFPAI<oops, iops, VFPBinaryFrm, itin, opc, asm, pattern> {
  let Inst{27-23} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{11-8}  = 0b1011;
  let Inst{6}     = op6;
  let Inst{4}     = op4;
  list<Predicate> Predicates = [HasVFP2, UseVMLx];
}

// Single precision, unary
class ASuI<bits<5> opcod1, bits<2> opcod2, bits<4> opcod3, bits<2> opcod4,
           bit opcod5, dag oops, dag iops, InstrItinClass itin, string opc,
           string asm, list<dag> pattern>
  : VFPAI<oops, iops, VFPUnaryFrm, itin, opc, asm, pattern> {
  let Inst{27-23} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{19-16} = opcod3;
  let Inst{11-8}  = 0b1010;
  let Inst{7-6}   = opcod4;
  let Inst{4}     = opcod5;
}

// Single precision unary, if no NEON
// Same as ASuI except not available if NEON is enabled
class ASuIn<bits<5> opcod1, bits<2> opcod2, bits<4> opcod3, bits<2> opcod4,
            bit opcod5, dag oops, dag iops, InstrItinClass itin, string opc,
            string asm, list<dag> pattern>
  : ASuI<opcod1, opcod2, opcod3, opcod4, opcod5, oops, iops, itin, opc, asm,
         pattern> {
  list<Predicate> Predicates = [HasVFP2,DontUseNEONForFP];
}

// Single precision, binary
class ASbI<bits<5> opcod1, bits<2> opcod2, bit op6, bit op4, dag oops, dag iops,
           InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : VFPAI<oops, iops, VFPBinaryFrm, itin, opc, asm, pattern> {
  let Inst{27-23} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{11-8}  = 0b1010;
  let Inst{6}     = op6;
  let Inst{4}     = op4;
}

// Single precision binary, if no NEON
// Same as ASbI except not available if NEON is enabled
class ASbIn<bits<5> opcod1, bits<2> opcod2, bit op6, bit op4, dag oops,
            dag iops, InstrItinClass itin, string opc, string asm,
            list<dag> pattern>
  : ASbI<opcod1, opcod2, op6, op4, oops, iops, itin, opc, asm, pattern> {
  list<Predicate> Predicates = [HasVFP2,DontUseNEONForFP];
}

// VFP conversion instructions
class AVConv1I<bits<5> opcod1, bits<2> opcod2, bits<4> opcod3, bits<4> opcod4,
               dag oops, dag iops, InstrItinClass itin, string opc, string asm,
               list<dag> pattern>
  : VFPAI<oops, iops, VFPConv1Frm, itin, opc, asm, pattern> {
  let Inst{27-23} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{19-16} = opcod3;
  let Inst{11-8}  = opcod4;
  let Inst{6}     = 1;
  let Inst{4}     = 0;
}

// VFP conversion between floating-point and fixed-point
class AVConv1XI<bits<5> op1, bits<2> op2, bits<4> op3, bits<4> op4, bit op5,
                dag oops, dag iops, InstrItinClass itin, string opc, string asm,
                list<dag> pattern>
  : AVConv1I<op1, op2, op3, op4, oops, iops, itin, opc, asm, pattern> {
  // size (fixed-point number): sx == 0 ? 16 : 32
  let Inst{7} = op5; // sx
}

// VFP conversion instructions, if no NEON
class AVConv1In<bits<5> opcod1, bits<2> opcod2, bits<4> opcod3, bits<4> opcod4,
                dag oops, dag iops, InstrItinClass itin,
                string opc, string asm, list<dag> pattern>
  : AVConv1I<opcod1, opcod2, opcod3, opcod4, oops, iops, itin, opc, asm,
             pattern> {
  list<Predicate> Predicates = [HasVFP2,DontUseNEONForFP];
}

class AVConvXI<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops, Format f,
               InstrItinClass itin,
               string opc, string asm, list<dag> pattern>
  : VFPAI<oops, iops, f, itin, opc, asm, pattern> {
  let Inst{27-20} = opcod1;
  let Inst{11-8}  = opcod2;
  let Inst{4}     = 1;
}

class AVConv2I<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops,
               InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : AVConvXI<opcod1, opcod2, oops, iops, VFPConv2Frm, itin, opc, asm, pattern>;

class AVConv3I<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops,
               InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : AVConvXI<opcod1, opcod2, oops, iops, VFPConv3Frm, itin, opc, asm, pattern>;

class AVConv4I<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops,
               InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : AVConvXI<opcod1, opcod2, oops, iops, VFPConv4Frm, itin, opc, asm, pattern>;

class AVConv5I<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops,
               InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : AVConvXI<opcod1, opcod2, oops, iops, VFPConv5Frm, itin, opc, asm, pattern>;

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM NEON Instruction templates.
//

class NeonI<dag oops, dag iops, AddrMode am, IndexMode im, Format f,
            InstrItinClass itin, string opc, string dt, string asm, string cstr,
            list<dag> pattern>
  : InstARM<am, Size4Bytes, im, f, NeonDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p));
  let AsmString = !strconcat(
                     !strconcat(!strconcat(opc, "${p}"), !strconcat(".", dt)),
                     !strconcat("\t", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [HasNEON];
}

// Same as NeonI except it does not have a "data type" specifier.
class NeonXI<dag oops, dag iops, AddrMode am, IndexMode im, Format f,
             InstrItinClass itin, string opc, string asm, string cstr,
             list<dag> pattern>
  : InstARM<am, Size4Bytes, im, f, NeonDomain, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p));
  let AsmString = !strconcat(!strconcat(opc, "${p}"), !strconcat("\t", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [HasNEON];
}

class NLdSt<bit op23, bits<2> op21_20, bits<4> op11_8, bits<4> op7_4,
            dag oops, dag iops, InstrItinClass itin,
            string opc, string dt, string asm, string cstr, list<dag> pattern>
  : NeonI<oops, iops, AddrMode6, IndexModeNone, NLdStFrm, itin, opc, dt, asm,
          cstr, pattern> {
  let Inst{31-24} = 0b11110100;
  let Inst{23}    = op23;
  let Inst{21-20} = op21_20;
  let Inst{11-8}  = op11_8;
  let Inst{7-4}   = op7_4;
}

class PseudoNLdSt<dag oops, dag iops, InstrItinClass itin, string cstr>
  : InstARM<AddrMode6, Size4Bytes, IndexModeNone, Pseudo, NeonDomain, cstr,
            itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p));
  list<Predicate> Predicates = [HasNEON];
}

class NDataI<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string dt, string asm, string cstr, list<dag> pattern>
  : NeonI<oops, iops, AddrModeNone, IndexModeNone, f, itin, opc, dt, asm, cstr,
          pattern> {
  let Inst{31-25} = 0b1111001;
}

class NDataXI<dag oops, dag iops, Format f, InstrItinClass itin,
              string opc, string asm, string cstr, list<dag> pattern>
  : NeonXI<oops, iops, AddrModeNone, IndexModeNone, f, itin, opc, asm,
           cstr, pattern> {
  let Inst{31-25} = 0b1111001;
}

// NEON "one register and a modified immediate" format.
class N1ModImm<bit op23, bits<3> op21_19, bits<4> op11_8, bit op7, bit op6,
               bit op5, bit op4,
               dag oops, dag iops, InstrItinClass itin,
               string opc, string dt, string asm, string cstr,
               list<dag> pattern>
  : NDataI<oops, iops, N1RegModImmFrm, itin, opc, dt, asm, cstr, pattern> {
  let Inst{23}    = op23;
  let Inst{21-19} = op21_19;
  let Inst{11-8}  = op11_8;
  let Inst{7}     = op7;
  let Inst{6}     = op6;
  let Inst{5}     = op5;
  let Inst{4}     = op4;
}

// NEON 2 vector register format.
class N2V<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18, bits<2> op17_16,
          bits<5> op11_7, bit op6, bit op4,
          dag oops, dag iops, InstrItinClass itin,
          string opc, string dt, string asm, string cstr, list<dag> pattern>
  : NDataI<oops, iops, N2RegFrm, itin, opc, dt, asm, cstr, pattern> {
  let Inst{24-23} = op24_23;
  let Inst{21-20} = op21_20;
  let Inst{19-18} = op19_18;
  let Inst{17-16} = op17_16;
  let Inst{11-7}  = op11_7;
  let Inst{6}     = op6;
  let Inst{4}     = op4;
}

// Same as N2V except it doesn't have a datatype suffix.
class N2VX<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18, bits<2> op17_16,
           bits<5> op11_7, bit op6, bit op4,
           dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, string cstr, list<dag> pattern>
  : NDataXI<oops, iops, N2RegFrm, itin, opc, asm, cstr, pattern> {
  let Inst{24-23} = op24_23;
  let Inst{21-20} = op21_20;
  let Inst{19-18} = op19_18;
  let Inst{17-16} = op17_16;
  let Inst{11-7}  = op11_7;
  let Inst{6}     = op6;
  let Inst{4}     = op4;
}

// NEON 2 vector register with immediate.
class N2VImm<bit op24, bit op23, bits<4> op11_8, bit op7, bit op6, bit op4,
             dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string dt, string asm, string cstr, list<dag> pattern>
  : NDataI<oops, iops, f, itin, opc, dt, asm, cstr, pattern> {
  let Inst{24}   = op24;
  let Inst{23}   = op23;
  let Inst{11-8} = op11_8;
  let Inst{7}    = op7;
  let Inst{6}    = op6;
  let Inst{4}    = op4;
}

// NEON 3 vector register format.
class N3V<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op6, bit op4,
          dag oops, dag iops, Format f, InstrItinClass itin,
          string opc, string dt, string asm, string cstr, list<dag> pattern>
  : NDataI<oops, iops, f, itin, opc, dt, asm, cstr, pattern> {
  let Inst{24}    = op24;
  let Inst{23}    = op23;
  let Inst{21-20} = op21_20;
  let Inst{11-8}  = op11_8;
  let Inst{6}     = op6;
  let Inst{4}     = op4;
}

// Same as N3V except it doesn't have a data type suffix.
class N3VX<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op6,
           bit op4,
           dag oops, dag iops, Format f, InstrItinClass itin,
           string opc, string asm, string cstr, list<dag> pattern>
  : NDataXI<oops, iops, f, itin, opc, asm, cstr, pattern> {
  let Inst{24}    = op24;
  let Inst{23}    = op23;
  let Inst{21-20} = op21_20;
  let Inst{11-8}  = op11_8;
  let Inst{6}     = op6;
  let Inst{4}     = op4;
}

// NEON VMOVs between scalar and core registers.
class NVLaneOp<bits<8> opcod1, bits<4> opcod2, bits<2> opcod3,
               dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string dt, string asm, list<dag> pattern>
  : InstARM<AddrModeNone, Size4Bytes, IndexModeNone, f, GenericDomain,
            "", itin> {
  let Inst{27-20} = opcod1;
  let Inst{11-8}  = opcod2;
  let Inst{6-5}   = opcod3;
  let Inst{4}     = 1;

  let OutOperandList = oops;
  let InOperandList = !con(iops, (ins pred:$p));
  let AsmString = !strconcat(
                     !strconcat(!strconcat(opc, "${p}"), !strconcat(".", dt)),
                     !strconcat("\t", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [HasNEON];
}
class NVGetLane<bits<8> opcod1, bits<4> opcod2, bits<2> opcod3,
                dag oops, dag iops, InstrItinClass itin,
                string opc, string dt, string asm, list<dag> pattern>
  : NVLaneOp<opcod1, opcod2, opcod3, oops, iops, NGetLnFrm, itin,
             opc, dt, asm, pattern>;
class NVSetLane<bits<8> opcod1, bits<4> opcod2, bits<2> opcod3,
                dag oops, dag iops, InstrItinClass itin,
                string opc, string dt, string asm, list<dag> pattern>
  : NVLaneOp<opcod1, opcod2, opcod3, oops, iops, NSetLnFrm, itin,
             opc, dt, asm, pattern>;
class NVDup<bits<8> opcod1, bits<4> opcod2, bits<2> opcod3,
            dag oops, dag iops, InstrItinClass itin,
            string opc, string dt, string asm, list<dag> pattern>
  : NVLaneOp<opcod1, opcod2, opcod3, oops, iops, NDupFrm, itin,
             opc, dt, asm, pattern>;

// Vector Duplicate Lane (from scalar to all elements)
class NVDupLane<bits<4> op19_16, bit op6, dag oops, dag iops,
                InstrItinClass itin, string opc, string dt, string asm,
                list<dag> pattern>
  : NDataI<oops, iops, NVDupLnFrm, itin, opc, dt, asm, "", pattern> {
  let Inst{24-23} = 0b11;
  let Inst{21-20} = 0b11;
  let Inst{19-16} = op19_16;
  let Inst{11-7}  = 0b11000;
  let Inst{6}     = op6;
  let Inst{4}     = 0;
}

// NEONFPPat - Same as Pat<>, but requires that the compiler be using NEON
// for single-precision FP.
class NEONFPPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [HasNEON,UseNEONForFP];
}
