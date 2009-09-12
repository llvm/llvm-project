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
class Format<bits<5> val> {
  bits<5> Value = val;
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

def ArithMiscFrm  : Format<11>;
def ExtFrm        : Format<12>;

def VFPUnaryFrm   : Format<13>;
def VFPBinaryFrm  : Format<14>;
def VFPConv1Frm   : Format<15>;
def VFPConv2Frm   : Format<16>;
def VFPConv3Frm   : Format<17>;
def VFPConv4Frm   : Format<18>;
def VFPConv5Frm   : Format<19>;
def VFPLdStFrm    : Format<20>;
def VFPLdStMulFrm : Format<21>;
def VFPMiscFrm    : Format<22>;

def ThumbFrm      : Format<23>;

def NEONFrm       : Format<24>;
def NEONGetLnFrm  : Format<25>;
def NEONSetLnFrm  : Format<26>;
def NEONDupFrm    : Format<27>;

// Misc flags.

// the instruction has a Rn register operand.
// UnaryDP - Indicates this is a unary data processing instruction, i.e.
// it doesn't have a Rn operand.
class UnaryDP    { bit isUnaryDataProc = 1; }

// Xform16Bit - Indicates this Thumb2 instruction may be transformed into
// a 16-bit Thumb instruction if certain conditions are met.
class Xform16Bit { bit canXformTo16Bit = 1; }

//===----------------------------------------------------------------------===//
// ARM Instruction flags.  These need to match ARMInstrInfo.h.
//

// Addressing mode.
class AddrMode<bits<4> val> {
  bits<4> Value = val;
}
def AddrModeNone  : AddrMode<0>;
def AddrMode1     : AddrMode<1>;
def AddrMode2     : AddrMode<2>;
def AddrMode3     : AddrMode<3>;
def AddrMode4     : AddrMode<4>;
def AddrMode5     : AddrMode<5>;
def AddrMode6     : AddrMode<6>;
def AddrModeT1_1  : AddrMode<7>;
def AddrModeT1_2  : AddrMode<8>;
def AddrModeT1_4  : AddrMode<9>;
def AddrModeT1_s  : AddrMode<10>;
def AddrModeT2_i12: AddrMode<11>;
def AddrModeT2_i8 : AddrMode<12>;
def AddrModeT2_so : AddrMode<13>;
def AddrModeT2_pc : AddrMode<14>;
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

//===----------------------------------------------------------------------===//

// ARM special operands.
//

// ARM Predicate operand. Default to 14 = always (AL). Second part is CC
// register whose default is 0 (no register).
def pred : PredicateOperand<OtherVT, (ops i32imm, CCR),
                                     (ops (i32 14), (i32 zero_reg))> {
  let PrintMethod = "printPredicateOperand";
}

// Conditional code result for instructions whose 's' bit is set, e.g. subs.
def cc_out : OptionalDefOperand<OtherVT, (ops CCR), (ops (i32 zero_reg))> {
  let PrintMethod = "printSBitModifierOperand";
}

// Same as cc_out except it defaults to setting CPSR.
def s_cc_out : OptionalDefOperand<OtherVT, (ops CCR), (ops (i32 CPSR))> {
  let PrintMethod = "printSBitModifierOperand";
}

//===----------------------------------------------------------------------===//

// ARM Instruction templates.
//

class InstARM<AddrMode am, SizeFlagVal sz, IndexMode im,
              Format f, string cstr, InstrItinClass itin>
  : Instruction {
  field bits<32> Inst;

  let Namespace = "ARM";

  // TSFlagsFields
  AddrMode AM = am;
  bits<4> AddrModeBits = AM.Value;
  
  SizeFlagVal SZ = sz;
  bits<3> SizeFlag = SZ.Value;

  IndexMode IM = im;
  bits<2> IndexModeBits = IM.Value;
  
  Format F = f;
  bits<5> Form = F.Value;

  //
  // Attributes specific to ARM instructions...
  //
  bit isUnaryDataProc = 0;
  bit canXformTo16Bit = 0;
  
  let Constraints = cstr;
  let Itinerary = itin;
}

class PseudoInst<dag oops, dag iops, InstrItinClass itin, 
                 string asm, list<dag> pattern>
  : InstARM<AddrModeNone, SizeSpecial, IndexModeNone, Pseudo, "", itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
  let Pattern = pattern;
}

// Almost all ARM instructions are predicable.
class I<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
        IndexMode im, Format f, InstrItinClass itin, 
        string opc, string asm, string cstr,
        list<dag> pattern>
  : InstARM<am, sz, im, f, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p));
  let AsmString   = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

// Same as I except it can optionally modify CPSR. Note it's modeled as
// an input operand since by default it's a zero register. It will
// become an implicit def once it's "flipped".
class sI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
         IndexMode im, Format f, InstrItinClass itin,
         string opc, string asm, string cstr,
         list<dag> pattern>
  : InstARM<am, sz, im, f, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p, cc_out:$s));
  let AsmString   = !strconcat(opc, !strconcat("${p}${s}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

// Special cases
class XI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
         IndexMode im, Format f, InstrItinClass itin,
         string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, im, f, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
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
  : XI<oops, iops, AddrModeNone, Size8Bytes, IndexModeNone, BrMiscFrm, itin,
       asm, "", pattern>;

// BR_JT instructions
class JTI<dag oops, dag iops, InstrItinClass itin,
          string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, SizeSpecial, IndexModeNone, BrMiscFrm, itin,
       asm, "", pattern>;

// addrmode1 instructions
class AI1<bits<4> opcod, dag oops, dag iops, Format f, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{24-21} = opcod;
  let Inst{27-26} = {0,0};
}
class AsI1<bits<4> opcod, dag oops, dag iops, Format f, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : sI<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, itin,
       opc, asm, "", pattern> {
  let Inst{24-21} = opcod;
  let Inst{27-26} = {0,0};
}
class AXI1<bits<4> opcod, dag oops, dag iops, Format f, InstrItinClass itin,
           string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{24-21} = opcod;
  let Inst{27-26} = {0,0};
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
  let Inst{27-26} = {0,1};
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
  let Inst{27-26} = {0,1};
}
class AXI2ldw<dag oops, dag iops, Format f, InstrItinClass itin, 
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AI2ldb<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AXI2ldb<dag oops, dag iops, Format f, InstrItinClass itin, 
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
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
  let Inst{27-26} = {0,1};
}
class AXI2stw<dag oops, dag iops, Format f, InstrItinClass itin,
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AI2stb<dag oops, dag iops, Format f, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
      opc, asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AXI2stb<dag oops, dag iops, Format f, InstrItinClass itin,
              string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
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
  let Inst{27-26} = {0,1};
}
class AI2ldbpr<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
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
  let Inst{27-26} = {0,1};
}
class AI2stbpr<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, itin,
      opc, asm, cstr, pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
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
  let Inst{27-26} = {0,1};
}
class AI2ldbpo<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = {0,1};
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
  let Inst{27-26} = {0,1};
}
class AI2stbpo<dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, itin,
      opc, asm, cstr,pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = {0,1};
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
  let Inst{21}    = 1; // W bit
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
  let Inst{21}    = 1; // W bit
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
  let Inst{21}    = 1; // W bit
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
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 0; // P bit
  let Inst{27-25} = 0b000;
}


// addrmode4 instructions
class AXI4ld<dag oops, dag iops, Format f, InstrItinClass itin,
             string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode4, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{22}    = 0; // S bit
  let Inst{27-25} = 0b100;
}
class AXI4st<dag oops, dag iops, Format f, InstrItinClass itin,
             string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode4, Size4Bytes, IndexModeNone, f, itin,
       asm, "", pattern> {
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
  : InstARM<am, sz, IndexModeNone, ThumbFrm, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb];
}

class TI<dag oops, dag iops, InstrItinClass itin, string asm, list<dag> pattern>
  : ThumbI<oops, iops, AddrModeNone, Size2Bytes, itin, asm, "", pattern>;

// Two-address instructions
class TIt<dag oops, dag iops, InstrItinClass itin, string asm, list<dag> pattern>
  : ThumbI<oops, iops, AddrModeNone, Size2Bytes, itin, asm, "$lhs = $dst", pattern>;

// tBL, tBX instructions
class TIx2<dag oops, dag iops, InstrItinClass itin, string asm, list<dag> pattern>
  : ThumbI<oops, iops, AddrModeNone, Size4Bytes, itin, asm, "", pattern>;

// BR_JT instructions
class TJTI<dag oops, dag iops, InstrItinClass itin, string asm, list<dag> pattern>
  : ThumbI<oops, iops, AddrModeNone, SizeSpecial, itin, asm, "", pattern>;

// Thumb1 only
class Thumb1I<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
              InstrItinClass itin, string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
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
           string asm, list<dag> pattern>
  : Thumb1I<oops, iops, AddrModeNone, Size2Bytes, itin, 
            asm, "$lhs = $dst", pattern>;

// Thumb1 instruction that can either be predicated or set CPSR.
class Thumb1sI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
               InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, cstr, itin> {
  let OutOperandList = !con(oops, (ops s_cc_out:$s));
  let InOperandList = !con(iops, (ops pred:$p));
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
  : InstARM<am, sz, IndexModeNone, ThumbFrm, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p));
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

// Thumb2I - Thumb2 instruction. Almost all Thumb2 instructions are predicable.
class Thumb2I<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
              InstrItinClass itin,
              string opc, string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p));
  let AsmString = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb2];
}

// Same as Thumb2I except it can optionally modify CPSR. Note it's modeled as
// an input operand since by default it's a zero register. It will
// become an implicit def once it's "flipped".
// FIXME: This uses unified syntax so {s} comes before {p}. We should make it
// more consistent.
class Thumb2sI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
               InstrItinClass itin,
               string opc, string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p, cc_out:$s));
  let AsmString   = !strconcat(opc, !strconcat("${s}${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb2];
}

// Special cases
class Thumb2XI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
               InstrItinClass itin,
               string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb2];
}

class T2I<dag oops, dag iops, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeNone, Size4Bytes, itin, opc, asm, "", pattern>;
class T2Ii12<dag oops, dag iops, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_i12, Size4Bytes, itin, opc, asm, "", pattern>;
class T2Ii8<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_i8, Size4Bytes, itin, opc, asm, "", pattern>;
class T2Iso<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_so, Size4Bytes, itin, opc, asm, "", pattern>;
class T2Ipc<dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_pc, Size4Bytes, itin, opc, asm, "", pattern>;
class T2Ii8s4<dag oops, dag iops, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeT2_i8s4, Size4Bytes, itin, opc, asm, "", pattern>;

class T2sI<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : Thumb2sI<oops, iops, AddrModeNone, Size4Bytes, itin, opc, asm, "", pattern>;

class T2XI<dag oops, dag iops, InstrItinClass itin,
           string asm, list<dag> pattern>
  : Thumb2XI<oops, iops, AddrModeNone, Size4Bytes, itin, asm, "", pattern>;
class T2JTI<dag oops, dag iops, InstrItinClass itin,
            string asm, list<dag> pattern>
  : Thumb2XI<oops, iops, AddrModeNone, SizeSpecial, itin, asm, "", pattern>;

// T2Iidxldst - Thumb2 indexed load / store instructions.
class T2Iidxldst<dag oops, dag iops, AddrMode am, IndexMode im,
                 InstrItinClass itin,
                 string opc, string asm, string cstr, list<dag> pattern>
  : InstARM<am, Size4Bytes, im, ThumbFrm, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p));
  let AsmString = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb2];
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
  : InstARM<am, sz, im, f, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p));
  let AsmString   = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [HasVFP2];
}

// Special cases
class VFPXI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
            IndexMode im, Format f, InstrItinClass itin,
            string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, im, f, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
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
class AXDI5<dag oops, dag iops, InstrItinClass itin,
            string asm, list<dag> pattern>
  : VFPXI<oops, iops, AddrMode5, Size4Bytes, IndexModeNone,
       VFPLdStMulFrm, itin, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-25} = 0b110;
  let Inst{11-8}  = 0b1011;
}

class AXSI5<dag oops, dag iops, InstrItinClass itin,
            string asm, list<dag> pattern>
  : VFPXI<oops, iops, AddrMode5, Size4Bytes, IndexModeNone,
       VFPLdStMulFrm, itin, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-25} = 0b110;
  let Inst{11-8}  = 0b1010;
}

// Double precision, unary
class ADuI<bits<8> opcod1, bits<4> opcod2, bits<4> opcod3, dag oops, dag iops,
           InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : VFPAI<oops, iops, VFPUnaryFrm, itin, opc, asm, pattern> {
  let Inst{27-20} = opcod1;
  let Inst{19-16} = opcod2;
  let Inst{11-8}  = 0b1011;
  let Inst{7-4}   = opcod3;
}

// Double precision, binary
class ADbI<bits<8> opcod, dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : VFPAI<oops, iops, VFPBinaryFrm, itin, opc, asm, pattern> {
  let Inst{27-20} = opcod;
  let Inst{11-8}  = 0b1011;
}

// Single precision, unary
class ASuI<bits<8> opcod1, bits<4> opcod2, bits<4> opcod3, dag oops, dag iops,
           InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : VFPAI<oops, iops, VFPUnaryFrm, itin, opc, asm, pattern> {
  // Bits 22 (D bit) and 5 (M bit) will be changed during instruction encoding.
  let Inst{27-20} = opcod1;
  let Inst{19-16} = opcod2;
  let Inst{11-8}  = 0b1010;
  let Inst{7-4}   = opcod3;
}

// Single precision unary, if no NEON
// Same as ASuI except not available if NEON is enabled
class ASuIn<bits<8> opcod1, bits<4> opcod2, bits<4> opcod3, dag oops, dag iops,
            InstrItinClass itin,    string opc, string asm, list<dag> pattern>
  : ASuI<opcod1, opcod2, opcod2, oops, iops, itin, opc, asm, pattern> {
  list<Predicate> Predicates = [HasVFP2,DontUseNEONForFP];
}

// Single precision, binary
class ASbI<bits<8> opcod, dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : VFPAI<oops, iops, VFPBinaryFrm, itin, opc, asm, pattern> {
  // Bit 22 (D bit) can be changed during instruction encoding.
  let Inst{27-20} = opcod;
  let Inst{11-8}  = 0b1010;
}

// Single precision binary, if no NEON
// Same as ASbI except not available if NEON is enabled
class ASbIn<bits<8> opcod, dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : ASbI<opcod, oops, iops, itin, opc, asm, pattern> {
  list<Predicate> Predicates = [HasVFP2,DontUseNEONForFP];
}

// VFP conversion instructions
class AVConv1I<bits<8> opcod1, bits<4> opcod2, bits<4> opcod3,
               dag oops, dag iops, InstrItinClass itin,
               string opc, string asm, list<dag> pattern>
  : VFPAI<oops, iops, VFPConv1Frm, itin, opc, asm, pattern> {
  let Inst{27-20} = opcod1;
  let Inst{19-16} = opcod2;
  let Inst{11-8}  = opcod3;
  let Inst{6}     = 1;
}

// VFP conversion instructions, if no NEON
class AVConv1In<bits<8> opcod1, bits<4> opcod2, bits<4> opcod3,
                dag oops, dag iops, InstrItinClass itin,
                string opc, string asm, list<dag> pattern>
  : AVConv1I<opcod1, opcod2, opcod3, oops, iops, itin, opc, asm, pattern> {
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

class NeonI<dag oops, dag iops, AddrMode am, IndexMode im, InstrItinClass itin,
            string asm, string cstr, list<dag> pattern>
  : InstARM<am, Size4Bytes, im, NEONFrm, cstr, itin> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [HasNEON];
}

class NI<dag oops, dag iops, InstrItinClass itin, string asm, list<dag> pattern>
  : NeonI<oops, iops, AddrModeNone, IndexModeNone, itin, asm, "", pattern> {
}

class NI4<dag oops, dag iops, InstrItinClass itin, string asm, list<dag> pattern>
  : NeonI<oops, iops, AddrMode4, IndexModeNone, itin, asm, "", pattern> {
}

class NLdSt<dag oops, dag iops, InstrItinClass itin,
            string asm, list<dag> pattern>
  : NeonI<oops, iops, AddrMode6, IndexModeNone, itin, asm, "", pattern> {
  let Inst{31-24} = 0b11110100;
}

class NDataI<dag oops, dag iops, InstrItinClass itin,
             string asm, string cstr, list<dag> pattern>
  : NeonI<oops, iops, AddrModeNone, IndexModeNone, itin, asm, cstr, pattern> {
  let Inst{31-25} = 0b1111001;
}

// NEON "one register and a modified immediate" format.
class N1ModImm<bit op23, bits<3> op21_19, bits<4> op11_8, bit op7, bit op6,
               bit op5, bit op4,
               dag oops, dag iops, InstrItinClass itin,
               string asm, string cstr, list<dag> pattern>
  : NDataI<oops, iops, itin, asm, cstr, pattern> {
  let Inst{23} = op23;
  let Inst{21-19} = op21_19;
  let Inst{11-8} = op11_8;
  let Inst{7} = op7;
  let Inst{6} = op6;
  let Inst{5} = op5;
  let Inst{4} = op4;
}

// NEON 2 vector register format.
class N2V<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18, bits<2> op17_16,
          bits<5> op11_7, bit op6, bit op4,
          dag oops, dag iops, InstrItinClass itin,
          string asm, string cstr, list<dag> pattern>
  : NDataI<oops, iops, itin, asm, cstr, pattern> {
  let Inst{24-23} = op24_23;
  let Inst{21-20} = op21_20;
  let Inst{19-18} = op19_18;
  let Inst{17-16} = op17_16;
  let Inst{11-7} = op11_7;
  let Inst{6} = op6;
  let Inst{4} = op4;
}

// NEON 2 vector register with immediate.
class N2VImm<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
             bit op6, bit op4,
             dag oops, dag iops, InstrItinClass itin,
             string asm, string cstr, list<dag> pattern>
  : NDataI<oops, iops, itin, asm, cstr, pattern> {
  let Inst{24} = op24;
  let Inst{23} = op23;
  let Inst{21-16} = op21_16;
  let Inst{11-8} = op11_8;
  let Inst{7} = op7;
  let Inst{6} = op6;
  let Inst{4} = op4;
}

// NEON 3 vector register format.
class N3V<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op6, bit op4,
          dag oops, dag iops, InstrItinClass itin,
          string asm, string cstr, list<dag> pattern>
  : NDataI<oops, iops, itin, asm, cstr, pattern> {
  let Inst{24} = op24;
  let Inst{23} = op23;
  let Inst{21-20} = op21_20;
  let Inst{11-8} = op11_8;
  let Inst{6} = op6;
  let Inst{4} = op4;
}

// NEON VMOVs between scalar and core registers.
class NVLaneOp<bits<8> opcod1, bits<4> opcod2, bits<2> opcod3,
               dag oops, dag iops, Format f, InstrItinClass itin,
               string opc, string asm, list<dag> pattern>
  : AI<oops, iops, f, itin, opc, asm, pattern> {
  let Inst{27-20} = opcod1;
  let Inst{11-8} = opcod2;
  let Inst{6-5} = opcod3;
  let Inst{4} = 1;
  list<Predicate> Predicates = [HasNEON];
}
class NVGetLane<bits<8> opcod1, bits<4> opcod2, bits<2> opcod3,
                dag oops, dag iops, InstrItinClass itin,
                string opc, string asm, list<dag> pattern>
  : NVLaneOp<opcod1, opcod2, opcod3, oops, iops, NEONGetLnFrm, itin,
             opc, asm, pattern>;
class NVSetLane<bits<8> opcod1, bits<4> opcod2, bits<2> opcod3,
                dag oops, dag iops, InstrItinClass itin,
                string opc, string asm, list<dag> pattern>
  : NVLaneOp<opcod1, opcod2, opcod3, oops, iops, NEONSetLnFrm, itin,
             opc, asm, pattern>;
class NVDup<bits<8> opcod1, bits<4> opcod2, bits<2> opcod3,
            dag oops, dag iops, InstrItinClass itin,
            string opc, string asm, list<dag> pattern>
  : NVLaneOp<opcod1, opcod2, opcod3, oops, iops, NEONDupFrm, itin,
             opc, asm, pattern>;

// NEONFPPat - Same as Pat<>, but requires that the compiler be using NEON
// for single-precision FP.
class NEONFPPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [HasNEON,UseNEONForFP];
}
