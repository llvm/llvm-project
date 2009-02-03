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

// Misc flag for data processing instructions that indicates whether
// the instruction has a Rn register operand.
class UnaryDP  { bit isUnaryDataProc = 1; }

//===----------------------------------------------------------------------===//

// ARM Instruction templates.
//

class InstARM<AddrMode am, SizeFlagVal sz, IndexMode im,
              Format f, string cstr>
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
  
  let Constraints = cstr;
}

class PseudoInst<dag oops, dag iops, string asm, list<dag> pattern>
  : InstARM<AddrModeNone, SizeSpecial, IndexModeNone, Pseudo, ""> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
  let Pattern = pattern;
}

// Almost all ARM instructions are predicable.
class I<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
        IndexMode im, Format f, string opc, string asm, string cstr,
        list<dag> pattern>
  : InstARM<am, sz, im, f, cstr> {
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
         IndexMode im, Format f, string opc, string asm, string cstr,
         list<dag> pattern>
  : InstARM<am, sz, im, f, cstr> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p, cc_out:$s));
  let AsmString   = !strconcat(opc, !strconcat("${p}${s}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

// Special cases
class XI<dag oops, dag iops, AddrMode am, SizeFlagVal sz,
         IndexMode im, Format f, string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, im, f, cstr> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

class AI<dag oops, dag iops, Format f, string opc,
         string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern>;
class AsI<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : sI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, opc,
       asm, "", pattern>;
class AXI<dag oops, dag iops, Format f, string asm,
          list<dag> pattern>
  : XI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, asm,
       "", pattern>;

// Ctrl flow instructions
class ABI<bits<4> opcod, dag oops, dag iops, string opc,
         string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, BrFrm, opc,
      asm, "", pattern> {
  let Inst{27-24} = opcod;
}
class ABXI<bits<4> opcod, dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, BrFrm, asm,
       "", pattern> {
  let Inst{27-24} = opcod;
}
class ABXIx2<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, Size8Bytes, IndexModeNone, BrMiscFrm, asm,
       "", pattern>;

// BR_JT instructions
class JTI<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, SizeSpecial, IndexModeNone, BrMiscFrm,
       asm, "", pattern>;

// addrmode1 instructions
class AI1<bits<4> opcod, dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{24-21} = opcod;
  let Inst{27-26} = {0,0};
}
class AsI1<bits<4> opcod, dag oops, dag iops, Format f, string opc,
           string asm, list<dag> pattern>
  : sI<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, opc,
       asm, "", pattern> {
  let Inst{24-21} = opcod;
  let Inst{27-26} = {0,0};
}
class AXI1<bits<4> opcod, dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, asm,
       "", pattern> {
  let Inst{24-21} = opcod;
  let Inst{27-26} = {0,0};
}
class AI1x2<dag oops, dag iops, Format f, string opc,
            string asm, list<dag> pattern>
  : I<oops, iops, AddrMode1, Size8Bytes, IndexModeNone, f, opc,
      asm, "", pattern>;


// addrmode2 loads and stores
class AI2<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{27-26} = {0,1};
}

// loads
class AI2ldw<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AXI2ldw<dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f,
       asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AI2ldb<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AXI2ldb<dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f,
       asm, "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}

// stores
class AI2stw<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AXI2stw<dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f,
       asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AI2stb<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AXI2stb<dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f,
       asm, "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}

// Pre-indexed loads
class AI2ldwpr<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AI2ldbpr<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}

// Pre-indexed stores
class AI2stwpr<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}
class AI2stbpr<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 1; // P bit
  let Inst{27-26} = {0,1};
}

// Post-indexed loads
class AI2ldwpo<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = {0,1};
}
class AI2ldbpo<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = {0,1};
}

// Post-indexed stores
class AI2stwpo<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 0; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = {0,1};
}
class AI2stbpo<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{22}    = 1; // B bit
  let Inst{24}    = 0; // P bit
  let Inst{27-26} = {0,1};
}

// addrmode3 instructions
class AI3<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern>;
class AXI3<dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, asm,
       "", pattern>;

// loads
class AI3ldh<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AXI3ldh<dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f,
       asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3ldsh<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AXI3ldsh<dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f,
       asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3ldsb<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AXI3ldsb<dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f,
       asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3ldd<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}

// stores
class AI3sth<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AXI3sth<dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f,
       asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3std<dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 0; // W bit
  let Inst{24}    = 1; // P bit
}

// Pre-indexed loads
class AI3ldhpr<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3ldshpr<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
}
class AI3ldsbpr<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
}

// Pre-indexed stores
class AI3sthpr<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 1; // P bit
}

// Post-indexed loads
class AI3ldhpo<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 0; // P bit
}
class AI3ldshpo<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 0; // P bit
}
class AI3ldsbpo<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 0; // H bit
  let Inst{6}     = 1; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 1; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 0; // P bit
}

// Post-indexed stores
class AI3sthpo<dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern> {
  let Inst{4}     = 1;
  let Inst{5}     = 1; // H bit
  let Inst{6}     = 0; // S bit
  let Inst{7}     = 1;
  let Inst{20}    = 0; // L bit
  let Inst{21}    = 1; // W bit
  let Inst{24}    = 0; // P bit
}


// addrmode4 instructions
class AXI4ld<dag oops, dag iops, Format f, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode4, Size4Bytes, IndexModeNone, f, asm,
       "", pattern> {
  let Inst{20}    = 1; // L bit
  let Inst{22}    = 0; // S bit
  let Inst{27-25} = 0b100;
}
class AXI4st<dag oops, dag iops, Format f, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode4, Size4Bytes, IndexModeNone, f, asm,
       "", pattern> {
  let Inst{20}    = 0; // L bit
  let Inst{22}    = 0; // S bit
  let Inst{27-25} = 0b100;
}

// Unsigned multiply, multiply-accumulate instructions.
class AMul1I<bits<7> opcod, dag oops, dag iops, string opc,
         string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, MulFrm, opc,
      asm, "", pattern> {
  let Inst{7-4}   = 0b1001;
  let Inst{20}    = 0; // S bit
  let Inst{27-21} = opcod;
}
class AsMul1I<bits<7> opcod, dag oops, dag iops, string opc,
          string asm, list<dag> pattern>
  : sI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, MulFrm, opc,
       asm, "", pattern> {
  let Inst{7-4}   = 0b1001;
  let Inst{27-21} = opcod;
}

// Most significant word multiply
class AMul2I<bits<7> opcod, dag oops, dag iops, string opc,
         string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, MulFrm, opc,
      asm, "", pattern> {
  let Inst{7-4}   = 0b1001;
  let Inst{20}    = 1;
  let Inst{27-21} = opcod;
}

// SMUL<x><y> / SMULW<y> / SMLA<x><y> / SMLAW<x><y>
class AMulxyI<bits<7> opcod, dag oops, dag iops, string opc,
         string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, MulFrm, opc,
      asm, "", pattern> {
  let Inst{4}     = 0;
  let Inst{7}     = 1;
  let Inst{20}    = 0;
  let Inst{27-21} = opcod;
}

// Extend instructions.
class AExtI<bits<8> opcod, dag oops, dag iops, string opc,
            string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, ExtFrm, opc,
      asm, "", pattern> {
  let Inst{7-4}   = 0b0111;
  let Inst{27-20} = opcod;
}

// Misc Arithmetic instructions.
class AMiscA1I<bits<8> opcod, dag oops, dag iops, string opc,
               string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, ArithMiscFrm, opc,
      asm, "", pattern> {
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

class ThumbI<dag outs, dag ins, AddrMode am, SizeFlagVal sz,
             string asm, string cstr, list<dag> pattern>
  : InstARM<am, sz, IndexModeNone, ThumbFrm, cstr> {
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString   = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb];
}

class TI<dag outs, dag ins, string asm, list<dag> pattern>
  : ThumbI<outs, ins, AddrModeNone, Size2Bytes, asm, "", pattern>;
class TI1<dag outs, dag ins, string asm, list<dag> pattern>
  : ThumbI<outs, ins, AddrModeT1, Size2Bytes, asm, "", pattern>;
class TI2<dag outs, dag ins, string asm, list<dag> pattern>
  : ThumbI<outs, ins, AddrModeT2, Size2Bytes, asm, "", pattern>;
class TI4<dag outs, dag ins, string asm, list<dag> pattern>
  : ThumbI<outs, ins, AddrModeT4, Size2Bytes, asm, "", pattern>;
class TIs<dag outs, dag ins, string asm, list<dag> pattern>
  : ThumbI<outs, ins, AddrModeTs, Size2Bytes, asm, "", pattern>;

// Two-address instructions
class TIt<dag outs, dag ins, string asm, list<dag> pattern>
  : ThumbI<outs, ins, AddrModeNone, Size2Bytes, asm, "$lhs = $dst", pattern>;

// BL, BLX(1) are translated by assembler into two instructions
class TIx2<dag outs, dag ins, string asm, list<dag> pattern>
  : ThumbI<outs, ins, AddrModeNone, Size4Bytes, asm, "", pattern>;

// BR_JT instructions
class TJTI<dag outs, dag ins, string asm, list<dag> pattern>
  : ThumbI<outs, ins, AddrModeNone, SizeSpecial, asm, "", pattern>;


//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM VFP Instruction templates.
//

// ARM VFP addrmode5 loads and stores
class ADI5<bits<4> opcod1, bits<2> opcod2, dag oops, dag iops,
           string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode5, Size4Bytes, IndexModeNone,
      VFPLdStFrm, opc, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-24} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{11-8}  = 0b1011;
}

class ASI5<bits<4> opcod1, bits<2> opcod2, dag oops, dag iops,
           string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode5, Size4Bytes, IndexModeNone,
      VFPLdStFrm, opc, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-24} = opcod1;
  let Inst{21-20} = opcod2;
  let Inst{11-8}  = 0b1010;
}

// Load / store multiple
class AXSI5<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode5, Size4Bytes, IndexModeNone,
       VFPLdStMulFrm, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-25} = 0b110;
  let Inst{11-8}  = 0b1011;
}

class AXDI5<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode5, Size4Bytes, IndexModeNone,
       VFPLdStMulFrm, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
  let Inst{27-25} = 0b110;
  let Inst{11-8}  = 0b1010;
}


// Double precision, unary
class ADuI<bits<8> opcod1, bits<4> opcod2, bits<4> opcod3, dag oops, dag iops,
           string opc, string asm, list<dag> pattern>
  : AI<oops, iops, VFPUnaryFrm, opc, asm, pattern> {
  let Inst{27-20} = opcod1;
  let Inst{19-16} = opcod2;
  let Inst{11-8}  = 0b1011;
  let Inst{7-4}   = opcod3;
}

// Double precision, binary
class ADbI<bits<8> opcod, dag oops, dag iops, string opc,
           string asm, list<dag> pattern>
  : AI<oops, iops, VFPBinaryFrm, opc, asm, pattern> {
  let Inst{27-20} = opcod;
  let Inst{11-8}  = 0b1011;
}

// Single precision, unary
class ASuI<bits<8> opcod1, bits<4> opcod2, bits<4> opcod3, dag oops, dag iops,
           string opc, string asm, list<dag> pattern>
  : AI<oops, iops, VFPUnaryFrm, opc, asm, pattern> {
  // Bits 22 (D bit) and 5 (M bit) will be changed during instruction encoding.
  let Inst{27-20} = opcod1;
  let Inst{19-16} = opcod2;
  let Inst{11-8}  = 0b1010;
  let Inst{7-4}   = opcod3;
}

// Single precision, binary
class ASbI<bits<8> opcod, dag oops, dag iops, string opc,
           string asm, list<dag> pattern>
  : AI<oops, iops, VFPBinaryFrm, opc, asm, pattern> {
  // Bit 22 (D bit) can be changed during instruction encoding.
  let Inst{27-20} = opcod;
  let Inst{11-8}  = 0b1010;
}

// VFP conversion instructions
class AVConv1I<bits<8> opcod1, bits<4> opcod2, bits<4> opcod3,
               dag oops, dag iops, string opc, string asm, list<dag> pattern>
  : AI<oops, iops, VFPConv1Frm, opc, asm, pattern> {
  let Inst{27-20} = opcod1;
  let Inst{19-16} = opcod2;
  let Inst{11-8}  = opcod3;
  let Inst{6}     = 1;
}

class AVConvXI<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops, Format f,
             string opc, string asm, list<dag> pattern>
  : AI<oops, iops, f, opc, asm, pattern> {
  let Inst{27-20} = opcod1;
  let Inst{11-8}  = opcod2;
  let Inst{4}     = 1;
}

class AVConv2I<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops, string opc,
              string asm, list<dag> pattern>
  : AVConvXI<opcod1, opcod2, oops, iops, VFPConv2Frm, opc, asm, pattern>;

class AVConv3I<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops, string opc,
              string asm, list<dag> pattern>
  : AVConvXI<opcod1, opcod2, oops, iops, VFPConv3Frm, opc, asm, pattern>;

class AVConv4I<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops, string opc,
              string asm, list<dag> pattern>
  : AVConvXI<opcod1, opcod2, oops, iops, VFPConv4Frm, opc, asm, pattern>;

class AVConv5I<bits<8> opcod1, bits<4> opcod2, dag oops, dag iops, string opc,
              string asm, list<dag> pattern>
  : AVConvXI<opcod1, opcod2, oops, iops, VFPConv5Frm, opc, asm, pattern>;

//===----------------------------------------------------------------------===//


// ThumbPat - Same as Pat<>, but requires that the compiler be in Thumb mode.
class ThumbPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsThumb];
}

class ThumbV5Pat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsThumb, HasV5T];
}
