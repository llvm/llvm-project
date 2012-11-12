//===-- X86InstrFormats.td - X86 Instruction Formats -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<6> val> {
  bits<6> Value = val;
}

def Pseudo     : Format<0>; def RawFrm     : Format<1>;
def AddRegFrm  : Format<2>; def MRMDestReg : Format<3>;
def MRMDestMem : Format<4>; def MRMSrcReg  : Format<5>;
def MRMSrcMem  : Format<6>;
def MRM0r  : Format<16>; def MRM1r  : Format<17>; def MRM2r  : Format<18>;
def MRM3r  : Format<19>; def MRM4r  : Format<20>; def MRM5r  : Format<21>;
def MRM6r  : Format<22>; def MRM7r  : Format<23>;
def MRM0m  : Format<24>; def MRM1m  : Format<25>; def MRM2m  : Format<26>;
def MRM3m  : Format<27>; def MRM4m  : Format<28>; def MRM5m  : Format<29>;
def MRM6m  : Format<30>; def MRM7m  : Format<31>;
def MRMInitReg : Format<32>;
def MRM_C1 : Format<33>;
def MRM_C2 : Format<34>;
def MRM_C3 : Format<35>;
def MRM_C4 : Format<36>;
def MRM_C8 : Format<37>;
def MRM_C9 : Format<38>;
def MRM_E8 : Format<39>;
def MRM_F0 : Format<40>;
def MRM_F8 : Format<41>;
def MRM_F9 : Format<42>;
def RawFrmImm8 : Format<43>;
def RawFrmImm16 : Format<44>;
def MRM_D0 : Format<45>;
def MRM_D1 : Format<46>;
def MRM_D4 : Format<47>;
def MRM_D5 : Format<48>;
def MRM_D8 : Format<49>;
def MRM_D9 : Format<50>;
def MRM_DA : Format<51>;
def MRM_DB : Format<52>;
def MRM_DC : Format<53>;
def MRM_DD : Format<54>;
def MRM_DE : Format<55>;
def MRM_DF : Format<56>;

// ImmType - This specifies the immediate type used by an instruction. This is
// part of the ad-hoc solution used to emit machine instruction encodings by our
// machine code emitter.
class ImmType<bits<3> val> {
  bits<3> Value = val;
}
def NoImm      : ImmType<0>;
def Imm8       : ImmType<1>;
def Imm8PCRel  : ImmType<2>;
def Imm16      : ImmType<3>;
def Imm16PCRel : ImmType<4>;
def Imm32      : ImmType<5>;
def Imm32PCRel : ImmType<6>;
def Imm64      : ImmType<7>;

// FPFormat - This specifies what form this FP instruction has.  This is used by
// the Floating-Point stackifier pass.
class FPFormat<bits<3> val> {
  bits<3> Value = val;
}
def NotFP      : FPFormat<0>;
def ZeroArgFP  : FPFormat<1>;
def OneArgFP   : FPFormat<2>;
def OneArgFPRW : FPFormat<3>;
def TwoArgFP   : FPFormat<4>;
def CompareFP  : FPFormat<5>;
def CondMovFP  : FPFormat<6>;
def SpecialFP  : FPFormat<7>;

// Class specifying the SSE execution domain, used by the SSEDomainFix pass.
// Keep in sync with tables in X86InstrInfo.cpp.
class Domain<bits<2> val> {
  bits<2> Value = val;
}
def GenericDomain   : Domain<0>;
def SSEPackedSingle : Domain<1>;
def SSEPackedDouble : Domain<2>;
def SSEPackedInt    : Domain<3>;

// Prefix byte classes which are used to indicate to the ad-hoc machine code
// emitter that various prefix bytes are required.
class OpSize { bit hasOpSizePrefix = 1; }
class AdSize { bit hasAdSizePrefix = 1; }
class REX_W  { bit hasREX_WPrefix = 1; }
class LOCK   { bit hasLockPrefix = 1; }
class SegFS  { bits<2> SegOvrBits = 1; }
class SegGS  { bits<2> SegOvrBits = 2; }
class TB     { bits<5> Prefix = 1; }
class REP    { bits<5> Prefix = 2; }
class D8     { bits<5> Prefix = 3; }
class D9     { bits<5> Prefix = 4; }
class DA     { bits<5> Prefix = 5; }
class DB     { bits<5> Prefix = 6; }
class DC     { bits<5> Prefix = 7; }
class DD     { bits<5> Prefix = 8; }
class DE     { bits<5> Prefix = 9; }
class DF     { bits<5> Prefix = 10; }
class XD     { bits<5> Prefix = 11; }
class XS     { bits<5> Prefix = 12; }
class T8     { bits<5> Prefix = 13; }
class TA     { bits<5> Prefix = 14; }
class A6     { bits<5> Prefix = 15; }
class A7     { bits<5> Prefix = 16; }
class T8XD   { bits<5> Prefix = 17; }
class T8XS   { bits<5> Prefix = 18; }
class TAXD   { bits<5> Prefix = 19; }
class XOP8   { bits<5> Prefix = 20; }
class XOP9   { bits<5> Prefix = 21; }
class VEX    { bit hasVEXPrefix = 1; }
class VEX_W  { bit hasVEX_WPrefix = 1; }
class VEX_4V : VEX { bit hasVEX_4VPrefix = 1; }
class VEX_4VOp3 : VEX { bit hasVEX_4VOp3Prefix = 1; }
class VEX_I8IMM { bit hasVEX_i8ImmReg = 1; }
class VEX_L  { bit hasVEX_L = 1; }
class VEX_LIG { bit ignoresVEX_L = 1; }
class Has3DNow0F0FOpcode  { bit has3DNow0F0FOpcode = 1; }
class MemOp4 { bit hasMemOp4Prefix = 1; }
class XOP { bit hasXOP_Prefix = 1; }
class X86Inst<bits<8> opcod, Format f, ImmType i, dag outs, dag ins,
              string AsmStr,
              InstrItinClass itin,
              Domain d = GenericDomain>
  : Instruction {
  let Namespace = "X86";

  bits<8> Opcode = opcod;
  Format Form = f;
  bits<6> FormBits = Form.Value;
  ImmType ImmT = i;

  dag OutOperandList = outs;
  dag InOperandList = ins;
  string AsmString = AsmStr;

  // If this is a pseudo instruction, mark it isCodeGenOnly.
  let isCodeGenOnly = !eq(!cast<string>(f), "Pseudo");

  let Itinerary = itin;

  //
  // Attributes specific to X86 instructions...
  //
  bit hasOpSizePrefix = 0;  // Does this inst have a 0x66 prefix?
  bit hasAdSizePrefix = 0;  // Does this inst have a 0x67 prefix?

  bits<5> Prefix = 0;       // Which prefix byte does this inst have?
  bit hasREX_WPrefix  = 0;  // Does this inst require the REX.W prefix?
  FPFormat FPForm = NotFP;  // What flavor of FP instruction is this?
  bit hasLockPrefix = 0;    // Does this inst have a 0xF0 prefix?
  bits<2> SegOvrBits = 0;   // Segment override prefix.
  Domain ExeDomain = d;
  bit hasVEXPrefix = 0;     // Does this inst require a VEX prefix?
  bit hasVEX_WPrefix = 0;   // Does this inst set the VEX_W field?
  bit hasVEX_4VPrefix = 0;  // Does this inst require the VEX.VVVV field?
  bit hasVEX_4VOp3Prefix = 0;  // Does this inst require the VEX.VVVV field to
                               // encode the third operand?
  bit hasVEX_i8ImmReg = 0;  // Does this inst require the last source register
                            // to be encoded in a immediate field?
  bit hasVEX_L = 0;         // Does this inst use large (256-bit) registers?
  bit ignoresVEX_L = 0;     // Does this instruction ignore the L-bit
  bit has3DNow0F0FOpcode =0;// Wacky 3dNow! encoding?
  bit hasMemOp4Prefix = 0;  // Same bit as VEX_W, but used for swapping operands
  bit hasXOP_Prefix = 0;    // Does this inst require an XOP prefix?

  // TSFlags layout should be kept in sync with X86InstrInfo.h.
  let TSFlags{5-0}   = FormBits;
  let TSFlags{6}     = hasOpSizePrefix;
  let TSFlags{7}     = hasAdSizePrefix;
  let TSFlags{12-8}  = Prefix;
  let TSFlags{13}    = hasREX_WPrefix;
  let TSFlags{16-14} = ImmT.Value;
  let TSFlags{19-17} = FPForm.Value;
  let TSFlags{20}    = hasLockPrefix;
  let TSFlags{22-21} = SegOvrBits;
  let TSFlags{24-23} = ExeDomain.Value;
  let TSFlags{32-25} = Opcode;
  let TSFlags{33}    = hasVEXPrefix;
  let TSFlags{34}    = hasVEX_WPrefix;
  let TSFlags{35}    = hasVEX_4VPrefix;
  let TSFlags{36}    = hasVEX_4VOp3Prefix;
  let TSFlags{37}    = hasVEX_i8ImmReg;
  let TSFlags{38}    = hasVEX_L;
  let TSFlags{39}    = ignoresVEX_L;
  let TSFlags{40}    = has3DNow0F0FOpcode;
  let TSFlags{41}    = hasMemOp4Prefix;
  let TSFlags{42}    = hasXOP_Prefix;
}

class PseudoI<dag oops, dag iops, list<dag> pattern>
  : X86Inst<0, Pseudo, NoImm, oops, iops, "", NoItinerary> {
  let Pattern = pattern;
}

class I<bits<8> o, Format f, dag outs, dag ins, string asm,
        list<dag> pattern, InstrItinClass itin = IIC_DEFAULT,
        Domain d = GenericDomain>
  : X86Inst<o, f, NoImm, outs, ins, asm, itin, d> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii8 <bits<8> o, Format f, dag outs, dag ins, string asm, 
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT,
           Domain d = GenericDomain>
  : X86Inst<o, f, Imm8, outs, ins, asm, itin, d> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii8PCRel<bits<8> o, Format f, dag outs, dag ins, string asm, 
               list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
  : X86Inst<o, f, Imm8PCRel, outs, ins, asm, itin> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii16<bits<8> o, Format f, dag outs, dag ins, string asm, 
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
  : X86Inst<o, f, Imm16, outs, ins, asm, itin> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii32<bits<8> o, Format f, dag outs, dag ins, string asm, 
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
  : X86Inst<o, f, Imm32, outs, ins, asm, itin> {
  let Pattern = pattern;
  let CodeSize = 3;
}

class Ii16PCRel<bits<8> o, Format f, dag outs, dag ins, string asm, 
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
           : X86Inst<o, f, Imm16PCRel, outs, ins, asm, itin> {
  let Pattern = pattern;
  let CodeSize = 3;
}

class Ii32PCRel<bits<8> o, Format f, dag outs, dag ins, string asm, 
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
  : X86Inst<o, f, Imm32PCRel, outs, ins, asm, itin> {
  let Pattern = pattern;
  let CodeSize = 3;
}

// FPStack Instruction Templates:
// FPI - Floating Point Instruction template.
class FPI<bits<8> o, Format F, dag outs, dag ins, string asm,
          InstrItinClass itin = IIC_DEFAULT>
  : I<o, F, outs, ins, asm, [], itin> {}

// FpI_ - Floating Point Pseudo Instruction template. Not Predicated.
class FpI_<dag outs, dag ins, FPFormat fp, list<dag> pattern,
           InstrItinClass itin = IIC_DEFAULT>
  : X86Inst<0, Pseudo, NoImm, outs, ins, "", itin> {
  let FPForm = fp;
  let Pattern = pattern;
}

// Templates for instructions that use a 16- or 32-bit segmented address as
//  their only operand: lcall (FAR CALL) and ljmp (FAR JMP)
//
//   Iseg16 - 16-bit segment selector, 16-bit offset
//   Iseg32 - 16-bit segment selector, 32-bit offset

class Iseg16 <bits<8> o, Format f, dag outs, dag ins, string asm, 
              list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : X86Inst<o, f, Imm16, outs, ins, asm, itin> {
  let Pattern = pattern;
  let CodeSize = 3;
}

class Iseg32 <bits<8> o, Format f, dag outs, dag ins, string asm, 
              list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : X86Inst<o, f, Imm32, outs, ins, asm, itin> {
  let Pattern = pattern;
  let CodeSize = 3;
}

def __xs : XS;

// SI - SSE 1 & 2 scalar instructions
class SI<bits<8> o, Format F, dag outs, dag ins, string asm,
         list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin> {
  let Predicates = !if(hasVEXPrefix /* VEX */, [HasAVX],
            !if(!eq(Prefix, __xs.Prefix), [UseSSE1], [UseSSE2]));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(hasVEXPrefix, !strconcat("v", asm), asm);
}

// SIi8 - SSE 1 & 2 scalar instructions
class SIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin> {
  let Predicates = !if(hasVEXPrefix /* VEX */, [HasAVX],
            !if(!eq(Prefix, __xs.Prefix), [UseSSE1], [UseSSE2]));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(hasVEXPrefix, !strconcat("v", asm), asm);
}

// PI - SSE 1 & 2 packed instructions
class PI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern,
         InstrItinClass itin, Domain d>
      : I<o, F, outs, ins, asm, pattern, itin, d> {
  let Predicates = !if(hasVEXPrefix /* VEX */, [HasAVX],
        !if(hasOpSizePrefix /* OpSize */, [UseSSE2], [UseSSE1]));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(hasVEXPrefix, !strconcat("v", asm), asm);
}

// MMXPI - SSE 1 & 2 packed instructions with MMX operands
class MMXPI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern,
            InstrItinClass itin, Domain d>
      : I<o, F, outs, ins, asm, pattern, itin, d> {
  let Predicates = !if(hasOpSizePrefix /* OpSize */, [HasSSE2], [HasSSE1]);
}

// PIi8 - SSE 1 & 2 packed instructions with immediate
class PIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin, Domain d>
      : Ii8<o, F, outs, ins, asm, pattern, itin, d> {
  let Predicates = !if(hasVEX_4VPrefix /* VEX */, [HasAVX],
        !if(hasOpSizePrefix /* OpSize */, [UseSSE2], [UseSSE1]));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(hasVEX_4VPrefix, !strconcat("v", asm), asm);
}

// SSE1 Instruction Templates:
// 
//   SSI   - SSE1 instructions with XS prefix.
//   PSI   - SSE1 instructions with TB prefix.
//   PSIi8 - SSE1 instructions with ImmT == Imm8 and TB prefix.
//   VSSI  - SSE1 instructions with XS prefix in AVX form.
//   VPSI  - SSE1 instructions with TB prefix in AVX form.

class SSI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, XS, Requires<[UseSSE1]>;
class SSIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin>, XS, Requires<[UseSSE1]>;
class PSI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedSingle>, TB,
        Requires<[UseSSE1]>;
class PSIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedSingle>, TB,
        Requires<[UseSSE1]>;
class VSSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern, itin>, XS,
        Requires<[HasAVX]>;
class VPSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern, itin, SSEPackedSingle>, TB,
        Requires<[HasAVX]>;

// SSE2 Instruction Templates:
// 
//   SDI    - SSE2 instructions with XD prefix.
//   SDIi8  - SSE2 instructions with ImmT == Imm8 and XD prefix.
//   S2SI   - SSE2 instructions with XS prefix.
//   SSDIi8 - SSE2 instructions with ImmT == Imm8 and XS prefix.
//   PDI    - SSE2 instructions with TB and OpSize prefixes.
//   PDIi8  - SSE2 instructions with ImmT == Imm8 and TB and OpSize prefixes.
//   VSDI   - SSE2 instructions with XD prefix in AVX form.
//   VPDI   - SSE2 instructions with TB and OpSize prefixes in AVX form.
//   MMXSDIi8  - SSE2 instructions with ImmT == Imm8 and XD prefix as well as
//               MMX operands.
//   MMXSSDIi8 - SSE2 instructions with ImmT == Imm8 and XS prefix as well as
//               MMX operands.

class SDI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, XD, Requires<[UseSSE2]>;
class SDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin>, XD, Requires<[UseSSE2]>;
class S2SI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, XS, Requires<[UseSSE2]>;
class S2SIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[UseSSE2]>;
class PDI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedDouble>, TB, OpSize,
        Requires<[UseSSE2]>;
class PDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedDouble>, TB, OpSize,
        Requires<[UseSSE2]>;
class VSDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern, itin>, XD,
        Requires<[HasAVX]>;
class VS2SI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern, itin>, XS,
        Requires<[HasAVX]>;
class VPDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern, itin, SSEPackedDouble>, TB,
        OpSize, Requires<[HasAVX]>;
class MMXSDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin>, XD, Requires<[HasSSE2]>;
class MMXS2SIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
                list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[HasSSE2]>;

// SSE3 Instruction Templates:
// 
//   S3I   - SSE3 instructions with TB and OpSize prefixes.
//   S3SI  - SSE3 instructions with XS prefix.
//   S3DI  - SSE3 instructions with XD prefix.

class S3SI<bits<8> o, Format F, dag outs, dag ins, string asm, 
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedSingle>, XS,
        Requires<[UseSSE3]>;
class S3DI<bits<8> o, Format F, dag outs, dag ins, string asm, 
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedDouble>, XD,
        Requires<[UseSSE3]>;
class S3I<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedDouble>, TB, OpSize,
        Requires<[UseSSE3]>;


// SSSE3 Instruction Templates:
// 
//   SS38I - SSSE3 instructions with T8 prefix.
//   SS3AI - SSSE3 instructions with TA prefix.
//   MMXSS38I - SSSE3 instructions with T8 prefix and MMX operands.
//   MMXSS3AI - SSSE3 instructions with TA prefix and MMX operands.
//
// Note: SSSE3 instructions have 64-bit and 128-bit versions. The 64-bit version
// uses the MMX registers. The 64-bit versions are grouped with the MMX
// classes. They need to be enabled even if AVX is enabled.

class SS38I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, T8,
        Requires<[UseSSSE3]>;
class SS3AI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA,
        Requires<[UseSSSE3]>;
class MMXSS38I<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, T8,
        Requires<[HasSSSE3]>;
class MMXSS3AI<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA,
        Requires<[HasSSSE3]>;

// SSE4.1 Instruction Templates:
// 
//   SS48I - SSE 4.1 instructions with T8 prefix.
//   SS41AIi8 - SSE 4.1 instructions with TA prefix and ImmT == Imm8.
//
class SS48I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, T8,
        Requires<[UseSSE41]>;
class SS4AIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA,
        Requires<[UseSSE41]>;

// SSE4.2 Instruction Templates:
// 
//   SS428I - SSE 4.2 instructions with T8 prefix.
class SS428I<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, T8,
        Requires<[UseSSE42]>;

//   SS42FI - SSE 4.2 instructions with T8XD prefix.
// NOTE: 'HasSSE42' is used as SS42FI is only used for CRC32 insns.
class SS42FI<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, T8XD, Requires<[HasSSE42]>;

//   SS42AI = SSE 4.2 instructions with TA prefix
class SS42AI<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA,
        Requires<[UseSSE42]>;

// AVX Instruction Templates:
//   Instructions introduced in AVX (no SSE equivalent forms)
//
//   AVX8I - AVX instructions with T8 and OpSize prefix.
//   AVXAIi8 - AVX instructions with TA, OpSize prefix and ImmT = Imm8.
class AVX8I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, T8, OpSize,
        Requires<[HasAVX]>;
class AVXAIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA, OpSize,
        Requires<[HasAVX]>;

// AVX2 Instruction Templates:
//   Instructions introduced in AVX2 (no SSE equivalent forms)
//
//   AVX28I - AVX2 instructions with T8 and OpSize prefix.
//   AVX2AIi8 - AVX2 instructions with TA, OpSize prefix and ImmT = Imm8.
class AVX28I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, T8, OpSize,
        Requires<[HasAVX2]>;
class AVX2AIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA, OpSize,
        Requires<[HasAVX2]>;

// AES Instruction Templates:
//
// AES8I
// These use the same encoding as the SSE4.2 T8 and TA encodings.
class AES8I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag>pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, T8,
        Requires<[HasAES]>;

class AESAI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA,
        Requires<[HasAES]>;

// PCLMUL Instruction Templates
class PCLMULIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag>pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA,
        OpSize, Requires<[HasPCLMUL]>;

class AVXPCLMULIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
                  list<dag>pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA,
        OpSize, VEX_4V, Requires<[HasAVX, HasPCLMUL]>;

// FMA3 Instruction Templates
class FMA3<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, T8,
        OpSize, VEX_4V, Requires<[HasFMA]>;

// FMA4 Instruction Templates
class FMA4<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA,
        OpSize, VEX_4V, VEX_I8IMM, Requires<[HasFMA4]>;

// XOP 2, 3 and 4 Operand Instruction Template
class IXOP<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin, SSEPackedDouble>,
         XOP, XOP9, Requires<[HasXOP]>;

// XOP 2, 3 and 4 Operand Instruction Templates with imm byte
class IXOPi8<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedDouble>,
         XOP, XOP8, Requires<[HasXOP]>;

//  XOP 5 operand instruction (VEX encoding!)
class IXOP5<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin, SSEPackedInt>, TA,
        OpSize, VEX_4V, VEX_I8IMM, Requires<[HasXOP]>;

// X86-64 Instruction templates...
//

class RI<bits<8> o, Format F, dag outs, dag ins, string asm,
         list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, REX_W;
class RIi8 <bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin>, REX_W;
class RIi32 <bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii32<o, F, outs, ins, asm, pattern, itin>, REX_W;

class RIi64<bits<8> o, Format f, dag outs, dag ins, string asm,
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
  : X86Inst<o, f, Imm64, outs, ins, asm, itin>, REX_W {
  let Pattern = pattern;
  let CodeSize = 3;
}

class RSSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : SSI<o, F, outs, ins, asm, pattern, itin>, REX_W;
class RSDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : SDI<o, F, outs, ins, asm, pattern, itin>, REX_W;
class RPDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : PDI<o, F, outs, ins, asm, pattern, itin>, REX_W;
class VRPDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : VPDI<o, F, outs, ins, asm, pattern, itin>, VEX_W;

// MMX Instruction templates
//

// MMXI   - MMX instructions with TB prefix.
// MMXI64 - MMX instructions with TB prefix valid only in 64 bit mode.
// MMX2I  - MMX / SSE2 instructions with TB and OpSize prefixes.
// MMXIi8 - MMX instructions with ImmT == Imm8 and TB prefix.
// MMXIi8 - MMX instructions with ImmT == Imm8 and TB prefix.
// MMXID  - MMX instructions with XD prefix.
// MMXIS  - MMX instructions with XS prefix.
class MMXI<bits<8> o, Format F, dag outs, dag ins, string asm, 
           list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, TB, Requires<[HasMMX]>;
class MMXI64<bits<8> o, Format F, dag outs, dag ins, string asm, 
             list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, TB, Requires<[HasMMX,In64BitMode]>;
class MMXRI<bits<8> o, Format F, dag outs, dag ins, string asm, 
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, TB, REX_W, Requires<[HasMMX]>;
class MMX2I<bits<8> o, Format F, dag outs, dag ins, string asm, 
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : I<o, F, outs, ins, asm, pattern, itin>, TB, OpSize, Requires<[HasMMX]>;
class MMXIi8<bits<8> o, Format F, dag outs, dag ins, string asm, 
             list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin>, TB, Requires<[HasMMX]>;
class MMXID<bits<8> o, Format F, dag outs, dag ins, string asm, 
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin>, XD, Requires<[HasMMX]>;
class MMXIS<bits<8> o, Format F, dag outs, dag ins, string asm, 
            list<dag> pattern, InstrItinClass itin = IIC_DEFAULT>
      : Ii8<o, F, outs, ins, asm, pattern, itin>, XS, Requires<[HasMMX]>;
