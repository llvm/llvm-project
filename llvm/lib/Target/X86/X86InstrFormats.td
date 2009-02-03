//===- X86InstrFormats.td - X86 Instruction Formats --------*- tablegen -*-===//
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


// ImmType - This specifies the immediate type used by an instruction. This is
// part of the ad-hoc solution used to emit machine instruction encodings by our
// machine code emitter.
class ImmType<bits<3> val> {
  bits<3> Value = val;
}
def NoImm  : ImmType<0>;
def Imm8   : ImmType<1>;
def Imm16  : ImmType<2>;
def Imm32  : ImmType<3>;
def Imm64  : ImmType<4>;

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

// Prefix byte classes which are used to indicate to the ad-hoc machine code
// emitter that various prefix bytes are required.
class OpSize { bit hasOpSizePrefix = 1; }
class AdSize { bit hasAdSizePrefix = 1; }
class REX_W  { bit hasREX_WPrefix = 1; }
class LOCK   { bit hasLockPrefix = 1; }
class SegFS  { bits<2> SegOvrBits = 1; }
class SegGS  { bits<2> SegOvrBits = 2; }
class TB     { bits<4> Prefix = 1; }
class REP    { bits<4> Prefix = 2; }
class D8     { bits<4> Prefix = 3; }
class D9     { bits<4> Prefix = 4; }
class DA     { bits<4> Prefix = 5; }
class DB     { bits<4> Prefix = 6; }
class DC     { bits<4> Prefix = 7; }
class DD     { bits<4> Prefix = 8; }
class DE     { bits<4> Prefix = 9; }
class DF     { bits<4> Prefix = 10; }
class XD     { bits<4> Prefix = 11; }
class XS     { bits<4> Prefix = 12; }
class T8     { bits<4> Prefix = 13; }
class TA     { bits<4> Prefix = 14; }

class X86Inst<bits<8> opcod, Format f, ImmType i, dag outs, dag ins,
              string AsmStr>
  : Instruction {
  let Namespace = "X86";

  bits<8> Opcode = opcod;
  Format Form = f;
  bits<6> FormBits = Form.Value;
  ImmType ImmT = i;
  bits<3> ImmTypeBits = ImmT.Value;

  dag OutOperandList = outs;
  dag InOperandList = ins;
  string AsmString = AsmStr;

  //
  // Attributes specific to X86 instructions...
  //
  bit hasOpSizePrefix = 0;  // Does this inst have a 0x66 prefix?
  bit hasAdSizePrefix = 0;  // Does this inst have a 0x67 prefix?

  bits<4> Prefix = 0;       // Which prefix byte does this inst have?
  bit hasREX_WPrefix  = 0;  // Does this inst requires the REX.W prefix?
  FPFormat FPForm;          // What flavor of FP instruction is this?
  bits<3> FPFormBits = 0;
  bit hasLockPrefix = 0;    // Does this inst have a 0xF0 prefix?
  bits<2> SegOvrBits = 0;   // Segment override prefix.
}

class I<bits<8> o, Format f, dag outs, dag ins, string asm, list<dag> pattern>
  : X86Inst<o, f, NoImm, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii8 <bits<8> o, Format f, dag outs, dag ins, string asm, list<dag> pattern>
  : X86Inst<o, f, Imm8 , outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii16<bits<8> o, Format f, dag outs, dag ins, string asm, list<dag> pattern>
  : X86Inst<o, f, Imm16, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii32<bits<8> o, Format f, dag outs, dag ins, string asm, list<dag> pattern>
  : X86Inst<o, f, Imm32, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}

// FPStack Instruction Templates:
// FPI - Floating Point Instruction template.
class FPI<bits<8> o, Format F, dag outs, dag ins, string asm>
  : I<o, F, outs, ins, asm, []> {}

// FpI_ - Floating Point Psuedo Instruction template. Not Predicated.
class FpI_<dag outs, dag ins, FPFormat fp, list<dag> pattern>
  : X86Inst<0, Pseudo, NoImm, outs, ins, ""> {
  let FPForm = fp; let FPFormBits = FPForm.Value;
  let Pattern = pattern;
}

// SSE1 Instruction Templates:
// 
//   SSI   - SSE1 instructions with XS prefix.
//   PSI   - SSE1 instructions with TB prefix.
//   PSIi8 - SSE1 instructions with ImmT == Imm8 and TB prefix.

class SSI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XS, Requires<[HasSSE1]>;
class SSIi8<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[HasSSE1]>;
class PSI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, TB, Requires<[HasSSE1]>;
class PSIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, TB, Requires<[HasSSE1]>;

// SSE2 Instruction Templates:
// 
//   SDI    - SSE2 instructions with XD prefix.
//   SDIi8  - SSE2 instructions with ImmT == Imm8 and XD prefix.
//   SSDIi8 - SSE2 instructions with ImmT == Imm8 and XS prefix.
//   PDI    - SSE2 instructions with TB and OpSize prefixes.
//   PDIi8  - SSE2 instructions with ImmT == Imm8 and TB and OpSize prefixes.

class SDI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XD, Requires<[HasSSE2]>;
class SDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XD, Requires<[HasSSE2]>;
class SSDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[HasSSE2]>;
class PDI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, TB, OpSize, Requires<[HasSSE2]>;
class PDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, TB, OpSize, Requires<[HasSSE2]>;

// SSE3 Instruction Templates:
// 
//   S3I   - SSE3 instructions with TB and OpSize prefixes.
//   S3SI  - SSE3 instructions with XS prefix.
//   S3DI  - SSE3 instructions with XD prefix.

class S3SI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XS, Requires<[HasSSE3]>;
class S3DI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XD, Requires<[HasSSE3]>;
class S3I<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, TB, OpSize, Requires<[HasSSE3]>;


// SSSE3 Instruction Templates:
// 
//   SS38I - SSSE3 instructions with T8 prefix.
//   SS3AI - SSSE3 instructions with TA prefix.
//
// Note: SSSE3 instructions have 64-bit and 128-bit versions. The 64-bit version
// uses the MMX registers. We put those instructions here because they better
// fit into the SSSE3 instruction category rather than the MMX category.

class SS38I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, T8, Requires<[HasSSSE3]>;
class SS3AI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, TA, Requires<[HasSSSE3]>;

// SSE4.1 Instruction Templates:
// 
//   SS48I - SSE 4.1 instructions with T8 prefix.
//   SS41AIi8 - SSE 4.1 instructions with TA prefix and ImmT == Imm8.
//
class SS48I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, T8, Requires<[HasSSE41]>;
class SS4AIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, TA, Requires<[HasSSE41]>;

// SSE4.2 Instruction Templates:
// 
//   SS428I - SSE 4.2 instructions with T8 prefix.
class SS428I<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, T8, Requires<[HasSSE42]>;

// X86-64 Instruction templates...
//

class RI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, REX_W;
class RIi8 <bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, REX_W;
class RIi32 <bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii32<o, F, outs, ins, asm, pattern>, REX_W;

class RIi64<bits<8> o, Format f, dag outs, dag ins, string asm,
            list<dag> pattern>
  : X86Inst<o, f, Imm64, outs, ins, asm>, REX_W {
  let Pattern = pattern;
  let CodeSize = 3;
}

class RSSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : SSI<o, F, outs, ins, asm, pattern>, REX_W;
class RSDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : SDI<o, F, outs, ins, asm, pattern>, REX_W;
class RPDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : PDI<o, F, outs, ins, asm, pattern>, REX_W;

// MMX Instruction templates
//

// MMXI   - MMX instructions with TB prefix.
// MMXI64 - MMX instructions with TB prefix valid only in 64 bit mode.
// MMX2I  - MMX / SSE2 instructions with TB and OpSize prefixes.
// MMXIi8 - MMX instructions with ImmT == Imm8 and TB prefix.
// MMXIi8 - MMX instructions with ImmT == Imm8 and TB prefix.
// MMXID  - MMX instructions with XD prefix.
// MMXIS  - MMX instructions with XS prefix.
class MMXI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, TB, Requires<[HasMMX]>;
class MMXI64<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, TB, Requires<[HasMMX,In64BitMode]>;
class MMXRI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, TB, REX_W, Requires<[HasMMX]>;
class MMX2I<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, TB, OpSize, Requires<[HasMMX]>;
class MMXIi8<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, TB, Requires<[HasMMX]>;
class MMXID<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XD, Requires<[HasMMX]>;
class MMXIS<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[HasMMX]>;

