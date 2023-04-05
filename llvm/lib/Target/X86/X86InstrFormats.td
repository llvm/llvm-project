//===-- X86InstrFormats.td - X86 Instruction Formats -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<7> val> {
  bits<7> Value = val;
}

def Pseudo        : Format<0>;
def RawFrm        : Format<1>;
def AddRegFrm     : Format<2>;
def RawFrmMemOffs : Format<3>;
def RawFrmSrc     : Format<4>;
def RawFrmDst     : Format<5>;
def RawFrmDstSrc  : Format<6>;
def RawFrmImm8    : Format<7>;
def RawFrmImm16   : Format<8>;
def AddCCFrm      : Format<9>;
def PrefixByte    : Format<10>;
def MRMDestMem4VOp3CC : Format<20>;
def MRMr0          : Format<21>;
def MRMSrcMemFSIB  : Format<22>;
def MRMDestMemFSIB : Format<23>;
def MRMDestMem     : Format<24>;
def MRMSrcMem      : Format<25>;
def MRMSrcMem4VOp3 : Format<26>;
def MRMSrcMemOp4   : Format<27>;
def MRMSrcMemCC    : Format<28>;
def MRMXmCC: Format<30>;
def MRMXm  : Format<31>;
def MRM0m  : Format<32>;  def MRM1m  : Format<33>;  def MRM2m  : Format<34>;
def MRM3m  : Format<35>;  def MRM4m  : Format<36>;  def MRM5m  : Format<37>;
def MRM6m  : Format<38>;  def MRM7m  : Format<39>;
def MRMDestReg     : Format<40>;
def MRMSrcReg      : Format<41>;
def MRMSrcReg4VOp3 : Format<42>;
def MRMSrcRegOp4   : Format<43>;
def MRMSrcRegCC    : Format<44>;
def MRMXrCC: Format<46>;
def MRMXr  : Format<47>;
def MRM0r  : Format<48>;  def MRM1r  : Format<49>;  def MRM2r  : Format<50>;
def MRM3r  : Format<51>;  def MRM4r  : Format<52>;  def MRM5r  : Format<53>;
def MRM6r  : Format<54>;  def MRM7r  : Format<55>;
def MRM0X  : Format<56>;  def MRM1X  : Format<57>;  def MRM2X  : Format<58>;
def MRM3X  : Format<59>;  def MRM4X  : Format<60>;  def MRM5X  : Format<61>;
def MRM6X  : Format<62>;  def MRM7X  : Format<63>;
def MRM_C0 : Format<64>;  def MRM_C1 : Format<65>;  def MRM_C2 : Format<66>;
def MRM_C3 : Format<67>;  def MRM_C4 : Format<68>;  def MRM_C5 : Format<69>;
def MRM_C6 : Format<70>;  def MRM_C7 : Format<71>;  def MRM_C8 : Format<72>;
def MRM_C9 : Format<73>;  def MRM_CA : Format<74>;  def MRM_CB : Format<75>;
def MRM_CC : Format<76>;  def MRM_CD : Format<77>;  def MRM_CE : Format<78>;
def MRM_CF : Format<79>;  def MRM_D0 : Format<80>;  def MRM_D1 : Format<81>;
def MRM_D2 : Format<82>;  def MRM_D3 : Format<83>;  def MRM_D4 : Format<84>;
def MRM_D5 : Format<85>;  def MRM_D6 : Format<86>;  def MRM_D7 : Format<87>;
def MRM_D8 : Format<88>;  def MRM_D9 : Format<89>;  def MRM_DA : Format<90>;
def MRM_DB : Format<91>;  def MRM_DC : Format<92>;  def MRM_DD : Format<93>;
def MRM_DE : Format<94>;  def MRM_DF : Format<95>;  def MRM_E0 : Format<96>;
def MRM_E1 : Format<97>;  def MRM_E2 : Format<98>;  def MRM_E3 : Format<99>;
def MRM_E4 : Format<100>; def MRM_E5 : Format<101>; def MRM_E6 : Format<102>;
def MRM_E7 : Format<103>; def MRM_E8 : Format<104>; def MRM_E9 : Format<105>;
def MRM_EA : Format<106>; def MRM_EB : Format<107>; def MRM_EC : Format<108>;
def MRM_ED : Format<109>; def MRM_EE : Format<110>; def MRM_EF : Format<111>;
def MRM_F0 : Format<112>; def MRM_F1 : Format<113>; def MRM_F2 : Format<114>;
def MRM_F3 : Format<115>; def MRM_F4 : Format<116>; def MRM_F5 : Format<117>;
def MRM_F6 : Format<118>; def MRM_F7 : Format<119>; def MRM_F8 : Format<120>;
def MRM_F9 : Format<121>; def MRM_FA : Format<122>; def MRM_FB : Format<123>;
def MRM_FC : Format<124>; def MRM_FD : Format<125>; def MRM_FE : Format<126>;
def MRM_FF : Format<127>;

// ImmType - This specifies the immediate type used by an instruction. This is
// part of the ad-hoc solution used to emit machine instruction encodings by our
// machine code emitter.
class ImmType<bits<4> val> {
  bits<4> Value = val;
}
def NoImm      : ImmType<0>;
def Imm8       : ImmType<1>;
def Imm8PCRel  : ImmType<2>;
def Imm8Reg    : ImmType<3>; // Register encoded in [7:4].
def Imm16      : ImmType<4>;
def Imm16PCRel : ImmType<5>;
def Imm32      : ImmType<6>;
def Imm32PCRel : ImmType<7>;
def Imm32S     : ImmType<8>;
def Imm64      : ImmType<9>;

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

// Class specifying the vector form of the decompressed
// displacement of 8-bit.
class CD8VForm<bits<3> val> {
  bits<3> Value = val;
}
def CD8VF  : CD8VForm<0>;  // v := VL
def CD8VH  : CD8VForm<1>;  // v := VL/2
def CD8VQ  : CD8VForm<2>;  // v := VL/4
def CD8VO  : CD8VForm<3>;  // v := VL/8
// The tuple (subvector) forms.
def CD8VT1 : CD8VForm<4>;  // v := 1
def CD8VT2 : CD8VForm<5>;  // v := 2
def CD8VT4 : CD8VForm<6>;  // v := 4
def CD8VT8 : CD8VForm<7>;  // v := 8

// Class specifying the prefix used an opcode extension.
class Prefix<bits<3> val> {
  bits<3> Value = val;
}
def NoPrfx : Prefix<0>;
def PD     : Prefix<1>;
def XS     : Prefix<2>;
def XD     : Prefix<3>;
def PS     : Prefix<4>; // Similar to NoPrfx, but disassembler uses this to know
                        // that other instructions with this opcode use PD/XS/XD
                        // and if any of those is not supported they shouldn't
                        // decode to this instruction. e.g. ANDSS/ANDSD don't
                        // exist, but the 0xf2/0xf3 encoding shouldn't
                        // disable to ANDPS.

// Class specifying the opcode map.
class Map<bits<4> val> {
  bits<4> Value = val;
}
def OB        : Map<0>;
def TB        : Map<1>;
def T8        : Map<2>;
def TA        : Map<3>;
def XOP8      : Map<4>;
def XOP9      : Map<5>;
def XOPA      : Map<6>;
def ThreeDNow : Map<7>;
def T_MAP5    : Map<8>;
def T_MAP6    : Map<9>;

// Class specifying the encoding
class Encoding<bits<2> val> {
  bits<2> Value = val;
}
def EncNormal : Encoding<0>;
def EncVEX    : Encoding<1>;
def EncXOP    : Encoding<2>;
def EncEVEX   : Encoding<3>;

// Operand size for encodings that change based on mode.
class OperandSize<bits<2> val> {
  bits<2> Value = val;
}
def OpSizeFixed  : OperandSize<0>; // Never needs a 0x66 prefix.
def OpSize16     : OperandSize<1>; // Needs 0x66 prefix in 32-bit mode.
def OpSize32     : OperandSize<2>; // Needs 0x66 prefix in 16-bit mode.

// Address size for encodings that change based on mode.
class AddressSize<bits<2> val> {
  bits<2> Value = val;
}
def AdSizeX  : AddressSize<0>; // Address size determined using addr operand.
def AdSize16 : AddressSize<1>; // Encodes a 16-bit address.
def AdSize32 : AddressSize<2>; // Encodes a 32-bit address.
def AdSize64 : AddressSize<3>; // Encodes a 64-bit address.

// Prefix byte classes which are used to indicate to the ad-hoc machine code
// emitter that various prefix bytes are required.
class OpSize16 { OperandSize OpSize = OpSize16; }
class OpSize32 { OperandSize OpSize = OpSize32; }
class AdSize16 { AddressSize AdSize = AdSize16; }
class AdSize32 { AddressSize AdSize = AdSize32; }
class AdSize64 { AddressSize AdSize = AdSize64; }
class REX_W  { bit hasREX_W = 1; }
class LOCK   { bit hasLockPrefix = 1; }
class REP    { bit hasREPPrefix = 1; }
class TB     { Map OpMap = TB; }
class T8     { Map OpMap = T8; }
class TA     { Map OpMap = TA; }
class XOP8   { Map OpMap = XOP8; Prefix OpPrefix = PS; }
class XOP9   { Map OpMap = XOP9; Prefix OpPrefix = PS; }
class XOPA   { Map OpMap = XOPA; Prefix OpPrefix = PS; }
class ThreeDNow { Map OpMap = ThreeDNow; }
class T_MAP5     { Map OpMap = T_MAP5; }
class T_MAP5PS : T_MAP5 { Prefix OpPrefix = PS; } // none
class T_MAP5PD : T_MAP5 { Prefix OpPrefix = PD; } // 0x66
class T_MAP5XS : T_MAP5 { Prefix OpPrefix = XS; } // 0xF3
class T_MAP5XD : T_MAP5 { Prefix OpPrefix = XD; } // 0xF2
class T_MAP6     { Map OpMap = T_MAP6; }
class T_MAP6PS : T_MAP6 { Prefix OpPrefix = PS; }
class T_MAP6PD : T_MAP6 { Prefix OpPrefix = PD; }
class T_MAP6XS : T_MAP6 { Prefix OpPrefix = XS; }
class T_MAP6XD : T_MAP6 { Prefix OpPrefix = XD; }
class OBXS   { Prefix OpPrefix = XS; }
class PS   : TB { Prefix OpPrefix = PS; }
class PD   : TB { Prefix OpPrefix = PD; }
class XD   : TB { Prefix OpPrefix = XD; }
class XS   : TB { Prefix OpPrefix = XS; }
class T8PS : T8 { Prefix OpPrefix = PS; }
class T8PD : T8 { Prefix OpPrefix = PD; }
class T8XD : T8 { Prefix OpPrefix = XD; }
class T8XS : T8 { Prefix OpPrefix = XS; }
class TAPS : TA { Prefix OpPrefix = PS; }
class TAPD : TA { Prefix OpPrefix = PD; }
class TAXD : TA { Prefix OpPrefix = XD; }
class TAXS : TA { Prefix OpPrefix = XS; }
class VEX    { Encoding OpEnc = EncVEX; }
class VEX_WIG  { bit IgnoresVEX_W = 1; }
// Special version of REX_W that can be changed to VEX.W==0 for EVEX2VEX.
class VEX_W1X  { bit hasREX_W = 1; bit EVEX_W1_VEX_W0 = 1; }
class VEX_4V : VEX { bit hasVEX_4V = 1; }
class VEX_L  { bit hasVEX_L = 1; }
class VEX_LIG { bit ignoresVEX_L = 1; }
class EVEX   { Encoding OpEnc = EncEVEX; }
class EVEX_4V : EVEX { bit hasVEX_4V = 1; }
class EVEX_K { bit hasEVEX_K = 1; }
class EVEX_KZ : EVEX_K { bit hasEVEX_Z = 1; }
class EVEX_B { bit hasEVEX_B = 1; }
class EVEX_RC { bit hasEVEX_RC = 1; }
class EVEX_V512 { bit hasEVEX_L2 = 1; bit hasVEX_L = 0; }
class EVEX_V256 { bit hasEVEX_L2 = 0; bit hasVEX_L = 1; }
class EVEX_V128 { bit hasEVEX_L2 = 0; bit hasVEX_L = 0; }
class NOTRACK { bit hasNoTrackPrefix = 1; }
class SIMD_EXC { list<Register> Uses = [MXCSR]; bit mayRaiseFPException = 1; }

// Specify AVX512 8-bit compressed displacement encoding based on the vector
// element size in bits (8, 16, 32, 64) and the CDisp8 form.
class EVEX_CD8<int esize, CD8VForm form> {
  int CD8_EltSize = !srl(esize, 3);
  bits<3> CD8_Form = form.Value;
}

class XOP { Encoding OpEnc = EncXOP; }
class XOP_4V : XOP { bit hasVEX_4V = 1; }

// Specify the alternative register form instruction to replace the current
// instruction in case it was picked during generation of memory folding tables
class FoldGenData<string _RegisterForm> {
  string FoldGenRegForm = _RegisterForm;
}

// Provide a specific instruction to be used by the EVEX2VEX conversion.
class EVEX2VEXOverride<string VEXInstrName> {
  string EVEX2VEXOverride = VEXInstrName;
}

// Prevent EVEX->VEX conversion from considering this instruction.
class NotEVEX2VEXConvertible { bit notEVEX2VEXConvertible = 1; }

// Force the instruction to use VEX encoding.
class ExplicitVEXPrefix { bit ExplicitVEXPrefix = 1; }

class X86Inst<bits<8> opcod, Format f, ImmType i, dag outs, dag ins,
              string AsmStr, Domain d = GenericDomain>
  : Instruction {
  let Namespace = "X86";

  bits<8> Opcode = opcod;
  Format Form = f;
  bits<7> FormBits = Form.Value;
  ImmType ImmT = i;

  dag OutOperandList = outs;
  dag InOperandList = ins;
  string AsmString = AsmStr;

  // If this is a pseudo instruction, mark it isCodeGenOnly.
  let isCodeGenOnly = !eq(!cast<string>(f), "Pseudo");

  let HasPositionOrder = 1;

  //
  // Attributes specific to X86 instructions...
  //
  bit ForceDisassemble = 0; // Force instruction to disassemble even though it's
                            // isCodeGenonly. Needed to hide an ambiguous
                            // AsmString from the parser, but still disassemble.

  OperandSize OpSize = OpSizeFixed; // Does this instruction's encoding change
                                    // based on operand size of the mode?
  bits<2> OpSizeBits = OpSize.Value;
  AddressSize AdSize = AdSizeX; // Does this instruction's encoding change
                                // based on address size of the mode?
  bits<2> AdSizeBits = AdSize.Value;

  Prefix OpPrefix = NoPrfx; // Which prefix byte does this inst have?
  bits<3> OpPrefixBits = OpPrefix.Value;
  Map OpMap = OB;           // Which opcode map does this inst have?
  bits<4> OpMapBits = OpMap.Value;
  bit hasREX_W  = 0;  // Does this inst require the REX.W prefix?
  FPFormat FPForm = NotFP;  // What flavor of FP instruction is this?
  bit hasLockPrefix = 0;    // Does this inst have a 0xF0 prefix?
  Domain ExeDomain = d;
  bit hasREPPrefix = 0;     // Does this inst have a REP prefix?
  Encoding OpEnc = EncNormal; // Encoding used by this instruction
  bits<2> OpEncBits = OpEnc.Value;
  bit IgnoresVEX_W = 0;     // Does this inst ignore REX_W field?
  bit EVEX_W1_VEX_W0 = 0;   // This EVEX inst with VEX.W==1 can become a VEX
                            // instruction with VEX.W == 0.
  bit hasVEX_4V = 0;        // Does this inst require the VEX.VVVV field?
  bit hasVEX_L = 0;         // Does this inst use large (256-bit) registers?
  bit ignoresVEX_L = 0;     // Does this instruction ignore the L-bit
  bit hasEVEX_K = 0;        // Does this inst require masking?
  bit hasEVEX_Z = 0;        // Does this inst set the EVEX_Z field?
  bit hasEVEX_L2 = 0;       // Does this inst set the EVEX_L2 field?
  bit hasEVEX_B = 0;        // Does this inst set the EVEX_B field?
  bits<3> CD8_Form = 0;     // Compressed disp8 form - vector-width.
  // Declare it int rather than bits<4> so that all bits are defined when
  // assigning to bits<7>.
  int CD8_EltSize = 0;      // Compressed disp8 form - element-size in bytes.
  bit hasEVEX_RC = 0;       // Explicitly specified rounding control in FP instruction.
  bit hasNoTrackPrefix = 0; // Does this inst has 0x3E (NoTrack) prefix?

  // Vector size in bytes.
  bits<7> VectSize = !if(hasEVEX_L2, 64, !if(hasVEX_L, 32, 16));

  // The scaling factor for AVX512's compressed displacement is either
  //   - the size of a  power-of-two number of elements or
  //   - the size of a single element for broadcasts or
  //   - the total vector size divided by a power-of-two number.
  // Possible values are: 0 (non-AVX512 inst), 1, 2, 4, 8, 16, 32 and 64.
  bits<7> CD8_Scale = !if (!eq (OpEnc.Value, EncEVEX.Value),
                           !if (CD8_Form{2},
                                !shl(CD8_EltSize, CD8_Form{1-0}),
                                !if (hasEVEX_B,
                                     CD8_EltSize,
                                     !srl(VectSize, CD8_Form{1-0}))), 0);

  // Used in the memory folding generation (TableGen backend) to point to an alternative
  // instruction to replace the current one in case it got picked during generation.
  string FoldGenRegForm = ?;

  // Used to prevent an explicit EVEX2VEX override for this instruction.
  string EVEX2VEXOverride = ?;

  bit notEVEX2VEXConvertible = 0; // Prevent EVEX->VEX conversion.
  bit ExplicitVEXPrefix = 0; // Force the instruction to use VEX encoding.
  // Force to check predicate before compress EVEX to VEX encoding.
  bit checkVEXPredicate = 0;
  // TSFlags layout should be kept in sync with X86BaseInfo.h.
  let TSFlags{6-0}   = FormBits;
  let TSFlags{8-7}   = OpSizeBits;
  let TSFlags{10-9}  = AdSizeBits;
  // No need for 3rd bit, we don't need to distinguish NoPrfx from PS.
  let TSFlags{12-11} = OpPrefixBits{1-0};
  let TSFlags{16-13} = OpMapBits;
  let TSFlags{17}    = hasREX_W;
  let TSFlags{21-18} = ImmT.Value;
  let TSFlags{24-22} = FPForm.Value;
  let TSFlags{25}    = hasLockPrefix;
  let TSFlags{26}    = hasREPPrefix;
  let TSFlags{28-27} = ExeDomain.Value;
  let TSFlags{30-29} = OpEncBits;
  let TSFlags{38-31} = Opcode;
  let TSFlags{39}    = hasVEX_4V;
  let TSFlags{40}    = hasVEX_L;
  let TSFlags{41}    = hasEVEX_K;
  let TSFlags{42}    = hasEVEX_Z;
  let TSFlags{43}    = hasEVEX_L2;
  let TSFlags{44}    = hasEVEX_B;
  let TSFlags{47-45} = !if(!eq(CD8_Scale, 0), 0, !add(!logtwo(CD8_Scale), 1));
  let TSFlags{48}    = hasEVEX_RC;
  let TSFlags{49}    = hasNoTrackPrefix;
  let TSFlags{50}    = ExplicitVEXPrefix;
}

class PseudoI<dag oops, dag iops, list<dag> pattern>
  : X86Inst<0, Pseudo, NoImm, oops, iops, ""> {
  let Pattern = pattern;
}

class I<bits<8> o, Format f, dag outs, dag ins, string asm,
        list<dag> pattern, Domain d = GenericDomain>
  : X86Inst<o, f, NoImm, outs, ins, asm, d> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii8<bits<8> o, Format f, dag outs, dag ins, string asm,
          list<dag> pattern, Domain d = GenericDomain>
  : X86Inst<o, f, Imm8, outs, ins, asm, d> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii8Reg<bits<8> o, Format f, dag outs, dag ins, string asm,
             list<dag> pattern, Domain d = GenericDomain>
  : X86Inst<o, f, Imm8Reg, outs, ins, asm, d> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii8PCRel<bits<8> o, Format f, dag outs, dag ins, string asm,
               list<dag> pattern>
  : X86Inst<o, f, Imm8PCRel, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii16<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
  : X86Inst<o, f, Imm16, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii32<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
  : X86Inst<o, f, Imm32, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}
class Ii32S<bits<8> o, Format f, dag outs, dag ins, string asm,
            list<dag> pattern>
  : X86Inst<o, f, Imm32S, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}

class Ii64<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
  : X86Inst<o, f, Imm64, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}

class Ii16PCRel<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
           : X86Inst<o, f, Imm16PCRel, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}

class Ii32PCRel<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
  : X86Inst<o, f, Imm32PCRel, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}

// FPStack Instruction Templates:
// FPI - Floating Point Instruction template.
class FPI<bits<8> o, Format F, dag outs, dag ins, string asm>
  : I<o, F, outs, ins, asm, []> {
  let Defs = [FPSW];
  let Predicates = [HasX87];
}

// FpI_ - Floating Point Pseudo Instruction template.
class FpI_<dag outs, dag ins, FPFormat fp, list<dag> pattern>
  : PseudoI<outs, ins, pattern> {
  let FPForm = fp;
  let Defs = [FPSW];
  let Predicates = [HasX87];
}

// Templates for instructions that use a 16- or 32-bit segmented address as
//  their only operand: lcall (FAR CALL) and ljmp (FAR JMP)
//
//   Iseg16 - 16-bit segment selector, 16-bit offset
//   Iseg32 - 16-bit segment selector, 32-bit offset

class Iseg16 <bits<8> o, Format f, dag outs, dag ins, string asm,
              list<dag> pattern>
      : X86Inst<o, f, Imm16, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}

class Iseg32 <bits<8> o, Format f, dag outs, dag ins, string asm,
              list<dag> pattern>
      : X86Inst<o, f, Imm32, outs, ins, asm> {
  let Pattern = pattern;
  let CodeSize = 3;
}

// SI - SSE 1 & 2 scalar instructions
class SI<bits<8> o, Format F, dag outs, dag ins, string asm,
         list<dag> pattern, Domain d = GenericDomain>
      : I<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [UseAVX],
                   !if(!eq(OpPrefix.Value, XS.Value), [UseSSE1],
                   !if(!eq(OpPrefix.Value, XD.Value), [UseSSE2],
                   !if(!eq(OpPrefix.Value, PD.Value), [UseSSE2],
                   [UseSSE1])))));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}

// SI - SSE 1 & 2 scalar intrinsics - vex form available on AVX512
class SI_Int<bits<8> o, Format F, dag outs, dag ins, string asm,
         list<dag> pattern, Domain d = GenericDomain>
      : I<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [UseAVX],
                   !if(!eq(OpPrefix.Value, XS.Value), [UseSSE1],
                   !if(!eq(OpPrefix.Value, XD.Value), [UseSSE2],
                   !if(!eq(OpPrefix.Value, PD.Value), [UseSSE2],
                   [UseSSE1])))));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}
// SIi8 - SSE 1 & 2 scalar instructions - vex form available on AVX512
class SIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [HasAVX],
                   !if(!eq(OpPrefix.Value, XS.Value), [UseSSE1],
                   [UseSSE2])));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}

// PI - SSE 1 & 2 packed instructions
class PI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern,
         Domain d>
      : I<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [HasAVX],
                   !if(!eq(OpPrefix.Value, PD.Value), [UseSSE2],
                   [UseSSE1])));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}

// MMXPI - SSE 1 & 2 packed instructions with MMX operands
class MMXPI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern,
            Domain d>
      : I<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpPrefix.Value, PD.Value), [HasMMX, HasSSE2],
                       [HasMMX, HasSSE1]);
}

// PIi8 - SSE 1 & 2 packed instructions with immediate
class PIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, Domain d>
      : Ii8<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [HasAVX],
                   !if(!eq(OpPrefix.Value, PD.Value), [UseSSE2],
                   [UseSSE1])));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}

// SSE1 Instruction Templates:
//
//   SSI   - SSE1 instructions with XS prefix.
//   PSI   - SSE1 instructions with PS prefix.
//   PSIi8 - SSE1 instructions with ImmT == Imm8 and PS prefix.
//   VSSI  - SSE1 instructions with XS prefix in AVX form.
//   VPSI  - SSE1 instructions with PS prefix in AVX form, packed single.

class SSI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XS, Requires<[UseSSE1]>;
class SSIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[UseSSE1]>;
class PSI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedSingle>, PS,
        Requires<[UseSSE1]>;
class PSIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedSingle>, PS,
        Requires<[UseSSE1]>;
class VSSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern>, XS,
        Requires<[HasAVX]>;
class VPSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern, SSEPackedSingle>, PS,
        Requires<[HasAVX]>;

// SSE2 Instruction Templates:
//
//   SDI    - SSE2 instructions with XD prefix.
//   SDIi8  - SSE2 instructions with ImmT == Imm8 and XD prefix.
//   S2SI   - SSE2 instructions with XS prefix.
//   SSDIi8 - SSE2 instructions with ImmT == Imm8 and XS prefix.
//   PDI    - SSE2 instructions with PD prefix, packed double domain.
//   PDIi8  - SSE2 instructions with ImmT == Imm8 and PD prefix.
//   VSDI   - SSE2 scalar instructions with XD prefix in AVX form.
//   VPDI   - SSE2 vector instructions with PD prefix in AVX form,
//                 packed double domain.
//   VS2I   - SSE2 scalar instructions with PD prefix in AVX form.
//   S2I    - SSE2 scalar instructions with PD prefix.
//   MMXSDIi8  - SSE2 instructions with ImmT == Imm8 and XD prefix as well as
//               MMX operands.
//   MMXSSDIi8 - SSE2 instructions with ImmT == Imm8 and XS prefix as well as
//               MMX operands.

class SDI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XD, Requires<[UseSSE2]>;
class SDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XD, Requires<[UseSSE2]>;
class S2SI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XS, Requires<[UseSSE2]>;
class S2SIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[UseSSE2]>;
class PDI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>, PD,
        Requires<[UseSSE2]>;
class PDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedDouble>, PD,
        Requires<[UseSSE2]>;
class VSDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern>, XD,
        Requires<[UseAVX]>;
class VS2SI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern>, XS,
        Requires<[HasAVX]>;
class VPDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern, SSEPackedDouble>,
        PD, Requires<[HasAVX]>;
class VS2I<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern>, PD,
        Requires<[UseAVX]>;
class S2I<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PD, Requires<[UseSSE2]>;
class MMXSDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XD, Requires<[HasMMX, HasSSE2]>;
class MMXS2SIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
                list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[HasMMX, HasSSE2]>;

// SSE3 Instruction Templates:
//
//   S3I   - SSE3 instructions with PD prefixes.
//   S3SI  - SSE3 instructions with XS prefix.
//   S3DI  - SSE3 instructions with XD prefix.

class S3SI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedSingle>, XS,
        Requires<[UseSSE3]>;
class S3DI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>, XD,
        Requires<[UseSSE3]>;
class S3I<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>, PD,
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
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[UseSSSE3]>;
class SS3AI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[UseSSSE3]>;
class MMXSS38I<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PS,
        Requires<[HasMMX, HasSSSE3]>;
class MMXSS3AI<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPS,
        Requires<[HasMMX, HasSSSE3]>;

// SSE4.1 Instruction Templates:
//
//   SS48I - SSE 4.1 instructions with T8 prefix.
//   SS41AIi8 - SSE 4.1 instructions with TA prefix and ImmT == Imm8.
//
class SS48I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[UseSSE41]>;
class SS4AIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[UseSSE41]>;

// SSE4.2 Instruction Templates:
//
//   SS428I - SSE 4.2 instructions with T8 prefix.
class SS428I<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[UseSSE42]>;

//   SS42AI = SSE 4.2 instructions with TA prefix
class SS42AI<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[UseSSE42]>;

//   CRC32I - SSE 4.2 CRC32 instructions.
// NOTE: 'HasCRC32' is used as CRC32 instructions are GPR only and not directly
// controlled by the SSE42 flag.
class CRC32I<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, T8XD, Requires<[HasCRC32]>;

// AVX Instruction Templates:
//   Instructions introduced in AVX (no SSE equivalent forms)
//
//   AVX8I - AVX instructions with T8PD prefix.
//   AVXAIi8 - AVX instructions with TAPD prefix and ImmT = Imm8.
class AVX8I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[HasAVX]>;
class AVXAIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[HasAVX]>;

// AVX2 Instruction Templates:
//   Instructions introduced in AVX2 (no SSE equivalent forms)
//
//   AVX28I - AVX2 instructions with T8PD prefix.
//   AVX2AIi8 - AVX2 instructions with TAPD prefix and ImmT = Imm8.
class AVX28I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[HasAVX2]>;
class AVX2AIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[HasAVX2]>;


// AVX-512 Instruction Templates:
//   Instructions introduced in AVX-512 (no SSE equivalent forms)
//
//   AVX5128I - AVX-512 instructions with T8PD prefix.
//   AVX512AIi8 - AVX-512 instructions with TAPD prefix and ImmT = Imm8.
//   AVX512PDI  - AVX-512 instructions with PD, double packed.
//   AVX512PSI  - AVX-512 instructions with PS, single packed.
//   AVX512XS8I - AVX-512 instructions with T8 and XS prefixes.
//   AVX512XSI  - AVX-512 instructions with XS prefix, generic domain.
//   AVX512BI   - AVX-512 instructions with PD, int packed domain.
//   AVX512SI   - AVX-512 scalar instructions with PD prefix.

class AVX5128I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[HasAVX512]>;
class AVX5128IBase : T8PD {
  Domain ExeDomain = SSEPackedInt;
}
class AVX512XS8I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8XS,
        Requires<[HasAVX512]>;
class AVX512XSI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XS,
        Requires<[HasAVX512]>;
class AVX512XDI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, XD,
        Requires<[HasAVX512]>;
class AVX512BI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, PD,
        Requires<[HasAVX512]>;
class AVX512BIBase : PD {
  Domain ExeDomain = SSEPackedInt;
}
class AVX512BIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, PD,
        Requires<[HasAVX512]>;
class AVX512BIi8Base : PD {
  Domain ExeDomain = SSEPackedInt;
  ImmType ImmT = Imm8;
}
class AVX512XSIi8Base : XS {
  Domain ExeDomain = SSEPackedInt;
  ImmType ImmT = Imm8;
}
class AVX512XDIi8Base : XD {
  Domain ExeDomain = SSEPackedInt;
  ImmType ImmT = Imm8;
}
class AVX512PSIi8Base : PS {
  Domain ExeDomain = SSEPackedSingle;
  ImmType ImmT = Imm8;
}
class AVX512PDIi8Base : PD {
  Domain ExeDomain = SSEPackedDouble;
  ImmType ImmT = Imm8;
}
class AVX512AIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[HasAVX512]>;
class AVX512AIi8Base : TAPD {
  ImmType ImmT = Imm8;
}
class AVX512Ii8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>,
        Requires<[HasAVX512]>;
class AVX512PDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>, PD,
        Requires<[HasAVX512]>;
class AVX512PSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedSingle>, PS,
        Requires<[HasAVX512]>;
class AVX512PIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern, Domain d>
      : Ii8<o, F, outs, ins, asm, pattern, d>, Requires<[HasAVX512]>;
class AVX512PI<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern, Domain d>
      : I<o, F, outs, ins, asm, pattern, d>, Requires<[HasAVX512]>;
class AVX512FMA3S<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, T8PD,
        EVEX_4V, Requires<[HasAVX512]>;

class AVX512<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, Requires<[HasAVX512]>;

// AES Instruction Templates:
//
// AES8I
// These use the same encoding as the SSE4.2 T8 and TA encodings.
class AES8I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[NoAVX, HasAES]>;

class AESAI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[NoAVX, HasAES]>;

// PCLMUL Instruction Templates
class PCLMULIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag>pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD;

// FMA3 Instruction Templates
class FMA3<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, T8PD,
        VEX_4V, FMASC, Requires<[HasFMA, NoFMA4, NoVLX]>;
class FMA3S<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, T8PD,
        VEX_4V, FMASC, Requires<[HasFMA, NoFMA4, NoAVX512]>;
class FMA3S_Int<bits<8> o, Format F, dag outs, dag ins, string asm,
                list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, T8PD,
        VEX_4V, FMASC, Requires<[HasFMA, NoAVX512]>;

// FMA4 Instruction Templates
class FMA4<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern>, TAPD,
        VEX_4V, FMASC, Requires<[HasFMA4, NoVLX]>;
class FMA4S<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag>pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern>, TAPD,
        VEX_4V, FMASC, Requires<[HasFMA4, NoAVX512]>;
class FMA4S_Int<bits<8> o, Format F, dag outs, dag ins, string asm,
                list<dag>pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern>, TAPD,
        VEX_4V, FMASC, Requires<[HasFMA4]>;

// XOP 2, 3 and 4 Operand Instruction Template
class IXOP<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>,
         XOP9, Requires<[HasXOP]>;

// XOP 2 and 3 Operand Instruction Templates with imm byte
class IXOPi8<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedDouble>,
         XOP8, Requires<[HasXOP]>;
// XOP 4 Operand Instruction Templates with imm byte
class IXOPi8Reg<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern, SSEPackedDouble>,
         XOP8, Requires<[HasXOP]>;

//  XOP 5 operand instruction (VEX encoding!)
class IXOP5<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        VEX_4V, Requires<[HasXOP]>;

// X86-64 Instruction templates...
//

class RI<bits<8> o, Format F, dag outs, dag ins, string asm,
         list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, REX_W;
class RIi8 <bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, REX_W;
class RIi16 <bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii16<o, F, outs, ins, asm, pattern>, REX_W;
class RIi32 <bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii32<o, F, outs, ins, asm, pattern>, REX_W;
class RIi32S <bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii32S<o, F, outs, ins, asm, pattern>, REX_W;
class RIi64<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii64<o, F, outs, ins, asm, pattern>, REX_W;

class RS2I<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : S2I<o, F, outs, ins, asm, pattern>, REX_W;
class VRS2I<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : VS2I<o, F, outs, ins, asm, pattern>, REX_W;

// MMX Instruction templates
//

// MMXI   - MMX instructions with TB prefix.
// MMXI32 - MMX instructions with TB prefix valid only in 32 bit mode.
// MMXI64 - MMX instructions with TB prefix valid only in 64 bit mode.
// MMX2I  - MMX / SSE2 instructions with PD prefix.
// MMXIi8 - MMX instructions with ImmT == Imm8 and PS prefix.
// MMXIi8 - MMX instructions with ImmT == Imm8 and PS prefix.
// MMXID  - MMX instructions with XD prefix.
// MMXIS  - MMX instructions with XS prefix.
class MMXI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PS, Requires<[HasMMX]>;
class MMXI32<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PS, Requires<[HasMMX,Not64BitMode]>;
class MMXI64<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PS, Requires<[HasMMX,In64BitMode]>;
class MMXRI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PS, REX_W,
        Requires<[HasMMX,In64BitMode]>;
class MMX2I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PD, Requires<[HasMMX]>;
class MMXIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, PS, Requires<[HasMMX]>;
class MMXID<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XD, Requires<[HasMMX]>;
class MMXIS<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[HasMMX]>;
