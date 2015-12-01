//===-- AMDGPUAsmParser.cpp - Parse SI asm to MCInst instructions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "AMDKernelCodeT.h"
#include "SIDefines.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace {

struct OptionalOperand;

class AMDGPUOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Immediate,
    Register,
    Expression
  } Kind;

  SMLoc StartLoc, EndLoc;

public:
  AMDGPUOperand(enum KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  MCContext *Ctx;

  enum ImmTy {
    ImmTyNone,
    ImmTyDSOffset0,
    ImmTyDSOffset1,
    ImmTyGDS,
    ImmTyOffset,
    ImmTyGLC,
    ImmTySLC,
    ImmTyTFE,
    ImmTyClamp,
    ImmTyOMod
  };

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct ImmOp {
    bool IsFPImm;
    ImmTy Type;
    int64_t Val;
  };

  struct RegOp {
    unsigned RegNo;
    int Modifiers;
    const MCRegisterInfo *TRI;
    bool IsForcedVOP3;
  };

  union {
    TokOp Tok;
    ImmOp Imm;
    RegOp Reg;
    const MCExpr *Expr;
  };

  void addImmOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::createImm(getImm()));
  }

  StringRef getToken() const {
    return StringRef(Tok.Data, Tok.Length);
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addRegOrImmOperands(MCInst &Inst, unsigned N) const {
    if (isReg())
      addRegOperands(Inst, N);
    else
      addImmOperands(Inst, N);
  }

  void addRegWithInputModsOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::createImm(
        Reg.Modifiers == -1 ? 0 : Reg.Modifiers));
    addRegOperands(Inst, N);
  }

  void addSoppBrTargetOperands(MCInst &Inst, unsigned N) const {
    if (isImm())
      addImmOperands(Inst, N);
    else {
      assert(isExpr());
      Inst.addOperand(MCOperand::createExpr(Expr));
    }
  }

  bool defaultTokenHasSuffix() const {
    StringRef Token(Tok.Data, Tok.Length);

    return Token.endswith("_e32") || Token.endswith("_e64");
  }

  bool isToken() const override {
    return Kind == Token;
  }

  bool isImm() const override {
    return Kind == Immediate;
  }

  bool isInlineImm() const {
    float F = BitsToFloat(Imm.Val);
    // TODO: Add 0.5pi for VI
    return isImm() && ((Imm.Val <= 64 && Imm.Val >= -16) ||
           (F == 0.0 || F == 0.5 || F == -0.5 || F == 1.0 || F == -1.0 ||
           F == 2.0 || F == -2.0 || F == 4.0 || F == -4.0));
  }

  bool isDSOffset0() const {
    assert(isImm());
    return Imm.Type == ImmTyDSOffset0;
  }

  bool isDSOffset1() const {
    assert(isImm());
    return Imm.Type == ImmTyDSOffset1;
  }

  int64_t getImm() const {
    return Imm.Val;
  }

  enum ImmTy getImmTy() const {
    assert(isImm());
    return Imm.Type;
  }

  bool isRegKind() const {
    return Kind == Register;
  }

  bool isReg() const override {
    return Kind == Register && Reg.Modifiers == -1;
  }

  bool isRegWithInputMods() const {
    return Kind == Register && (Reg.IsForcedVOP3 || Reg.Modifiers != -1);
  }

  void setModifiers(unsigned Mods) {
    assert(isReg());
    Reg.Modifiers = Mods;
  }

  bool hasModifiers() const {
    assert(isRegKind());
    return Reg.Modifiers != -1;
  }

  unsigned getReg() const override {
    return Reg.RegNo;
  }

  bool isRegOrImm() const {
    return isReg() || isImm();
  }

  bool isRegClass(unsigned RCID) const {
    return Reg.TRI->getRegClass(RCID).contains(getReg());
  }

  bool isSCSrc32() const {
    return isInlineImm() || (isReg() && isRegClass(AMDGPU::SReg_32RegClassID));
  }

  bool isSSrc32() const {
    return isImm() || (isReg() && isRegClass(AMDGPU::SReg_32RegClassID));
  }

  bool isSSrc64() const {
    return isImm() || isInlineImm() ||
           (isReg() && isRegClass(AMDGPU::SReg_64RegClassID));
  }

  bool isSCSrc64() const {
    return (isReg() && isRegClass(AMDGPU::SReg_64RegClassID)) || isInlineImm();
  }

  bool isVCSrc32() const {
    return isInlineImm() || (isReg() && isRegClass(AMDGPU::VS_32RegClassID));
  }

  bool isVCSrc64() const {
    return isInlineImm() || (isReg() && isRegClass(AMDGPU::VS_64RegClassID));
  }

  bool isVSrc32() const {
    return isImm() || (isReg() && isRegClass(AMDGPU::VS_32RegClassID));
  }

  bool isVSrc64() const {
    return isImm() || (isReg() && isRegClass(AMDGPU::VS_64RegClassID));
  }

  bool isMem() const override {
    return false;
  }

  bool isExpr() const {
    return Kind == Expression;
  }

  bool isSoppBrTarget() const {
    return isExpr() || isImm();
  }

  SMLoc getStartLoc() const override {
    return StartLoc;
  }

  SMLoc getEndLoc() const override {
    return EndLoc;
  }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case Register:
      OS << "<register " << getReg() << " mods: " << Reg.Modifiers << '>';
      break;
    case Immediate:
      OS << getImm();
      break;
    case Token:
      OS << '\'' << getToken() << '\'';
      break;
    case Expression:
      OS << "<expr " << *Expr << '>';
      break;
    }
  }

  static std::unique_ptr<AMDGPUOperand> CreateImm(int64_t Val, SMLoc Loc,
                                                  enum ImmTy Type = ImmTyNone,
                                                  bool IsFPImm = false) {
    auto Op = llvm::make_unique<AMDGPUOperand>(Immediate);
    Op->Imm.Val = Val;
    Op->Imm.IsFPImm = IsFPImm;
    Op->Imm.Type = Type;
    Op->StartLoc = Loc;
    Op->EndLoc = Loc;
    return Op;
  }

  static std::unique_ptr<AMDGPUOperand> CreateToken(StringRef Str, SMLoc Loc,
                                           bool HasExplicitEncodingSize = true) {
    auto Res = llvm::make_unique<AMDGPUOperand>(Token);
    Res->Tok.Data = Str.data();
    Res->Tok.Length = Str.size();
    Res->StartLoc = Loc;
    Res->EndLoc = Loc;
    return Res;
  }

  static std::unique_ptr<AMDGPUOperand> CreateReg(unsigned RegNo, SMLoc S,
                                                  SMLoc E,
                                                  const MCRegisterInfo *TRI,
                                                  bool ForceVOP3) {
    auto Op = llvm::make_unique<AMDGPUOperand>(Register);
    Op->Reg.RegNo = RegNo;
    Op->Reg.TRI = TRI;
    Op->Reg.Modifiers = -1;
    Op->Reg.IsForcedVOP3 = ForceVOP3;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<AMDGPUOperand> CreateExpr(const class MCExpr *Expr, SMLoc S) {
    auto Op = llvm::make_unique<AMDGPUOperand>(Expression);
    Op->Expr = Expr;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  bool isDSOffset() const;
  bool isDSOffset01() const;
  bool isSWaitCnt() const;
  bool isMubufOffset() const;
  bool isSMRDOffset() const;
  bool isSMRDLiteralOffset() const;
};

class AMDGPUAsmParser : public MCTargetAsmParser {
  const MCInstrInfo &MII;
  MCAsmParser &Parser;

  unsigned ForcedEncodingSize;

  bool isSI() const {
    return STI->getFeatureBits()[AMDGPU::FeatureSouthernIslands];
  }

  bool isCI() const {
    return STI->getFeatureBits()[AMDGPU::FeatureSeaIslands];
  }

  bool isVI() const {
    return getSTI().getFeatureBits()[AMDGPU::FeatureVolcanicIslands];
  }

  bool hasSGPR102_SGPR103() const {
    return !isVI();
  }

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "AMDGPUGenAsmMatcher.inc"

  /// }

private:
  bool ParseDirectiveMajorMinor(uint32_t &Major, uint32_t &Minor);
  bool ParseDirectiveHSACodeObjectVersion();
  bool ParseDirectiveHSACodeObjectISA();
  bool ParseAMDKernelCodeTValue(StringRef ID, amd_kernel_code_t &Header);
  bool ParseDirectiveAMDKernelCodeT();
  bool ParseSectionDirectiveHSAText();
  bool subtargetHasRegister(const MCRegisterInfo &MRI, unsigned RegNo) const;
  bool ParseDirectiveAMDGPUHsaKernel();

public:
public:
  enum AMDGPUMatchResultTy {
    Match_PreferE32 = FIRST_TARGET_MATCH_RESULT_TY
  };

  AMDGPUAsmParser(const MCSubtargetInfo &STI, MCAsmParser &_Parser,
               const MCInstrInfo &MII,
               const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI), MII(MII), Parser(_Parser),
        ForcedEncodingSize(0) {
    MCAsmParserExtension::Initialize(Parser);

    if (getSTI().getFeatureBits().none()) {
      // Set default features.
      copySTI().ToggleFeature("SOUTHERN_ISLANDS");
    }

    setAvailableFeatures(ComputeAvailableFeatures(getSTI().getFeatureBits()));
  }

  AMDGPUTargetStreamer &getTargetStreamer() {
    MCTargetStreamer &TS = *getParser().getStreamer().getTargetStreamer();
    return static_cast<AMDGPUTargetStreamer &>(TS);
  }

  unsigned getForcedEncodingSize() const {
    return ForcedEncodingSize;
  }

  void setForcedEncodingSize(unsigned Size) {
    ForcedEncodingSize = Size;
  }

  bool isForcedVOP3() const {
    return ForcedEncodingSize == 64;
  }

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  unsigned checkTargetMatchPredicate(MCInst &Inst) override;
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;
  bool ParseDirective(AsmToken DirectiveID) override;
  OperandMatchResultTy parseOperand(OperandVector &Operands, StringRef Mnemonic);
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  OperandMatchResultTy parseIntWithPrefix(const char *Prefix, int64_t &Int,
                                          int64_t Default = 0);
  OperandMatchResultTy parseIntWithPrefix(const char *Prefix,
                                          OperandVector &Operands,
                                          enum AMDGPUOperand::ImmTy ImmTy =
                                                      AMDGPUOperand::ImmTyNone);
  OperandMatchResultTy parseNamedBit(const char *Name, OperandVector &Operands,
                                     enum AMDGPUOperand::ImmTy ImmTy =
                                                      AMDGPUOperand::ImmTyNone);
  OperandMatchResultTy parseOptionalOps(
                                   const ArrayRef<OptionalOperand> &OptionalOps,
                                   OperandVector &Operands);


  void cvtDSOffset01(MCInst &Inst, const OperandVector &Operands);
  void cvtDS(MCInst &Inst, const OperandVector &Operands);
  OperandMatchResultTy parseDSOptionalOps(OperandVector &Operands);
  OperandMatchResultTy parseDSOff01OptionalOps(OperandVector &Operands);
  OperandMatchResultTy parseDSOffsetOptional(OperandVector &Operands);

  bool parseCnt(int64_t &IntVal);
  OperandMatchResultTy parseSWaitCntOps(OperandVector &Operands);
  OperandMatchResultTy parseSOppBrTarget(OperandVector &Operands);

  OperandMatchResultTy parseFlatOptionalOps(OperandVector &Operands);
  OperandMatchResultTy parseFlatAtomicOptionalOps(OperandVector &Operands);
  void cvtFlat(MCInst &Inst, const OperandVector &Operands);

  void cvtMubuf(MCInst &Inst, const OperandVector &Operands);
  OperandMatchResultTy parseOffset(OperandVector &Operands);
  OperandMatchResultTy parseMubufOptionalOps(OperandVector &Operands);
  OperandMatchResultTy parseGLC(OperandVector &Operands);
  OperandMatchResultTy parseSLC(OperandVector &Operands);
  OperandMatchResultTy parseTFE(OperandVector &Operands);

  OperandMatchResultTy parseDMask(OperandVector &Operands);
  OperandMatchResultTy parseUNorm(OperandVector &Operands);
  OperandMatchResultTy parseR128(OperandVector &Operands);

  void cvtVOP3(MCInst &Inst, const OperandVector &Operands);
  OperandMatchResultTy parseVOP3OptionalOps(OperandVector &Operands);
};

struct OptionalOperand {
  const char *Name;
  AMDGPUOperand::ImmTy Type;
  bool IsBit;
  int64_t Default;
  bool (*ConvertResult)(int64_t&);
};

}

static int getRegClass(bool IsVgpr, unsigned RegWidth) {
  if (IsVgpr) {
    switch (RegWidth) {
      default: return -1;
      case 1: return AMDGPU::VGPR_32RegClassID;
      case 2: return AMDGPU::VReg_64RegClassID;
      case 3: return AMDGPU::VReg_96RegClassID;
      case 4: return AMDGPU::VReg_128RegClassID;
      case 8: return AMDGPU::VReg_256RegClassID;
      case 16: return AMDGPU::VReg_512RegClassID;
    }
  }

  switch (RegWidth) {
    default: return -1;
    case 1: return AMDGPU::SGPR_32RegClassID;
    case 2: return AMDGPU::SGPR_64RegClassID;
    case 4: return AMDGPU::SReg_128RegClassID;
    case 8: return AMDGPU::SReg_256RegClassID;
    case 16: return AMDGPU::SReg_512RegClassID;
  }
}

static unsigned getRegForName(StringRef RegName) {

  return StringSwitch<unsigned>(RegName)
    .Case("exec", AMDGPU::EXEC)
    .Case("vcc", AMDGPU::VCC)
    .Case("flat_scratch", AMDGPU::FLAT_SCR)
    .Case("m0", AMDGPU::M0)
    .Case("scc", AMDGPU::SCC)
    .Case("flat_scratch_lo", AMDGPU::FLAT_SCR_LO)
    .Case("flat_scratch_hi", AMDGPU::FLAT_SCR_HI)
    .Case("vcc_lo", AMDGPU::VCC_LO)
    .Case("vcc_hi", AMDGPU::VCC_HI)
    .Case("exec_lo", AMDGPU::EXEC_LO)
    .Case("exec_hi", AMDGPU::EXEC_HI)
    .Default(0);
}

bool AMDGPUAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  const MCRegisterInfo *TRI = getContext().getRegisterInfo();

  StringRef RegName = Tok.getString();
  RegNo = getRegForName(RegName);

  if (RegNo) {
    Parser.Lex();
    return !subtargetHasRegister(*TRI, RegNo);
  }

  // Match vgprs and sgprs
  if (RegName[0] != 's' && RegName[0] != 'v')
    return true;

  bool IsVgpr = RegName[0] == 'v';
  unsigned RegWidth;
  unsigned RegIndexInClass;
  if (RegName.size() > 1) {
    // We have a 32-bit register
    RegWidth = 1;
    if (RegName.substr(1).getAsInteger(10, RegIndexInClass))
      return true;
    Parser.Lex();
  } else {
    // We have a register greater than 32-bits.

    int64_t RegLo, RegHi;
    Parser.Lex();
    if (getLexer().isNot(AsmToken::LBrac))
      return true;

    Parser.Lex();
    if (getParser().parseAbsoluteExpression(RegLo))
      return true;

    if (getLexer().isNot(AsmToken::Colon))
      return true;

    Parser.Lex();
    if (getParser().parseAbsoluteExpression(RegHi))
      return true;

    if (getLexer().isNot(AsmToken::RBrac))
      return true;

    Parser.Lex();
    RegWidth = (RegHi - RegLo) + 1;
    if (IsVgpr) {
      // VGPR registers aren't aligned.
      RegIndexInClass = RegLo;
    } else {
      // SGPR registers are aligned.  Max alignment is 4 dwords.
      unsigned Size = std::min(RegWidth, 4u);
      if (RegLo % Size != 0)
        return true;

      RegIndexInClass = RegLo / Size;
    }
  }

  int RCID = getRegClass(IsVgpr, RegWidth);
  if (RCID == -1)
    return true;

  const MCRegisterClass RC = TRI->getRegClass(RCID);
  if (RegIndexInClass >= RC.getNumRegs())
    return true;

  RegNo = RC.getRegister(RegIndexInClass);
  return !subtargetHasRegister(*TRI, RegNo);
}

unsigned AMDGPUAsmParser::checkTargetMatchPredicate(MCInst &Inst) {

  uint64_t TSFlags = MII.get(Inst.getOpcode()).TSFlags;

  if ((getForcedEncodingSize() == 32 && (TSFlags & SIInstrFlags::VOP3)) ||
      (getForcedEncodingSize() == 64 && !(TSFlags & SIInstrFlags::VOP3)))
    return Match_InvalidOperand;

  if ((TSFlags & SIInstrFlags::VOP3) &&
      (TSFlags & SIInstrFlags::VOPAsmPrefer32Bit) &&
      getForcedEncodingSize() != 64)
    return Match_PreferE32;

  return Match_Success;
}


bool AMDGPUAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                              OperandVector &Operands,
                                              MCStreamer &Out,
                                              uint64_t &ErrorInfo,
                                              bool MatchingInlineAsm) {
  MCInst Inst;

  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm)) {
    default: break;
    case Match_Success:
      Inst.setLoc(IDLoc);
      Out.EmitInstruction(Inst, getSTI());
      return false;
    case Match_MissingFeature:
      return Error(IDLoc, "instruction not supported on this GPU");

    case Match_MnemonicFail:
      return Error(IDLoc, "unrecognized instruction mnemonic");

    case Match_InvalidOperand: {
      SMLoc ErrorLoc = IDLoc;
      if (ErrorInfo != ~0ULL) {
        if (ErrorInfo >= Operands.size()) {
          if (isForcedVOP3()) {
            // If 64-bit encoding has been forced we can end up with no
            // clamp or omod operands if none of the registers have modifiers,
            // so we need to add these to the operand list.
            AMDGPUOperand &LastOp =
                ((AMDGPUOperand &)*Operands[Operands.size() - 1]);
            if (LastOp.isRegKind() ||
               (LastOp.isImm() &&
                LastOp.getImmTy() != AMDGPUOperand::ImmTyNone)) {
              SMLoc S = Parser.getTok().getLoc();
              Operands.push_back(AMDGPUOperand::CreateImm(0, S,
                                 AMDGPUOperand::ImmTyClamp));
              Operands.push_back(AMDGPUOperand::CreateImm(0, S,
                                 AMDGPUOperand::ImmTyOMod));
              bool Res = MatchAndEmitInstruction(IDLoc, Opcode, Operands,
                                                 Out, ErrorInfo,
                                                 MatchingInlineAsm);
              if (!Res)
                return Res;
            }

          }
          return Error(IDLoc, "too few operands for instruction");
        }

        ErrorLoc = ((AMDGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
        if (ErrorLoc == SMLoc())
          ErrorLoc = IDLoc;
      }
      return Error(ErrorLoc, "invalid operand for instruction");
    }
    case Match_PreferE32:
      return Error(IDLoc, "internal error: instruction without _e64 suffix "
                          "should be encoded as e32");
  }
  llvm_unreachable("Implement any new match types added!");
}

bool AMDGPUAsmParser::ParseDirectiveMajorMinor(uint32_t &Major,
                                               uint32_t &Minor) {
  if (getLexer().isNot(AsmToken::Integer))
    return TokError("invalid major version");

  Major = getLexer().getTok().getIntVal();
  Lex();

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("minor version number required, comma expected");
  Lex();

  if (getLexer().isNot(AsmToken::Integer))
    return TokError("invalid minor version");

  Minor = getLexer().getTok().getIntVal();
  Lex();

  return false;
}

bool AMDGPUAsmParser::ParseDirectiveHSACodeObjectVersion() {

  uint32_t Major;
  uint32_t Minor;

  if (ParseDirectiveMajorMinor(Major, Minor))
    return true;

  getTargetStreamer().EmitDirectiveHSACodeObjectVersion(Major, Minor);
  return false;
}

bool AMDGPUAsmParser::ParseDirectiveHSACodeObjectISA() {

  uint32_t Major;
  uint32_t Minor;
  uint32_t Stepping;
  StringRef VendorName;
  StringRef ArchName;

  // If this directive has no arguments, then use the ISA version for the
  // targeted GPU.
  if (getLexer().is(AsmToken::EndOfStatement)) {
    AMDGPU::IsaVersion Isa = AMDGPU::getIsaVersion(getSTI().getFeatureBits());
    getTargetStreamer().EmitDirectiveHSACodeObjectISA(Isa.Major, Isa.Minor,
                                                      Isa.Stepping,
                                                      "AMD", "AMDGPU");
    return false;
  }


  if (ParseDirectiveMajorMinor(Major, Minor))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("stepping version number required, comma expected");
  Lex();

  if (getLexer().isNot(AsmToken::Integer))
    return TokError("invalid stepping version");

  Stepping = getLexer().getTok().getIntVal();
  Lex();

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("vendor name required, comma expected");
  Lex();

  if (getLexer().isNot(AsmToken::String))
    return TokError("invalid vendor name");

  VendorName = getLexer().getTok().getStringContents();
  Lex();

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("arch name required, comma expected");
  Lex();

  if (getLexer().isNot(AsmToken::String))
    return TokError("invalid arch name");

  ArchName = getLexer().getTok().getStringContents();
  Lex();

  getTargetStreamer().EmitDirectiveHSACodeObjectISA(Major, Minor, Stepping,
                                                    VendorName, ArchName);
  return false;
}

bool AMDGPUAsmParser::ParseAMDKernelCodeTValue(StringRef ID,
                                               amd_kernel_code_t &Header) {

  if (getLexer().isNot(AsmToken::Equal))
    return TokError("expected '='");
  Lex();

  if (getLexer().isNot(AsmToken::Integer))
    return TokError("amd_kernel_code_t values must be integers");

  uint64_t Value = getLexer().getTok().getIntVal();
  Lex();

  if (ID == "kernel_code_version_major")
    Header.amd_kernel_code_version_major = Value;
  else if (ID == "kernel_code_version_minor")
    Header.amd_kernel_code_version_minor = Value;
  else if (ID == "machine_kind")
    Header.amd_machine_kind = Value;
  else if (ID == "machine_version_major")
    Header.amd_machine_version_major = Value;
  else if (ID == "machine_version_minor")
    Header.amd_machine_version_minor = Value;
  else if (ID == "machine_version_stepping")
    Header.amd_machine_version_stepping = Value;
  else if (ID == "kernel_code_entry_byte_offset")
    Header.kernel_code_entry_byte_offset = Value;
  else if (ID == "kernel_code_prefetch_byte_size")
    Header.kernel_code_prefetch_byte_size = Value;
  else if (ID == "max_scratch_backing_memory_byte_size")
    Header.max_scratch_backing_memory_byte_size = Value;
  else if (ID == "compute_pgm_rsrc1_vgprs")
    Header.compute_pgm_resource_registers |= S_00B848_VGPRS(Value);
  else if (ID == "compute_pgm_rsrc1_sgprs")
    Header.compute_pgm_resource_registers |= S_00B848_SGPRS(Value);
  else if (ID == "compute_pgm_rsrc1_priority")
    Header.compute_pgm_resource_registers |= S_00B848_PRIORITY(Value);
  else if (ID == "compute_pgm_rsrc1_float_mode")
    Header.compute_pgm_resource_registers |= S_00B848_FLOAT_MODE(Value);
  else if (ID == "compute_pgm_rsrc1_priv")
    Header.compute_pgm_resource_registers |= S_00B848_PRIV(Value);
  else if (ID == "compute_pgm_rsrc1_dx10_clamp")
    Header.compute_pgm_resource_registers |= S_00B848_DX10_CLAMP(Value);
  else if (ID == "compute_pgm_rsrc1_debug_mode")
    Header.compute_pgm_resource_registers |= S_00B848_DEBUG_MODE(Value);
  else if (ID == "compute_pgm_rsrc1_ieee_mode")
    Header.compute_pgm_resource_registers |= S_00B848_IEEE_MODE(Value);
  else if (ID == "compute_pgm_rsrc2_scratch_en")
    Header.compute_pgm_resource_registers |= (S_00B84C_SCRATCH_EN(Value) << 32);
  else if (ID == "compute_pgm_rsrc2_user_sgpr")
    Header.compute_pgm_resource_registers |= (S_00B84C_USER_SGPR(Value) << 32);
  else if (ID == "compute_pgm_rsrc2_tgid_x_en")
    Header.compute_pgm_resource_registers |= (S_00B84C_TGID_X_EN(Value) << 32);
  else if (ID == "compute_pgm_rsrc2_tgid_y_en")
    Header.compute_pgm_resource_registers |= (S_00B84C_TGID_Y_EN(Value) << 32);
  else if (ID == "compute_pgm_rsrc2_tgid_z_en")
    Header.compute_pgm_resource_registers |= (S_00B84C_TGID_Z_EN(Value) << 32);
  else if (ID == "compute_pgm_rsrc2_tg_size_en")
    Header.compute_pgm_resource_registers |= (S_00B84C_TG_SIZE_EN(Value) << 32);
  else if (ID == "compute_pgm_rsrc2_tidig_comp_cnt")
    Header.compute_pgm_resource_registers |=
        (S_00B84C_TIDIG_COMP_CNT(Value) << 32);
  else if (ID == "compute_pgm_rsrc2_excp_en_msb")
    Header.compute_pgm_resource_registers |=
        (S_00B84C_EXCP_EN_MSB(Value) << 32);
  else if (ID == "compute_pgm_rsrc2_lds_size")
    Header.compute_pgm_resource_registers |= (S_00B84C_LDS_SIZE(Value) << 32);
  else if (ID == "compute_pgm_rsrc2_excp_en")
    Header.compute_pgm_resource_registers |= (S_00B84C_EXCP_EN(Value) << 32);
  else if (ID == "compute_pgm_resource_registers")
    Header.compute_pgm_resource_registers = Value;
  else if (ID == "enable_sgpr_private_segment_buffer")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT);
  else if (ID == "enable_sgpr_dispatch_ptr")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR_SHIFT);
  else if (ID == "enable_sgpr_queue_ptr")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR_SHIFT);
  else if (ID == "enable_sgpr_kernarg_segment_ptr")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT);
  else if (ID == "enable_sgpr_dispatch_id")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID_SHIFT);
  else if (ID == "enable_sgpr_flat_scratch_init")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT);
  else if (ID == "enable_sgpr_private_segment_size")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT);
  else if (ID == "enable_sgpr_grid_workgroup_count_x")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT);
  else if (ID == "enable_sgpr_grid_workgroup_count_y")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT);
  else if (ID == "enable_sgpr_grid_workgroup_count_z")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT);
  else if (ID == "enable_ordered_append_gds")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_ENABLE_ORDERED_APPEND_GDS_SHIFT);
  else if (ID == "private_element_size")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_PRIVATE_ELEMENT_SIZE_SHIFT);
  else if (ID == "is_ptr64")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_IS_PTR64_SHIFT);
  else if (ID == "is_dynamic_callstack")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_IS_DYNAMIC_CALLSTACK_SHIFT);
  else if (ID == "is_debug_enabled")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_IS_DEBUG_SUPPORTED_SHIFT);
  else if (ID == "is_xnack_enabled")
    Header.code_properties |=
        (Value << AMD_CODE_PROPERTY_IS_XNACK_SUPPORTED_SHIFT);
  else if (ID == "workitem_private_segment_byte_size")
    Header.workitem_private_segment_byte_size = Value;
  else if (ID == "workgroup_group_segment_byte_size")
    Header.workgroup_group_segment_byte_size = Value;
  else if (ID == "gds_segment_byte_size")
    Header.gds_segment_byte_size = Value;
  else if (ID == "kernarg_segment_byte_size")
    Header.kernarg_segment_byte_size = Value;
  else if (ID == "workgroup_fbarrier_count")
    Header.workgroup_fbarrier_count = Value;
  else if (ID == "wavefront_sgpr_count")
    Header.wavefront_sgpr_count = Value;
  else if (ID == "workitem_vgpr_count")
    Header.workitem_vgpr_count = Value;
  else if (ID == "reserved_vgpr_first")
    Header.reserved_vgpr_first = Value;
  else if (ID == "reserved_vgpr_count")
    Header.reserved_vgpr_count = Value;
  else if (ID == "reserved_sgpr_first")
    Header.reserved_sgpr_first = Value;
  else if (ID == "reserved_sgpr_count")
    Header.reserved_sgpr_count = Value;
  else if (ID == "debug_wavefront_private_segment_offset_sgpr")
    Header.debug_wavefront_private_segment_offset_sgpr = Value;
  else if (ID == "debug_private_segment_buffer_sgpr")
    Header.debug_private_segment_buffer_sgpr = Value;
  else if (ID == "kernarg_segment_alignment")
    Header.kernarg_segment_alignment = Value;
  else if (ID == "group_segment_alignment")
    Header.group_segment_alignment = Value;
  else if (ID == "private_segment_alignment")
    Header.private_segment_alignment = Value;
  else if (ID == "wavefront_size")
    Header.wavefront_size = Value;
  else if (ID == "call_convention")
    Header.call_convention = Value;
  else if (ID == "runtime_loader_kernel_symbol")
    Header.runtime_loader_kernel_symbol = Value;
  else
    return TokError("amd_kernel_code_t value not recognized.");

  return false;
}

bool AMDGPUAsmParser::ParseDirectiveAMDKernelCodeT() {

  amd_kernel_code_t Header;
  AMDGPU::initDefaultAMDKernelCodeT(Header, getSTI().getFeatureBits());

  while (true) {

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("amd_kernel_code_t values must begin on a new line");

    // Lex EndOfStatement.  This is in a while loop, because lexing a comment
    // will set the current token to EndOfStatement.
    while(getLexer().is(AsmToken::EndOfStatement))
      Lex();

    if (getLexer().isNot(AsmToken::Identifier))
      return TokError("expected value identifier or .end_amd_kernel_code_t");

    StringRef ID = getLexer().getTok().getIdentifier();
    Lex();

    if (ID == ".end_amd_kernel_code_t")
      break;

    if (ParseAMDKernelCodeTValue(ID, Header))
      return true;
  }

  getTargetStreamer().EmitAMDKernelCodeT(Header);

  return false;
}

bool AMDGPUAsmParser::ParseSectionDirectiveHSAText() {
  getParser().getStreamer().SwitchSection(
      AMDGPU::getHSATextSection(getContext()));
  return false;
}

bool AMDGPUAsmParser::ParseDirectiveAMDGPUHsaKernel() {
  if (getLexer().isNot(AsmToken::Identifier))
    return TokError("expected symbol name");

  StringRef KernelName = Parser.getTok().getString();

  getTargetStreamer().EmitAMDGPUSymbolType(KernelName,
                                           ELF::STT_AMDGPU_HSA_KERNEL);
  Lex();
  return false;
}

bool AMDGPUAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getString();

  if (IDVal == ".hsa_code_object_version")
    return ParseDirectiveHSACodeObjectVersion();

  if (IDVal == ".hsa_code_object_isa")
    return ParseDirectiveHSACodeObjectISA();

  if (IDVal == ".amd_kernel_code_t")
    return ParseDirectiveAMDKernelCodeT();

  if (IDVal == ".hsatext" || IDVal == ".text")
    return ParseSectionDirectiveHSAText();

  if (IDVal == ".amdgpu_hsa_kernel")
    return ParseDirectiveAMDGPUHsaKernel();

  return true;
}

bool AMDGPUAsmParser::subtargetHasRegister(const MCRegisterInfo &MRI,
                                           unsigned RegNo) const {
  if (isCI())
    return true;

  if (isSI()) {
    // No flat_scr
    switch (RegNo) {
    case AMDGPU::FLAT_SCR:
    case AMDGPU::FLAT_SCR_LO:
    case AMDGPU::FLAT_SCR_HI:
      return false;
    default:
      return true;
    }
  }

  // VI only has 102 SGPRs, so make sure we aren't trying to use the 2 more that
  // SI/CI have.
  for (MCRegAliasIterator R(AMDGPU::SGPR102_SGPR103, &MRI, true);
       R.isValid(); ++R) {
    if (*R == RegNo)
      return false;
  }

  return true;
}

static bool operandsHaveModifiers(const OperandVector &Operands) {

  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
    const AMDGPUOperand &Op = ((AMDGPUOperand&)*Operands[i]);
    if (Op.isRegKind() && Op.hasModifiers())
      return true;
    if (Op.isImm() && (Op.getImmTy() == AMDGPUOperand::ImmTyOMod ||
                       Op.getImmTy() == AMDGPUOperand::ImmTyClamp))
      return true;
  }
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {

  // Try to parse with a custom parser
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);

  // If we successfully parsed the operand or if there as an error parsing,
  // we are done.
  //
  // If we are parsing after we reach EndOfStatement then this means we
  // are appending default values to the Operands list.  This is only done
  // by custom parser, so we shouldn't continue on to the generic parsing.
  if (ResTy == MatchOperand_Success || ResTy == MatchOperand_ParseFail ||
      getLexer().is(AsmToken::EndOfStatement))
    return ResTy;

  bool Negate = false, Abs = false;
  if (getLexer().getKind()== AsmToken::Minus) {
    Parser.Lex();
    Negate = true;
  }

  if (getLexer().getKind() == AsmToken::Pipe) {
    Parser.Lex();
    Abs = true;
  }

  switch(getLexer().getKind()) {
    case AsmToken::Integer: {
      SMLoc S = Parser.getTok().getLoc();
      int64_t IntVal;
      if (getParser().parseAbsoluteExpression(IntVal))
        return MatchOperand_ParseFail;
      if (!isInt<32>(IntVal) && !isUInt<32>(IntVal)) {
        Error(S, "invalid immediate: only 32-bit values are legal");
        return MatchOperand_ParseFail;
      }

      if (Negate)
        IntVal *= -1;
      Operands.push_back(AMDGPUOperand::CreateImm(IntVal, S));
      return MatchOperand_Success;
    }
    case AsmToken::Real: {
      // FIXME: We should emit an error if a double precisions floating-point
      // value is used.  I'm not sure the best way to detect this.
      SMLoc S = Parser.getTok().getLoc();
      int64_t IntVal;
      if (getParser().parseAbsoluteExpression(IntVal))
        return MatchOperand_ParseFail;

      APFloat F((float)BitsToDouble(IntVal));
      if (Negate)
        F.changeSign();
      Operands.push_back(
          AMDGPUOperand::CreateImm(F.bitcastToAPInt().getZExtValue(), S));
      return MatchOperand_Success;
    }
    case AsmToken::Identifier: {
      SMLoc S, E;
      unsigned RegNo;
      if (!ParseRegister(RegNo, S, E)) {

        bool HasModifiers = operandsHaveModifiers(Operands);
        unsigned Modifiers = 0;

        if (Negate)
          Modifiers |= 0x1;

        if (Abs) {
          if (getLexer().getKind() != AsmToken::Pipe)
            return MatchOperand_ParseFail;
          Parser.Lex();
          Modifiers |= 0x2;
        }

        if (Modifiers && !HasModifiers) {
          // We are adding a modifier to src1 or src2 and previous sources
          // don't have modifiers, so we need to go back and empty modifers
          // for each previous source.
          for (unsigned PrevRegIdx = Operands.size() - 1; PrevRegIdx > 1;
               --PrevRegIdx) {

            AMDGPUOperand &RegOp = ((AMDGPUOperand&)*Operands[PrevRegIdx]);
            RegOp.setModifiers(0);
          }
        }


        Operands.push_back(AMDGPUOperand::CreateReg(
            RegNo, S, E, getContext().getRegisterInfo(),
            isForcedVOP3()));

        if (HasModifiers || Modifiers) {
          AMDGPUOperand &RegOp = ((AMDGPUOperand&)*Operands[Operands.size() - 1]);
          RegOp.setModifiers(Modifiers);

        }
     }  else {
      Operands.push_back(AMDGPUOperand::CreateToken(Parser.getTok().getString(),
                                                    S));
      Parser.Lex();
     }
     return MatchOperand_Success;
    }
    default:
      return MatchOperand_NoMatch;
  }
}

bool AMDGPUAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                       StringRef Name,
                                       SMLoc NameLoc, OperandVector &Operands) {

  // Clear any forced encodings from the previous instruction.
  setForcedEncodingSize(0);

  if (Name.endswith("_e64"))
    setForcedEncodingSize(64);
  else if (Name.endswith("_e32"))
    setForcedEncodingSize(32);

  // Add the instruction mnemonic
  Operands.push_back(AMDGPUOperand::CreateToken(Name, NameLoc));

  while (!getLexer().is(AsmToken::EndOfStatement)) {
    AMDGPUAsmParser::OperandMatchResultTy Res = parseOperand(Operands, Name);

    // Eat the comma or space if there is one.
    if (getLexer().is(AsmToken::Comma))
      Parser.Lex();

    switch (Res) {
      case MatchOperand_Success: break;
      case MatchOperand_ParseFail: return Error(getLexer().getLoc(),
                                                "failed parsing operand.");
      case MatchOperand_NoMatch: return Error(getLexer().getLoc(),
                                              "not a valid operand.");
    }
  }

  // Once we reach end of statement, continue parsing so we can add default
  // values for optional arguments.
  AMDGPUAsmParser::OperandMatchResultTy Res;
  while ((Res = parseOperand(Operands, Name)) != MatchOperand_NoMatch) {
    if (Res != MatchOperand_Success)
      return Error(getLexer().getLoc(), "failed parsing operand.");
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseIntWithPrefix(const char *Prefix, int64_t &Int,
                                    int64_t Default) {

  // We are at the end of the statement, and this is a default argument, so
  // use a default value.
  if (getLexer().is(AsmToken::EndOfStatement)) {
    Int = Default;
    return MatchOperand_Success;
  }

  switch(getLexer().getKind()) {
    default: return MatchOperand_NoMatch;
    case AsmToken::Identifier: {
      StringRef OffsetName = Parser.getTok().getString();
      if (!OffsetName.equals(Prefix))
        return MatchOperand_NoMatch;

      Parser.Lex();
      if (getLexer().isNot(AsmToken::Colon))
        return MatchOperand_ParseFail;

      Parser.Lex();
      if (getLexer().isNot(AsmToken::Integer))
        return MatchOperand_ParseFail;

      if (getParser().parseAbsoluteExpression(Int))
        return MatchOperand_ParseFail;
      break;
    }
  }
  return MatchOperand_Success;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseIntWithPrefix(const char *Prefix, OperandVector &Operands,
                                    enum AMDGPUOperand::ImmTy ImmTy) {

  SMLoc S = Parser.getTok().getLoc();
  int64_t Offset = 0;

  AMDGPUAsmParser::OperandMatchResultTy Res = parseIntWithPrefix(Prefix, Offset);
  if (Res != MatchOperand_Success)
    return Res;

  Operands.push_back(AMDGPUOperand::CreateImm(Offset, S, ImmTy));
  return MatchOperand_Success;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseNamedBit(const char *Name, OperandVector &Operands,
                               enum AMDGPUOperand::ImmTy ImmTy) {
  int64_t Bit = 0;
  SMLoc S = Parser.getTok().getLoc();

  // We are at the end of the statement, and this is a default argument, so
  // use a default value.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    switch(getLexer().getKind()) {
      case AsmToken::Identifier: {
        StringRef Tok = Parser.getTok().getString();
        if (Tok == Name) {
          Bit = 1;
          Parser.Lex();
        } else if (Tok.startswith("no") && Tok.endswith(Name)) {
          Bit = 0;
          Parser.Lex();
        } else {
          return MatchOperand_NoMatch;
        }
        break;
      }
      default:
        return MatchOperand_NoMatch;
    }
  }

  Operands.push_back(AMDGPUOperand::CreateImm(Bit, S, ImmTy));
  return MatchOperand_Success;
}

static bool operandsHasOptionalOp(const OperandVector &Operands,
                                  const OptionalOperand &OOp) {
  for (unsigned i = 0; i < Operands.size(); i++) {
    const AMDGPUOperand &ParsedOp = ((const AMDGPUOperand &)*Operands[i]);
    if ((ParsedOp.isImm() && ParsedOp.getImmTy() == OOp.Type) ||
        (ParsedOp.isToken() && ParsedOp.getToken() == OOp.Name))
      return true;

  }
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseOptionalOps(const ArrayRef<OptionalOperand> &OptionalOps,
                                   OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  for (const OptionalOperand &Op : OptionalOps) {
    if (operandsHasOptionalOp(Operands, Op))
      continue;
    AMDGPUAsmParser::OperandMatchResultTy Res;
    int64_t Value;
    if (Op.IsBit) {
      Res = parseNamedBit(Op.Name, Operands, Op.Type);
      if (Res == MatchOperand_NoMatch)
        continue;
      return Res;
    }

    Res = parseIntWithPrefix(Op.Name, Value, Op.Default);

    if (Res == MatchOperand_NoMatch)
      continue;

    if (Res != MatchOperand_Success)
      return Res;

    if (Op.ConvertResult && !Op.ConvertResult(Value)) {
      return MatchOperand_ParseFail;
    }

    Operands.push_back(AMDGPUOperand::CreateImm(Value, S, Op.Type));
    return MatchOperand_Success;
  }
  return MatchOperand_NoMatch;
}

//===----------------------------------------------------------------------===//
// ds
//===----------------------------------------------------------------------===//

static const OptionalOperand DSOptionalOps [] = {
  {"offset",  AMDGPUOperand::ImmTyOffset, false, 0, nullptr},
  {"gds",     AMDGPUOperand::ImmTyGDS, true, 0, nullptr}
};

static const OptionalOperand DSOptionalOpsOff01 [] = {
  {"offset0", AMDGPUOperand::ImmTyDSOffset0, false, 0, nullptr},
  {"offset1", AMDGPUOperand::ImmTyDSOffset1, false, 0, nullptr},
  {"gds",     AMDGPUOperand::ImmTyGDS, true, 0, nullptr}
};

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDSOptionalOps(OperandVector &Operands) {
  return parseOptionalOps(DSOptionalOps, Operands);
}
AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDSOff01OptionalOps(OperandVector &Operands) {
  return parseOptionalOps(DSOptionalOpsOff01, Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDSOffsetOptional(OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  AMDGPUAsmParser::OperandMatchResultTy Res =
    parseIntWithPrefix("offset", Operands, AMDGPUOperand::ImmTyOffset);
  if (Res == MatchOperand_NoMatch) {
    Operands.push_back(AMDGPUOperand::CreateImm(0, S,
                       AMDGPUOperand::ImmTyOffset));
    Res = MatchOperand_Success;
  }
  return Res;
}

bool AMDGPUOperand::isDSOffset() const {
  return isImm() && isUInt<16>(getImm());
}

bool AMDGPUOperand::isDSOffset01() const {
  return isImm() && isUInt<8>(getImm());
}

void AMDGPUAsmParser::cvtDSOffset01(MCInst &Inst,
                                    const OperandVector &Operands) {

  std::map<enum AMDGPUOperand::ImmTy, unsigned> OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  unsigned Offset0Idx = OptionalIdx[AMDGPUOperand::ImmTyDSOffset0];
  unsigned Offset1Idx = OptionalIdx[AMDGPUOperand::ImmTyDSOffset1];
  unsigned GDSIdx = OptionalIdx[AMDGPUOperand::ImmTyGDS];

  ((AMDGPUOperand &)*Operands[Offset0Idx]).addImmOperands(Inst, 1); // offset0
  ((AMDGPUOperand &)*Operands[Offset1Idx]).addImmOperands(Inst, 1); // offset1
  ((AMDGPUOperand &)*Operands[GDSIdx]).addImmOperands(Inst, 1); // gds
  Inst.addOperand(MCOperand::createReg(AMDGPU::M0)); // m0
}

void AMDGPUAsmParser::cvtDS(MCInst &Inst, const OperandVector &Operands) {

  std::map<enum AMDGPUOperand::ImmTy, unsigned> OptionalIdx;
  bool GDSOnly = false;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    if (Op.isToken() && Op.getToken() == "gds") {
      GDSOnly = true;
      continue;
    }

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  unsigned OffsetIdx = OptionalIdx[AMDGPUOperand::ImmTyOffset];
  ((AMDGPUOperand &)*Operands[OffsetIdx]).addImmOperands(Inst, 1); // offset

  if (!GDSOnly) {
    unsigned GDSIdx = OptionalIdx[AMDGPUOperand::ImmTyGDS];
    ((AMDGPUOperand &)*Operands[GDSIdx]).addImmOperands(Inst, 1); // gds
  }
  Inst.addOperand(MCOperand::createReg(AMDGPU::M0)); // m0
}


//===----------------------------------------------------------------------===//
// s_waitcnt
//===----------------------------------------------------------------------===//

bool AMDGPUAsmParser::parseCnt(int64_t &IntVal) {
  StringRef CntName = Parser.getTok().getString();
  int64_t CntVal;

  Parser.Lex();
  if (getLexer().isNot(AsmToken::LParen))
    return true;

  Parser.Lex();
  if (getLexer().isNot(AsmToken::Integer))
    return true;

  if (getParser().parseAbsoluteExpression(CntVal))
    return true;

  if (getLexer().isNot(AsmToken::RParen))
    return true;

  Parser.Lex();
  if (getLexer().is(AsmToken::Amp) || getLexer().is(AsmToken::Comma))
    Parser.Lex();

  int CntShift;
  int CntMask;

  if (CntName == "vmcnt") {
    CntMask = 0xf;
    CntShift = 0;
  } else if (CntName == "expcnt") {
    CntMask = 0x7;
    CntShift = 4;
  } else if (CntName == "lgkmcnt") {
    CntMask = 0x7;
    CntShift = 8;
  } else {
    return true;
  }

  IntVal &= ~(CntMask << CntShift);
  IntVal |= (CntVal << CntShift);
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseSWaitCntOps(OperandVector &Operands) {
  // Disable all counters by default.
  // vmcnt   [3:0]
  // expcnt  [6:4]
  // lgkmcnt [10:8]
  int64_t CntVal = 0x77f;
  SMLoc S = Parser.getTok().getLoc();

  switch(getLexer().getKind()) {
    default: return MatchOperand_ParseFail;
    case AsmToken::Integer:
      // The operand can be an integer value.
      if (getParser().parseAbsoluteExpression(CntVal))
        return MatchOperand_ParseFail;
      break;

    case AsmToken::Identifier:
      do {
        if (parseCnt(CntVal))
          return MatchOperand_ParseFail;
      } while(getLexer().isNot(AsmToken::EndOfStatement));
      break;
  }
  Operands.push_back(AMDGPUOperand::CreateImm(CntVal, S));
  return MatchOperand_Success;
}

bool AMDGPUOperand::isSWaitCnt() const {
  return isImm();
}

//===----------------------------------------------------------------------===//
// sopp branch targets
//===----------------------------------------------------------------------===//

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseSOppBrTarget(OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();

  switch (getLexer().getKind()) {
    default: return MatchOperand_ParseFail;
    case AsmToken::Integer: {
      int64_t Imm;
      if (getParser().parseAbsoluteExpression(Imm))
        return MatchOperand_ParseFail;
      Operands.push_back(AMDGPUOperand::CreateImm(Imm, S));
      return MatchOperand_Success;
    }

    case AsmToken::Identifier:
      Operands.push_back(AMDGPUOperand::CreateExpr(
          MCSymbolRefExpr::create(getContext().getOrCreateSymbol(
                                  Parser.getTok().getString()), getContext()), S));
      Parser.Lex();
      return MatchOperand_Success;
  }
}

//===----------------------------------------------------------------------===//
// flat
//===----------------------------------------------------------------------===//

static const OptionalOperand FlatOptionalOps [] = {
  {"glc",    AMDGPUOperand::ImmTyGLC, true, 0, nullptr},
  {"slc",    AMDGPUOperand::ImmTySLC, true, 0, nullptr},
  {"tfe",    AMDGPUOperand::ImmTyTFE, true, 0, nullptr}
};

static const OptionalOperand FlatAtomicOptionalOps [] = {
  {"slc",    AMDGPUOperand::ImmTySLC, true, 0, nullptr},
  {"tfe",    AMDGPUOperand::ImmTyTFE, true, 0, nullptr}
};

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseFlatOptionalOps(OperandVector &Operands) {
  return parseOptionalOps(FlatOptionalOps, Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseFlatAtomicOptionalOps(OperandVector &Operands) {
  return parseOptionalOps(FlatAtomicOptionalOps, Operands);
}

void AMDGPUAsmParser::cvtFlat(MCInst &Inst,
                               const OperandVector &Operands) {
  std::map<AMDGPUOperand::ImmTy, unsigned> OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    // Handle 'glc' token which is sometimes hard-coded into the
    // asm string.  There are no MCInst operands for these.
    if (Op.isToken())
      continue;

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;

  }

  // flat atomic instructions don't have a glc argument.
  if (OptionalIdx.count(AMDGPUOperand::ImmTyGLC)) {
    unsigned GLCIdx = OptionalIdx[AMDGPUOperand::ImmTyGLC];
    ((AMDGPUOperand &)*Operands[GLCIdx]).addImmOperands(Inst, 1);
  }

  unsigned SLCIdx = OptionalIdx[AMDGPUOperand::ImmTySLC];
  unsigned TFEIdx = OptionalIdx[AMDGPUOperand::ImmTyTFE];

  ((AMDGPUOperand &)*Operands[SLCIdx]).addImmOperands(Inst, 1);
  ((AMDGPUOperand &)*Operands[TFEIdx]).addImmOperands(Inst, 1);
}

//===----------------------------------------------------------------------===//
// mubuf
//===----------------------------------------------------------------------===//

static const OptionalOperand MubufOptionalOps [] = {
  {"offset", AMDGPUOperand::ImmTyOffset, false, 0, nullptr},
  {"glc",    AMDGPUOperand::ImmTyGLC, true, 0, nullptr},
  {"slc",    AMDGPUOperand::ImmTySLC, true, 0, nullptr},
  {"tfe",    AMDGPUOperand::ImmTyTFE, true, 0, nullptr}
};

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseMubufOptionalOps(OperandVector &Operands) {
  return parseOptionalOps(MubufOptionalOps, Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseOffset(OperandVector &Operands) {
  return parseIntWithPrefix("offset", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseGLC(OperandVector &Operands) {
  return parseNamedBit("glc", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseSLC(OperandVector &Operands) {
  return parseNamedBit("slc", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseTFE(OperandVector &Operands) {
  return parseNamedBit("tfe", Operands);
}

bool AMDGPUOperand::isMubufOffset() const {
  return isImm() && isUInt<12>(getImm());
}

void AMDGPUAsmParser::cvtMubuf(MCInst &Inst,
                               const OperandVector &Operands) {
  std::map<enum AMDGPUOperand::ImmTy, unsigned> OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    // Handle the case where soffset is an immediate
    if (Op.isImm() && Op.getImmTy() == AMDGPUOperand::ImmTyNone) {
      Op.addImmOperands(Inst, 1);
      continue;
    }

    // Handle tokens like 'offen' which are sometimes hard-coded into the
    // asm string.  There are no MCInst operands for these.
    if (Op.isToken()) {
      continue;
    }
    assert(Op.isImm());

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  assert(OptionalIdx.size() == 4);

  unsigned OffsetIdx = OptionalIdx[AMDGPUOperand::ImmTyOffset];
  unsigned GLCIdx = OptionalIdx[AMDGPUOperand::ImmTyGLC];
  unsigned SLCIdx = OptionalIdx[AMDGPUOperand::ImmTySLC];
  unsigned TFEIdx = OptionalIdx[AMDGPUOperand::ImmTyTFE];

  ((AMDGPUOperand &)*Operands[OffsetIdx]).addImmOperands(Inst, 1);
  ((AMDGPUOperand &)*Operands[GLCIdx]).addImmOperands(Inst, 1);
  ((AMDGPUOperand &)*Operands[SLCIdx]).addImmOperands(Inst, 1);
  ((AMDGPUOperand &)*Operands[TFEIdx]).addImmOperands(Inst, 1);
}

//===----------------------------------------------------------------------===//
// mimg
//===----------------------------------------------------------------------===//

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDMask(OperandVector &Operands) {
  return parseIntWithPrefix("dmask", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseUNorm(OperandVector &Operands) {
  return parseNamedBit("unorm", Operands);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseR128(OperandVector &Operands) {
  return parseNamedBit("r128", Operands);
}

//===----------------------------------------------------------------------===//
// smrd
//===----------------------------------------------------------------------===//

bool AMDGPUOperand::isSMRDOffset() const {

  // FIXME: Support 20-bit offsets on VI.  We need to to pass subtarget
  // information here.
  return isImm() && isUInt<8>(getImm());
}

bool AMDGPUOperand::isSMRDLiteralOffset() const {
  // 32-bit literals are only supported on CI and we only want to use them
  // when the offset is > 8-bits.
  return isImm() && !isUInt<8>(getImm()) && isUInt<32>(getImm());
}

//===----------------------------------------------------------------------===//
// vop3
//===----------------------------------------------------------------------===//

static bool ConvertOmodMul(int64_t &Mul) {
  if (Mul != 1 && Mul != 2 && Mul != 4)
    return false;

  Mul >>= 1;
  return true;
}

static bool ConvertOmodDiv(int64_t &Div) {
  if (Div == 1) {
    Div = 0;
    return true;
  }

  if (Div == 2) {
    Div = 3;
    return true;
  }

  return false;
}

static const OptionalOperand VOP3OptionalOps [] = {
  {"clamp", AMDGPUOperand::ImmTyClamp, true, 0, nullptr},
  {"mul",   AMDGPUOperand::ImmTyOMod, false, 1, ConvertOmodMul},
  {"div",   AMDGPUOperand::ImmTyOMod, false, 1, ConvertOmodDiv},
};

static bool isVOP3(OperandVector &Operands) {
  if (operandsHaveModifiers(Operands))
    return true;

  AMDGPUOperand &DstOp = ((AMDGPUOperand&)*Operands[1]);

  if (DstOp.isReg() && DstOp.isRegClass(AMDGPU::SGPR_64RegClassID))
    return true;

  if (Operands.size() >= 5)
    return true;

  if (Operands.size() > 3) {
    AMDGPUOperand &Src1Op = ((AMDGPUOperand&)*Operands[3]);
    if (Src1Op.getReg() && (Src1Op.isRegClass(AMDGPU::SReg_32RegClassID) ||
                            Src1Op.isRegClass(AMDGPU::SReg_64RegClassID)))
      return true;
  }
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseVOP3OptionalOps(OperandVector &Operands) {

  // The value returned by this function may change after parsing
  // an operand so store the original value here.
  bool HasModifiers = operandsHaveModifiers(Operands);

  bool IsVOP3 = isVOP3(Operands);
  if (HasModifiers || IsVOP3 ||
      getLexer().isNot(AsmToken::EndOfStatement) ||
      getForcedEncodingSize() == 64) {

    AMDGPUAsmParser::OperandMatchResultTy Res =
        parseOptionalOps(VOP3OptionalOps, Operands);

    if (!HasModifiers && Res == MatchOperand_Success) {
      // We have added a modifier operation, so we need to make sure all
      // previous register operands have modifiers
      for (unsigned i = 2, e = Operands.size(); i != e; ++i) {
        AMDGPUOperand &Op = ((AMDGPUOperand&)*Operands[i]);
        if (Op.isReg())
          Op.setModifiers(0);
      }
    }
    return Res;
  }
  return MatchOperand_NoMatch;
}

void AMDGPUAsmParser::cvtVOP3(MCInst &Inst, const OperandVector &Operands) {

  unsigned i = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  if (Desc.getNumDefs() > 0) {
    ((AMDGPUOperand &)*Operands[i++]).addRegOperands(Inst, 1);
  }

  std::map<enum AMDGPUOperand::ImmTy, unsigned> OptionalIdx;

  if (operandsHaveModifiers(Operands)) {
    for (unsigned e = Operands.size(); i != e; ++i) {
      AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

      if (Op.isRegWithInputMods()) {
        ((AMDGPUOperand &)*Operands[i]).addRegWithInputModsOperands(Inst, 2);
        continue;
      }
      OptionalIdx[Op.getImmTy()] = i;
    }

    unsigned ClampIdx = OptionalIdx[AMDGPUOperand::ImmTyClamp];
    unsigned OModIdx = OptionalIdx[AMDGPUOperand::ImmTyOMod];

    ((AMDGPUOperand &)*Operands[ClampIdx]).addImmOperands(Inst, 1);
    ((AMDGPUOperand &)*Operands[OModIdx]).addImmOperands(Inst, 1);
  } else {
    for (unsigned e = Operands.size(); i != e; ++i)
      ((AMDGPUOperand &)*Operands[i]).addRegOrImmOperands(Inst, 1);
  }
}

/// Force static initialization.
extern "C" void LLVMInitializeAMDGPUAsmParser() {
  RegisterMCAsmParser<AMDGPUAsmParser> A(TheAMDGPUTarget);
  RegisterMCAsmParser<AMDGPUAsmParser> B(TheGCNTarget);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "AMDGPUGenAsmMatcher.inc"

