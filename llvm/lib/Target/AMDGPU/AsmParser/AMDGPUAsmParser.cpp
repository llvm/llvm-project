//===-- AMDGPUAsmParser.cpp - Parse SI asm to MCInst instructions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDKernelCodeT.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDKernelCodeTUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

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
    ImmTyOMod,
    ImmTyDppCtrl,
    ImmTyDppRowMask,
    ImmTyDppBankMask,
    ImmTyDppBoundCtrl,
    ImmTyDMask,
    ImmTyUNorm,
    ImmTyDA,
    ImmTyR128,
    ImmTyLWE,
  };

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct ImmOp {
    bool IsFPImm;
    ImmTy Type;
    int64_t Val;
    int Modifiers;
  };

  struct RegOp {
    unsigned RegNo;
    int Modifiers;
    const MCRegisterInfo *TRI;
    const MCSubtargetInfo *STI;
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
    Inst.addOperand(MCOperand::createReg(AMDGPU::getMCReg(getReg(), *Reg.STI)));
  }

  void addRegOrImmOperands(MCInst &Inst, unsigned N) const {
    if (isRegKind())
      addRegOperands(Inst, N);
    else
      addImmOperands(Inst, N);
  }

  void addRegOrImmWithInputModsOperands(MCInst &Inst, unsigned N) const {
    if (isRegKind()) {
      Inst.addOperand(MCOperand::createImm(Reg.Modifiers));
      addRegOperands(Inst, N);
    } else {
      Inst.addOperand(MCOperand::createImm(Imm.Modifiers));
      addImmOperands(Inst, N);
    }
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

    return Token.endswith("_e32") || Token.endswith("_e64") ||
      Token.endswith("_dpp");
  }

  bool isToken() const override {
    return Kind == Token;
  }

  bool isImm() const override {
    return Kind == Immediate;
  }

  bool isInlinableImm() const {
    if (!isImm() || Imm.Type != AMDGPUOperand::ImmTyNone /* Only plain
      immediates are inlinable (e.g. "clamp" attribute is not) */ )
      return false;
    // TODO: We should avoid using host float here. It would be better to
    // check the float bit values which is what a few other places do.
    // We've had bot failures before due to weird NaN support on mips hosts.
    const float F = BitsToFloat(Imm.Val);
    // TODO: Add 1/(2*pi) for VI
    return (Imm.Val <= 64 && Imm.Val >= -16) ||
           (F == 0.0 || F == 0.5 || F == -0.5 || F == 1.0 || F == -1.0 ||
           F == 2.0 || F == -2.0 || F == 4.0 || F == -4.0);
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
    return Kind == Register && Reg.Modifiers == 0;
  }

  bool isRegOrImmWithInputMods() const {
    return Kind == Register || isInlinableImm();
  }

  bool isImmTy(ImmTy ImmT) const {
    return isImm() && Imm.Type == ImmT;
  }

  bool isClamp() const {
    return isImmTy(ImmTyClamp);
  }

  bool isOMod() const {
    return isImmTy(ImmTyOMod);
  }

  bool isImmModifier() const {
    return Kind == Immediate && Imm.Type != ImmTyNone;
  }

  bool isDMask() const {
    return isImmTy(ImmTyDMask);
  }

  bool isUNorm() const { return isImmTy(ImmTyUNorm); }
  bool isDA() const { return isImmTy(ImmTyDA); }
  bool isR128() const { return isImmTy(ImmTyUNorm); }
  bool isLWE() const { return isImmTy(ImmTyLWE); }

  bool isMod() const {
    return isClamp() || isOMod();
  }

  bool isGDS() const { return isImmTy(ImmTyGDS); }
  bool isGLC() const { return isImmTy(ImmTyGLC); }
  bool isSLC() const { return isImmTy(ImmTySLC); }
  bool isTFE() const { return isImmTy(ImmTyTFE); }

  bool isBankMask() const {
    return isImmTy(ImmTyDppBankMask);
  }

  bool isRowMask() const {
    return isImmTy(ImmTyDppRowMask);
  }

  bool isBoundCtrl() const {
    return isImmTy(ImmTyDppBoundCtrl);
  }

  void setModifiers(unsigned Mods) {
    assert(isReg() || (isImm() && Imm.Modifiers == 0));
    if (isReg())
      Reg.Modifiers = Mods;
    else
      Imm.Modifiers = Mods;
  }

  bool hasModifiers() const {
    assert(isRegKind() || isImm());
    return isRegKind() ? Reg.Modifiers != 0 : Imm.Modifiers != 0;
  }

  unsigned getReg() const override {
    return Reg.RegNo;
  }

  bool isRegOrImm() const {
    return isReg() || isImm();
  }

  bool isRegClass(unsigned RCID) const {
    return isReg() && Reg.TRI->getRegClass(RCID).contains(getReg());
  }

  bool isSCSrc32() const {
    return isInlinableImm() || isRegClass(AMDGPU::SReg_32RegClassID);
  }

  bool isSCSrc64() const {
    return isInlinableImm() || isRegClass(AMDGPU::SReg_64RegClassID);
  }

  bool isSSrc32() const {
    return isImm() || isSCSrc32();
  }

  bool isSSrc64() const {
    // TODO: Find out how SALU supports extension of 32-bit literals to 64 bits.
    // See isVSrc64().
    return isImm() || isSCSrc64();
  }

  bool isVCSrc32() const {
    return isInlinableImm() || isRegClass(AMDGPU::VS_32RegClassID);
  }

  bool isVCSrc64() const {
    return isInlinableImm() || isRegClass(AMDGPU::VS_64RegClassID);
  }

  bool isVSrc32() const {
    return isImm() || isVCSrc32();
  }

  bool isVSrc64() const {
    // TODO: Check if the 64-bit value (coming from assembly source) can be
    // narrowed to 32 bits (in the instruction stream). That require knowledge
    // of instruction type (unsigned/signed, floating or "untyped"/B64),
    // see [AMD GCN3 ISA 6.3.1].
    // TODO: How 64-bit values are formed from 32-bit literals in _B64 insns?
    return isImm() || isVCSrc64();
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
      if (Imm.Type != AMDGPUOperand::ImmTyNone)
        OS << getImm();
      else
        OS << '<' << getImm() << " mods: " << Imm.Modifiers << '>';
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
    Op->Imm.Modifiers = 0;
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
                                                  const MCSubtargetInfo *STI,
                                                  bool ForceVOP3) {
    auto Op = llvm::make_unique<AMDGPUOperand>(Register);
    Op->Reg.RegNo = RegNo;
    Op->Reg.TRI = TRI;
    Op->Reg.STI = STI;
    Op->Reg.Modifiers = 0;
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
  bool isDPPCtrl() const;
};

class AMDGPUAsmParser : public MCTargetAsmParser {
  const MCInstrInfo &MII;
  MCAsmParser &Parser;

  unsigned ForcedEncodingSize;

  bool isSI() const {
    return AMDGPU::isSI(getSTI());
  }

  bool isCI() const {
    return AMDGPU::isCI(getSTI());
  }

  bool isVI() const {
    return AMDGPU::isVI(getSTI());
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
  bool ParseDirectiveAMDGPUHsaModuleGlobal();
  bool ParseDirectiveAMDGPUHsaProgramGlobal();
  bool ParseSectionDirectiveHSADataGlobalAgent();
  bool ParseSectionDirectiveHSADataGlobalProgram();
  bool ParseSectionDirectiveHSARodataReadonlyAgent();

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

  std::unique_ptr<AMDGPUOperand> parseRegister();
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
  void cvtFlatAtomic(MCInst &Inst, const OperandVector &Operands);

  void cvtMubuf(MCInst &Inst, const OperandVector &Operands);
  OperandMatchResultTy parseOffset(OperandVector &Operands);
  OperandMatchResultTy parseMubufOptionalOps(OperandVector &Operands);
  OperandMatchResultTy parseGLC(OperandVector &Operands);
  OperandMatchResultTy parseSLC(OperandVector &Operands);
  OperandMatchResultTy parseTFE(OperandVector &Operands);

  OperandMatchResultTy parseDMask(OperandVector &Operands);
  OperandMatchResultTy parseUNorm(OperandVector &Operands);
  OperandMatchResultTy parseDA(OperandVector &Operands);
  OperandMatchResultTy parseR128(OperandVector &Operands);
  OperandMatchResultTy parseLWE(OperandVector &Operands);

  void cvtId(MCInst &Inst, const OperandVector &Operands);
  void cvtVOP3_2_mod(MCInst &Inst, const OperandVector &Operands);
  void cvtVOP3_2_nomod(MCInst &Inst, const OperandVector &Operands);
  void cvtVOP3_only(MCInst &Inst, const OperandVector &Operands);
  void cvtVOP3(MCInst &Inst, const OperandVector &Operands);

  void cvtMIMG(MCInst &Inst, const OperandVector &Operands);
  void cvtMIMGAtomic(MCInst &Inst, const OperandVector &Operands);
  OperandMatchResultTy parseVOP3OptionalOps(OperandVector &Operands);

  OperandMatchResultTy parseDPPCtrlOps(OperandVector &Operands);
  OperandMatchResultTy parseDPPOptionalOps(OperandVector &Operands);
  void cvtDPP_mod(MCInst &Inst, const OperandVector &Operands);
  void cvtDPP_nomod(MCInst &Inst, const OperandVector &Operands);
  void cvtDPP(MCInst &Inst, const OperandVector &Operands, bool HasMods);
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
  auto R = parseRegister();
  if (!R) return true;
  assert(R->isReg());
  RegNo = R->getReg();
  StartLoc = R->getStartLoc();
  EndLoc = R->getEndLoc();
  return false;
}

std::unique_ptr<AMDGPUOperand> AMDGPUAsmParser::parseRegister() {
  const AsmToken &Tok = Parser.getTok();
  SMLoc StartLoc = Tok.getLoc();
  SMLoc EndLoc = Tok.getEndLoc();
  const MCRegisterInfo *TRI = getContext().getRegisterInfo();

  StringRef RegName = Tok.getString();
  unsigned RegNo = getRegForName(RegName);

  if (RegNo) {
    Parser.Lex();
    if (!subtargetHasRegister(*TRI, RegNo))
      return nullptr;
    return AMDGPUOperand::CreateReg(RegNo, StartLoc, EndLoc,
                                    TRI, &getSTI(), false);
  }

  // Match vgprs and sgprs
  if (RegName[0] != 's' && RegName[0] != 'v')
    return nullptr;

  bool IsVgpr = RegName[0] == 'v';
  unsigned RegWidth;
  unsigned RegIndexInClass;
  if (RegName.size() > 1) {
    // We have a 32-bit register
    RegWidth = 1;
    if (RegName.substr(1).getAsInteger(10, RegIndexInClass))
      return nullptr;
    Parser.Lex();
  } else {
    // We have a register greater than 32-bits.

    int64_t RegLo, RegHi;
    Parser.Lex();
    if (getLexer().isNot(AsmToken::LBrac))
      return nullptr;

    Parser.Lex();
    if (getParser().parseAbsoluteExpression(RegLo))
      return nullptr;

    if (getLexer().isNot(AsmToken::Colon))
      return nullptr;

    Parser.Lex();
    if (getParser().parseAbsoluteExpression(RegHi))
      return nullptr;

    if (getLexer().isNot(AsmToken::RBrac))
      return nullptr;

    Parser.Lex();
    RegWidth = (RegHi - RegLo) + 1;
    if (IsVgpr) {
      // VGPR registers aren't aligned.
      RegIndexInClass = RegLo;
    } else {
      // SGPR registers are aligned.  Max alignment is 4 dwords.
      unsigned Size = std::min(RegWidth, 4u);
      if (RegLo % Size != 0)
        return nullptr;

      RegIndexInClass = RegLo / Size;
    }
  }

  int RCID = getRegClass(IsVgpr, RegWidth);
  if (RCID == -1)
    return nullptr;

  const MCRegisterClass RC = TRI->getRegClass(RCID);
  if (RegIndexInClass >= RC.getNumRegs())
    return nullptr;

  RegNo = RC.getRegister(RegIndexInClass);
  if (!subtargetHasRegister(*TRI, RegNo))
    return nullptr;

  return AMDGPUOperand::CreateReg(RegNo, StartLoc, EndLoc,
                                  TRI, &getSTI(), false);
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
  SmallString<40> ErrStr;
  raw_svector_ostream Err(ErrStr);
  if (!parseAmdKernelCodeField(ID, getLexer(), Header, Err)) {
    return TokError(Err.str());
  }
  Lex();
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

bool AMDGPUAsmParser::ParseDirectiveAMDGPUHsaModuleGlobal() {
  if (getLexer().isNot(AsmToken::Identifier))
    return TokError("expected symbol name");

  StringRef GlobalName = Parser.getTok().getIdentifier();

  getTargetStreamer().EmitAMDGPUHsaModuleScopeGlobal(GlobalName);
  Lex();
  return false;
}

bool AMDGPUAsmParser::ParseDirectiveAMDGPUHsaProgramGlobal() {
  if (getLexer().isNot(AsmToken::Identifier))
    return TokError("expected symbol name");

  StringRef GlobalName = Parser.getTok().getIdentifier();

  getTargetStreamer().EmitAMDGPUHsaProgramScopeGlobal(GlobalName);
  Lex();
  return false;
}

bool AMDGPUAsmParser::ParseSectionDirectiveHSADataGlobalAgent() {
  getParser().getStreamer().SwitchSection(
      AMDGPU::getHSADataGlobalAgentSection(getContext()));
  return false;
}

bool AMDGPUAsmParser::ParseSectionDirectiveHSADataGlobalProgram() {
  getParser().getStreamer().SwitchSection(
      AMDGPU::getHSADataGlobalProgramSection(getContext()));
  return false;
}

bool AMDGPUAsmParser::ParseSectionDirectiveHSARodataReadonlyAgent() {
  getParser().getStreamer().SwitchSection(
      AMDGPU::getHSARodataReadonlyAgentSection(getContext()));
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

  if (IDVal == ".amdgpu_hsa_module_global")
    return ParseDirectiveAMDGPUHsaModuleGlobal();

  if (IDVal == ".amdgpu_hsa_program_global")
    return ParseDirectiveAMDGPUHsaProgramGlobal();

  if (IDVal == ".hsadata_global_agent")
    return ParseSectionDirectiveHSADataGlobalAgent();

  if (IDVal == ".hsadata_global_program")
    return ParseSectionDirectiveHSADataGlobalProgram();

  if (IDVal == ".hsarodata_readonly_agent")
    return ParseSectionDirectiveHSARodataReadonlyAgent();

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
    if (Op.isImm() && Op.hasModifiers())
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
  if (ResTy == MatchOperand_Success || ResTy == MatchOperand_ParseFail||
      getLexer().is(AsmToken::EndOfStatement))
    return ResTy;

  bool Negate = false, Abs = false, Abs2 = false;

  if (getLexer().getKind()== AsmToken::Minus) {
    Parser.Lex();
    Negate = true;
  }

  if (getLexer().getKind() == AsmToken::Identifier && Parser.getTok().getString() == "abs") {
    Parser.Lex();
    Abs2 = true;
    if (getLexer().isNot(AsmToken::LParen)) {
      Error(Parser.getTok().getLoc(), "expected left paren after abs");
      return MatchOperand_ParseFail;
    }
    Parser.Lex();
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
      if (auto R = parseRegister()) {
        unsigned Modifiers = 0;

        if (Negate)
          Modifiers |= 0x1;

        if (Abs) {
          if (getLexer().getKind() != AsmToken::Pipe)
            return MatchOperand_ParseFail;
          Parser.Lex();
          Modifiers |= 0x2;
        }
        if (Abs2) {
          if (getLexer().isNot(AsmToken::RParen)) {
            return MatchOperand_ParseFail;
          }
          Parser.Lex();
          Modifiers |= 0x2;
        }
        assert(R->isReg());
        R->Reg.IsForcedVOP3 = isForcedVOP3();
        if (Modifiers) {
          R->setModifiers(Modifiers);
        }
        Operands.push_back(std::move(R));
      } else {
        ResTy = parseVOP3OptionalOps(Operands);
        if (ResTy == MatchOperand_NoMatch) {
          const auto &Tok = Parser.getTok();
          Operands.push_back(AMDGPUOperand::CreateToken(Tok.getString(),
                                                        Tok.getLoc()));
          Parser.Lex();
        }
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

typedef std::map<enum AMDGPUOperand::ImmTy, unsigned> OptionalImmIndexMap;

void addOptionalImmOperand(MCInst& Inst, const OperandVector& Operands,
                           OptionalImmIndexMap& OptionalIdx,
                           enum AMDGPUOperand::ImmTy ImmT, int64_t Default = 0) {
  auto i = OptionalIdx.find(ImmT);
  if (i != OptionalIdx.end()) {
    unsigned Idx = i->second;
    ((AMDGPUOperand &)*Operands[Idx]).addImmOperands(Inst, 1);
  } else {
    Inst.addOperand(MCOperand::createImm(Default));
  }
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

    bool DefaultValue = (Value == Op.Default);

    if (Op.ConvertResult && !Op.ConvertResult(Value)) {
      return MatchOperand_ParseFail;
    }

    if (!DefaultValue) {
      Operands.push_back(AMDGPUOperand::CreateImm(Value, S, Op.Type));
    }
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

  OptionalImmIndexMap OptionalIdx;

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

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDSOffset0);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDSOffset1);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGDS);

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

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOffset);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGDS);

  if (!GDSOnly) {
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGDS);
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
    CntMask = 0xf;
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
  // lgkmcnt [11:8]
  int64_t CntVal = 0xf7f;
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
  OptionalImmIndexMap OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    OptionalIdx[Op.getImmTy()] = i;
  }
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyTFE);
}


void AMDGPUAsmParser::cvtFlatAtomic(MCInst &Inst,
                               const OperandVector &Operands) {
  OptionalImmIndexMap OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    // Handle 'glc' token for flat atomics.
    if (Op.isToken()) {
      continue;
    }

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyTFE);
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
  return isImmTy(ImmTyOffset) && isUInt<12>(getImm());
}

void AMDGPUAsmParser::cvtMubuf(MCInst &Inst,
                               const OperandVector &Operands) {
  OptionalImmIndexMap OptionalIdx;

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

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOffset);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyTFE);
}

//===----------------------------------------------------------------------===//
// mimg
//===----------------------------------------------------------------------===//

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDMask(OperandVector &Operands) {
  return parseIntWithPrefix("dmask", Operands, AMDGPUOperand::ImmTyDMask);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseUNorm(OperandVector &Operands) {
  return parseNamedBit("unorm", Operands, AMDGPUOperand::ImmTyUNorm);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDA(OperandVector &Operands) {
  return parseNamedBit("da", Operands, AMDGPUOperand::ImmTyDA);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseR128(OperandVector &Operands) {
  return parseNamedBit("r128", Operands, AMDGPUOperand::ImmTyR128);
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseLWE(OperandVector &Operands) {
  return parseNamedBit("lwe", Operands, AMDGPUOperand::ImmTyLWE);
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

  if (Operands.size() >= 2) {
    AMDGPUOperand &DstOp = ((AMDGPUOperand&)*Operands[1]);

    if (DstOp.isRegClass(AMDGPU::SGPR_64RegClassID))
      return true;
  }

  if (Operands.size() >= 5)
    return true;

  if (Operands.size() > 3) {
    AMDGPUOperand &Src1Op = ((AMDGPUOperand&)*Operands[3]);
    if (Src1Op.isRegClass(AMDGPU::SReg_32RegClassID) ||
        Src1Op.isRegClass(AMDGPU::SReg_64RegClassID))
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
        if ((Op.isReg() || Op.isImm()) && !Op.hasModifiers())
          Op.setModifiers(0);
      }
    }
    return Res;
  }
  return MatchOperand_NoMatch;
}

void AMDGPUAsmParser::cvtId(MCInst &Inst, const OperandVector &Operands) {
  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }
  for (unsigned E = Operands.size(); I != E; ++I)
    ((AMDGPUOperand &)*Operands[I]).addRegOrImmOperands(Inst, 1);
}

void AMDGPUAsmParser::cvtVOP3_2_mod(MCInst &Inst, const OperandVector &Operands) {
  uint64_t TSFlags = MII.get(Inst.getOpcode()).TSFlags;
  if (TSFlags & SIInstrFlags::VOP3) {
    cvtVOP3(Inst, Operands);
  } else {
    cvtId(Inst, Operands);
  }
}

void AMDGPUAsmParser::cvtVOP3_2_nomod(MCInst &Inst, const OperandVector &Operands) {
  if (operandsHaveModifiers(Operands)) {
    cvtVOP3(Inst, Operands);
  } else {
    cvtId(Inst, Operands);
  }
}

void AMDGPUAsmParser::cvtVOP3_only(MCInst &Inst, const OperandVector &Operands) {
  cvtVOP3(Inst, Operands);
}

void AMDGPUAsmParser::cvtVOP3(MCInst &Inst, const OperandVector &Operands) {
  OptionalImmIndexMap OptionalIdx;
  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }

  for (unsigned E = Operands.size(); I != E; ++I) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);
    if (Op.isRegOrImmWithInputMods()) {
      Op.addRegOrImmWithInputModsOperands(Inst, 2);
    } else if (Op.isImm()) {
      OptionalIdx[Op.getImmTy()] = I;
    } else {
      assert(false);
    }
  }

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyClamp);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOMod);
}

void AMDGPUAsmParser::cvtMIMG(MCInst &Inst, const OperandVector &Operands) {
  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }

  OptionalImmIndexMap OptionalIdx;

  for (unsigned E = Operands.size(); I != E; ++I) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);

    // Add the register arguments
    if (Op.isRegOrImm()) {
      Op.addRegOrImmOperands(Inst, 1);
      continue;
    } else if (Op.isImmModifier()) {
      OptionalIdx[Op.getImmTy()] = I;
    } else {
      assert(false);
    }
  }

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDMask);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyUNorm);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDA);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyR128);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyTFE);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyLWE);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySLC);
}

void AMDGPUAsmParser::cvtMIMGAtomic(MCInst &Inst, const OperandVector &Operands) {
  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }

  // Add src, same as dst
  ((AMDGPUOperand &)*Operands[I]).addRegOperands(Inst, 1);

  OptionalImmIndexMap OptionalIdx;

  for (unsigned E = Operands.size(); I != E; ++I) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);

    // Add the register arguments
    if (Op.isRegOrImm()) {
      Op.addRegOrImmOperands(Inst, 1);
      continue;
    } else if (Op.isImmModifier()) {
      OptionalIdx[Op.getImmTy()] = I;
    } else {
      assert(false);
    }
  }

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDMask);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyUNorm);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDA);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyR128);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyTFE);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyLWE);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySLC);
}

//===----------------------------------------------------------------------===//
// dpp
//===----------------------------------------------------------------------===//

bool AMDGPUOperand::isDPPCtrl() const {
  bool result = isImm() && getImmTy() == ImmTyDppCtrl && isUInt<9>(getImm());
  if (result) {
    int64_t Imm = getImm();
    return ((Imm >= 0x000) && (Imm <= 0x0ff)) ||
           ((Imm >= 0x101) && (Imm <= 0x10f)) ||
           ((Imm >= 0x111) && (Imm <= 0x11f)) ||
           ((Imm >= 0x121) && (Imm <= 0x12f)) ||
           (Imm == 0x130) ||
           (Imm == 0x134) ||
           (Imm == 0x138) ||
           (Imm == 0x13c) ||
           (Imm == 0x140) ||
           (Imm == 0x141) ||
           (Imm == 0x142) ||
           (Imm == 0x143);
  }
  return false;
}

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDPPCtrlOps(OperandVector &Operands) {
  // ToDo: use same syntax as sp3 for dpp_ctrl
  SMLoc S = Parser.getTok().getLoc();
  StringRef Prefix;
  int64_t Int;

  if (getLexer().getKind() == AsmToken::Identifier) {
    Prefix = Parser.getTok().getString();
  } else {
    return MatchOperand_NoMatch;
  }

  if (Prefix == "row_mirror") {
    Int = 0x140;
  } else if (Prefix == "row_half_mirror") {
    Int = 0x141;
  } else {
    Parser.Lex();
    if (getLexer().isNot(AsmToken::Colon))
      return MatchOperand_ParseFail;

    if (Prefix == "quad_perm") {
      // quad_perm:[%d,%d,%d,%d]
      Parser.Lex();
      if (getLexer().isNot(AsmToken::LBrac))
        return MatchOperand_ParseFail;

      Parser.Lex();
      if (getLexer().isNot(AsmToken::Integer))
        return MatchOperand_ParseFail;
      Int = getLexer().getTok().getIntVal();

      Parser.Lex();
      if (getLexer().isNot(AsmToken::Comma))
        return MatchOperand_ParseFail;
      Parser.Lex();
      if (getLexer().isNot(AsmToken::Integer))
        return MatchOperand_ParseFail;
      Int += (getLexer().getTok().getIntVal() << 2);

      Parser.Lex();
      if (getLexer().isNot(AsmToken::Comma))
        return MatchOperand_ParseFail;
      Parser.Lex();
      if (getLexer().isNot(AsmToken::Integer))
        return MatchOperand_ParseFail;
      Int += (getLexer().getTok().getIntVal() << 4);

      Parser.Lex();
      if (getLexer().isNot(AsmToken::Comma))
        return MatchOperand_ParseFail;
      Parser.Lex();
      if (getLexer().isNot(AsmToken::Integer))
        return MatchOperand_ParseFail;
      Int += (getLexer().getTok().getIntVal() << 6);

      Parser.Lex();
      if (getLexer().isNot(AsmToken::RBrac))
        return MatchOperand_ParseFail;

    } else {
      // sel:%d
      Parser.Lex();
      if (getLexer().isNot(AsmToken::Integer))
        return MatchOperand_ParseFail;
      Int = getLexer().getTok().getIntVal();

      if (Prefix == "row_shl") {
        Int |= 0x100;
      } else if (Prefix == "row_shr") {
        Int |= 0x110;
      } else if (Prefix == "row_ror") {
        Int |= 0x120;
      } else if (Prefix == "wave_shl") {
        Int = 0x130;
      } else if (Prefix == "wave_rol") {
        Int = 0x134;
      } else if (Prefix == "wave_shr") {
        Int = 0x138;
      } else if (Prefix == "wave_ror") {
        Int = 0x13C;
      } else if (Prefix == "row_bcast") {
        if (Int == 15) {
          Int = 0x142;
        } else if (Int == 31) {
          Int = 0x143;
        }
      } else {
        return MatchOperand_NoMatch;
      }
    }
  }
  Parser.Lex(); // eat last token

  Operands.push_back(AMDGPUOperand::CreateImm(Int, S,
                                              AMDGPUOperand::ImmTyDppCtrl));
  return MatchOperand_Success;
}

static const OptionalOperand DPPOptionalOps [] = {
  {"row_mask", AMDGPUOperand::ImmTyDppRowMask, false, 0xf, nullptr},
  {"bank_mask", AMDGPUOperand::ImmTyDppBankMask, false, 0xf, nullptr},
  {"bound_ctrl", AMDGPUOperand::ImmTyDppBoundCtrl, false, -1, nullptr}
};

AMDGPUAsmParser::OperandMatchResultTy
AMDGPUAsmParser::parseDPPOptionalOps(OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  OperandMatchResultTy Res = parseOptionalOps(DPPOptionalOps, Operands);
  // XXX - sp3 use syntax "bound_ctrl:0" to indicate that bound_ctrl bit was set
  if (Res == MatchOperand_Success) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands.back());
    // If last operand was parsed as bound_ctrl we should replace it with correct value (1)
    if (Op.isImmTy(AMDGPUOperand::ImmTyDppBoundCtrl)) {
      Operands.pop_back();
      Operands.push_back(
        AMDGPUOperand::CreateImm(1, S, AMDGPUOperand::ImmTyDppBoundCtrl));
        return MatchOperand_Success;
    }
  }
  return Res;
}

void AMDGPUAsmParser::cvtDPP_mod(MCInst &Inst, const OperandVector &Operands) {
  cvtDPP(Inst, Operands, true);
}

void AMDGPUAsmParser::cvtDPP_nomod(MCInst &Inst, const OperandVector &Operands) {
  cvtDPP(Inst, Operands, false);
}

void AMDGPUAsmParser::cvtDPP(MCInst &Inst, const OperandVector &Operands,
                             bool HasMods) {
  OptionalImmIndexMap OptionalIdx;

  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }

  for (unsigned E = Operands.size(); I != E; ++I) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);
    // Add the register arguments
    if (!HasMods && Op.isReg()) {
      Op.addRegOperands(Inst, 1);
    } else if (HasMods && Op.isRegOrImmWithInputMods()) {
      Op.addRegOrImmWithInputModsOperands(Inst, 2);
    } else if (Op.isDPPCtrl()) {
      Op.addImmOperands(Inst, 1);
    } else if (Op.isImm()) {
      // Handle optional arguments
      OptionalIdx[Op.getImmTy()] = I;
    } else {
      llvm_unreachable("Invalid operand type");
    }
  }

  // ToDo: fix default values for row_mask and bank_mask
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDppRowMask, 0xf);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDppBankMask, 0xf);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDppBoundCtrl);
}


/// Force static initialization.
extern "C" void LLVMInitializeAMDGPUAsmParser() {
  RegisterMCAsmParser<AMDGPUAsmParser> A(TheAMDGPUTarget);
  RegisterMCAsmParser<AMDGPUAsmParser> B(TheGCNTarget);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "AMDGPUGenAsmMatcher.inc"
