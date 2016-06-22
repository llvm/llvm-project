//===-- HexagonAsmParser.cpp - Parse Hexagon asm to MCInst instructions----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mcasmparser"

#include "Hexagon.h"
#include "HexagonRegisterInfo.h"
#include "HexagonTargetStreamer.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCAsmInfo.h"
#include "MCTargetDesc/HexagonMCChecker.h"
#include "MCTargetDesc/HexagonMCELFStreamer.h"
#include "MCTargetDesc/HexagonMCExpr.h"
#include "MCTargetDesc/HexagonMCShuffler.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "MCTargetDesc/HexagonShuffler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool> EnableFutureRegs("mfuture-regs",
                                      cl::desc("Enable future registers"));

static cl::opt<bool> WarnMissingParenthesis("mwarn-missing-parenthesis",
cl::desc("Warn for missing parenthesis around predicate registers"),
cl::init(true));
static cl::opt<bool> ErrorMissingParenthesis("merror-missing-parenthesis",
cl::desc("Error for missing parenthesis around predicate registers"),
cl::init(false));
static cl::opt<bool> WarnSignedMismatch("mwarn-sign-mismatch",
cl::desc("Warn for mismatching a signed and unsigned value"),
cl::init(true));
static cl::opt<bool> WarnNoncontigiousRegister("mwarn-noncontigious-register",
cl::desc("Warn for register names that arent contigious"),
cl::init(true));
static cl::opt<bool> ErrorNoncontigiousRegister("merror-noncontigious-register",
cl::desc("Error for register names that aren't contigious"),
cl::init(false));


namespace {
struct HexagonOperand;

class HexagonAsmParser : public MCTargetAsmParser {

  HexagonTargetStreamer &getTargetStreamer() {
    MCTargetStreamer &TS = *Parser.getStreamer().getTargetStreamer();
    return static_cast<HexagonTargetStreamer &>(TS);
  }

  MCAsmParser &Parser;
  MCAssembler *Assembler;
  MCInstrInfo const &MCII;
  MCInst MCB;
  bool InBrackets;

  MCAsmParser &getParser() const { return Parser; }
  MCAssembler *getAssembler() const { return Assembler; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  bool equalIsAsmAssignment() override { return false; }
  bool isLabel(AsmToken &Token) override;

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }
  bool ParseDirectiveFalign(unsigned Size, SMLoc L);

  virtual bool ParseRegister(unsigned &RegNo,
                             SMLoc &StartLoc,
                             SMLoc &EndLoc) override;
  bool ParseDirectiveSubsection(SMLoc L);
  bool ParseDirectiveValue(unsigned Size, SMLoc L);
  bool ParseDirectiveComm(bool IsLocal, SMLoc L);
  bool RegisterMatchesArch(unsigned MatchNum) const;

  bool matchBundleOptions();
  bool handleNoncontigiousRegister(bool Contigious, SMLoc &Loc);
  bool finishBundle(SMLoc IDLoc, MCStreamer &Out);
  void canonicalizeImmediates(MCInst &MCI);
  bool matchOneInstruction(MCInst &MCB, SMLoc IDLoc,
                           OperandVector &InstOperands, uint64_t &ErrorInfo,
                           bool MatchingInlineAsm);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo, bool MatchingInlineAsm) override;

  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op, unsigned Kind) override;
  void OutOfRange(SMLoc IDLoc, long long Val, long long Max);
  int processInstruction(MCInst &Inst, OperandVector const &Operands,
                         SMLoc IDLoc);

  // Check if we have an assembler and, if so, set the ELF e_header flags.
  void chksetELFHeaderEFlags(unsigned flags) {
    if (getAssembler())
      getAssembler()->setELFHeaderEFlags(flags);
  }

  unsigned matchRegister(StringRef Name);

/// @name Auto-generated Match Functions
/// {

#define GET_ASSEMBLER_HEADER
#include "HexagonGenAsmMatcher.inc"

  /// }

public:
  HexagonAsmParser(const MCSubtargetInfo &_STI, MCAsmParser &_Parser,
                   const MCInstrInfo &MII, const MCTargetOptions &Options)
    : MCTargetAsmParser(Options, _STI), Parser(_Parser),
      MCII (MII), MCB(HexagonMCInstrInfo::createBundle()), InBrackets(false) {
    setAvailableFeatures(ComputeAvailableFeatures(getSTI().getFeatureBits()));

  MCAsmParserExtension::Initialize(_Parser);

  Assembler = nullptr;
  // FIXME: need better way to detect AsmStreamer (upstream removed getKind())
  if (!Parser.getStreamer().hasRawTextSupport()) {
    MCELFStreamer *MES = static_cast<MCELFStreamer *>(&Parser.getStreamer());
    Assembler = &MES->getAssembler();
  }
  }

  bool splitIdentifier(OperandVector &Operands);
  bool parseOperand(OperandVector &Operands);
  bool parseInstruction(OperandVector &Operands);
  bool implicitExpressionLocation(OperandVector &Operands);
  bool parseExpressionOrOperand(OperandVector &Operands);
  bool parseExpression(MCExpr const *& Expr);
  virtual bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                SMLoc NameLoc, OperandVector &Operands) override
  {
    llvm_unreachable("Unimplemented");
  }
  virtual bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                AsmToken ID, OperandVector &Operands) override;

  virtual bool ParseDirective(AsmToken DirectiveID) override;
};

/// HexagonOperand - Instances of this class represent a parsed Hexagon machine
/// instruction.
struct HexagonOperand : public MCParsedAsmOperand {
  enum KindTy { Token, Immediate, Register } Kind;

  SMLoc StartLoc, EndLoc;

  struct TokTy {
    const char *Data;
    unsigned Length;
  };

  struct RegTy {
    unsigned RegNum;
  };

  struct ImmTy {
    const MCExpr *Val;
  };

  struct InstTy {
    OperandVector *SubInsts;
  };

  union {
    struct TokTy Tok;
    struct RegTy Reg;
    struct ImmTy Imm;
  };

  HexagonOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

public:
  HexagonOperand(const HexagonOperand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    switch (Kind) {
    case Register:
      Reg = o.Reg;
      break;
    case Immediate:
      Imm = o.Imm;
      break;
    case Token:
      Tok = o.Tok;
      break;
    }
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }

  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }

  unsigned getReg() const {
    assert(Kind == Register && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  bool isToken() const { return Kind == Token; }
  bool isImm() const { return Kind == Immediate; }
  bool isMem() const { llvm_unreachable("No isMem"); }
  bool isReg() const { return Kind == Register; }

  bool CheckImmRange(int immBits, int zeroBits, bool isSigned,
                     bool isRelocatable, bool Extendable) const {
    if (Kind == Immediate) {
      const MCExpr *myMCExpr = &HexagonMCInstrInfo::getExpr(*getImm());
      if (HexagonMCInstrInfo::mustExtend(*Imm.Val) && !Extendable)
        return false;
      int64_t Res;
      if (myMCExpr->evaluateAsAbsolute(Res)) {
        int bits = immBits + zeroBits;
        // Field bit range is zerobits + bits
        // zeroBits must be 0
        if (Res & ((1 << zeroBits) - 1))
          return false;
        if (isSigned) {
          if (Res < (1LL << (bits - 1)) && Res >= -(1LL << (bits - 1)))
            return true;
        } else {
          if (bits == 64)
            return true;
          if (Res >= 0)
            return ((uint64_t)Res < (uint64_t)(1ULL << bits)) ? true : false;
          else {
            const int64_t high_bit_set = 1ULL << 63;
            const uint64_t mask = (high_bit_set >> (63 - bits));
            return (((uint64_t)Res & mask) == mask) ? true : false;
          }
        }
      } else if (myMCExpr->getKind() == MCExpr::SymbolRef && isRelocatable)
        return true;
      else if (myMCExpr->getKind() == MCExpr::Binary ||
               myMCExpr->getKind() == MCExpr::Unary)
        return true;
    }
    return false;
  }

  bool isf32Ext() const { return false; }
  bool iss32Imm() const { return CheckImmRange(32, 0, true, true, false); }
  bool iss23_2Imm() const { return CheckImmRange(23, 2, true, true, false); }
  bool iss8Imm() const { return CheckImmRange(8, 0, true, false, false); }
  bool iss8Imm64() const { return CheckImmRange(8, 0, true, true, false); }
  bool iss7Imm() const { return CheckImmRange(7, 0, true, false, false); }
  bool iss6Imm() const { return CheckImmRange(6, 0, true, false, false); }
  bool iss4Imm() const { return CheckImmRange(4, 0, true, false, false); }
  bool iss4_0Imm() const { return CheckImmRange(4, 0, true, false, false); }
  bool iss4_1Imm() const { return CheckImmRange(4, 1, true, false, false); }
  bool iss4_2Imm() const { return CheckImmRange(4, 2, true, false, false); }
  bool iss4_3Imm() const { return CheckImmRange(4, 3, true, false, false); }
  bool iss4_6Imm() const { return CheckImmRange(4, 0, true, false, false); }
  bool iss3_6Imm() const { return CheckImmRange(3, 0, true, false, false); }
  bool iss3Imm() const { return CheckImmRange(3, 0, true, false, false); }

  bool isu64Imm() const { return CheckImmRange(64, 0, false, true, true); }
  bool isu32Imm() const { return CheckImmRange(32, 0, false, true, false); }
  bool isu26_6Imm() const { return CheckImmRange(26, 6, false, true, false); }
  bool isu16Imm() const { return CheckImmRange(16, 0, false, true, false); }
  bool isu16_0Imm() const { return CheckImmRange(16, 0, false, true, false); }
  bool isu16_1Imm() const { return CheckImmRange(16, 1, false, true, false); }
  bool isu16_2Imm() const { return CheckImmRange(16, 2, false, true, false); }
  bool isu16_3Imm() const { return CheckImmRange(16, 3, false, true, false); }
  bool isu11_3Imm() const { return CheckImmRange(11, 3, false, false, false); }
  bool isu6_0Imm() const { return CheckImmRange(6, 0, false, false, false); }
  bool isu6_1Imm() const { return CheckImmRange(6, 1, false, false, false); }
  bool isu6_2Imm() const { return CheckImmRange(6, 2, false, false, false); }
  bool isu6_3Imm() const { return CheckImmRange(6, 3, false, false, false); }
  bool isu10Imm() const { return CheckImmRange(10, 0, false, false, false); }
  bool isu9Imm() const { return CheckImmRange(9, 0, false, false, false); }
  bool isu8Imm() const { return CheckImmRange(8, 0, false, false, false); }
  bool isu7Imm() const { return CheckImmRange(7, 0, false, false, false); }
  bool isu6Imm() const { return CheckImmRange(6, 0, false, false, false); }
  bool isu5Imm() const { return CheckImmRange(5, 0, false, false, false); }
  bool isu4Imm() const { return CheckImmRange(4, 0, false, false, false); }
  bool isu3Imm() const { return CheckImmRange(3, 0, false, false, false); }
  bool isu2Imm() const { return CheckImmRange(2, 0, false, false, false); }
  bool isu1Imm() const { return CheckImmRange(1, 0, false, false, false); }

  bool ism6Imm() const { return CheckImmRange(6, 0, false, false, false); }
  bool isn8Imm() const { return CheckImmRange(8, 0, false, false, false); }

  bool iss16Ext() const { return CheckImmRange(16 + 26, 0, true, true, true); }
  bool iss12Ext() const { return CheckImmRange(12 + 26, 0, true, true, true); }
  bool iss10Ext() const { return CheckImmRange(10 + 26, 0, true, true, true); }
  bool iss9Ext() const { return CheckImmRange(9 + 26, 0, true, true, true); }
  bool iss8Ext() const { return CheckImmRange(8 + 26, 0, true, true, true); }
  bool iss7Ext() const { return CheckImmRange(7 + 26, 0, true, true, true); }
  bool iss6Ext() const { return CheckImmRange(6 + 26, 0, true, true, true); }
  bool iss11_0Ext() const {
    return CheckImmRange(11 + 26, 0, true, true, true);
  }
  bool iss11_1Ext() const {
    return CheckImmRange(11 + 26, 1, true, true, true);
  }
  bool iss11_2Ext() const {
    return CheckImmRange(11 + 26, 2, true, true, true);
  }
  bool iss11_3Ext() const {
    return CheckImmRange(11 + 26, 3, true, true, true);
  }

  bool isu6Ext() const { return CheckImmRange(6 + 26, 0, false, true, true); }
  bool isu7Ext() const { return CheckImmRange(7 + 26, 0, false, true, true); }
  bool isu8Ext() const { return CheckImmRange(8 + 26, 0, false, true, true); }
  bool isu9Ext() const { return CheckImmRange(9 + 26, 0, false, true, true); }
  bool isu10Ext() const { return CheckImmRange(10 + 26, 0, false, true, true); }
  bool isu6_0Ext() const { return CheckImmRange(6 + 26, 0, false, true, true); }
  bool isu6_1Ext() const { return CheckImmRange(6 + 26, 1, false, true, true); }
  bool isu6_2Ext() const { return CheckImmRange(6 + 26, 2, false, true, true); }
  bool isu6_3Ext() const { return CheckImmRange(6 + 26, 3, false, true, true); }
  bool isu32MustExt() const { return isImm(); }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createExpr(getImm()));
  }

  void addSignedImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    HexagonMCExpr *Expr =
        const_cast<HexagonMCExpr *>(cast<HexagonMCExpr>(getImm()));
    int64_t Value;
    if (!Expr->evaluateAsAbsolute(Value)) {
      Inst.addOperand(MCOperand::createExpr(Expr));
      return;
    }
    int64_t Extended = SignExtend64(Value, 32);
    if ((Extended < 0) != (Value < 0))
      Expr->setSignMismatch();
    Inst.addOperand(MCOperand::createExpr(Expr));
  }

  void addf32ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void adds32ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds23_2ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds8ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds8Imm64Operands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds6ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds4ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds4_0ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds4_1ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds4_2ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds4_3ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds3ImmOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }

  void addu64ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu32ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu26_6ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu16ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu16_0ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu16_1ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu16_2ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu16_3ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu11_3ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu10ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu9ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu8ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu7ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu6ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu6_0ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu6_1ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu6_2ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu6_3ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu5ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu4ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu3ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu2ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu1ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addm6ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addn8ImmOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void adds16ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds12ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds10ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds9ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds8ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds6ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds11_0ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds11_1ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds11_2ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }
  void adds11_3ExtOperands(MCInst &Inst, unsigned N) const {
    addSignedImmOperands(Inst, N);
  }

  void addu6ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu7ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu8ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu9ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu10ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu6_0ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu6_1ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu6_2ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu6_3ExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }
  void addu32MustExtOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void adds4_6ImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *CE =
        dyn_cast<MCConstantExpr>(&HexagonMCInstrInfo::getExpr(*getImm()));
    Inst.addOperand(MCOperand::createImm(CE->getValue() * 64));
  }

  void adds3_6ImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *CE =
        dyn_cast<MCConstantExpr>(&HexagonMCInstrInfo::getExpr(*getImm()));
    Inst.addOperand(MCOperand::createImm(CE->getValue() * 64));
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  virtual void print(raw_ostream &OS) const;

  static std::unique_ptr<HexagonOperand> CreateToken(StringRef Str, SMLoc S) {
    HexagonOperand *Op = new HexagonOperand(Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return std::unique_ptr<HexagonOperand>(Op);
  }

  static std::unique_ptr<HexagonOperand> CreateReg(unsigned RegNum, SMLoc S,
                                                   SMLoc E) {
    HexagonOperand *Op = new HexagonOperand(Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return std::unique_ptr<HexagonOperand>(Op);
  }

  static std::unique_ptr<HexagonOperand> CreateImm(const MCExpr *Val, SMLoc S,
                                                   SMLoc E) {
    HexagonOperand *Op = new HexagonOperand(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return std::unique_ptr<HexagonOperand>(Op);
  }
};

} // end anonymous namespace.

void HexagonOperand::print(raw_ostream &OS) const {
  switch (Kind) {
  case Immediate:
    getImm()->print(OS, nullptr);
    break;
  case Register:
    OS << "<register R";
    OS << getReg() << ">";
    break;
  case Token:
    OS << "'" << getToken() << "'";
    break;
  }
}

bool HexagonAsmParser::finishBundle(SMLoc IDLoc, MCStreamer &Out) {
  DEBUG(dbgs() << "Bundle:");
  DEBUG(MCB.dump_pretty(dbgs()));
  DEBUG(dbgs() << "--\n");

  // Check the bundle for errors.
  const MCRegisterInfo *RI = getContext().getRegisterInfo();
  HexagonMCChecker Check(MCII, getSTI(), MCB, MCB, *RI);

  bool CheckOk = HexagonMCInstrInfo::canonicalizePacket(MCII, getSTI(),
                                                        getContext(), MCB,
                                                        &Check);

  while (Check.getNextErrInfo() == true) {
    unsigned Reg = Check.getErrRegister();
    Twine R(RI->getName(Reg));

    uint64_t Err = Check.getError();
    if (Err != HexagonMCErrInfo::CHECK_SUCCESS) {
      if (HexagonMCErrInfo::CHECK_ERROR_BRANCHES & Err)
        Error(IDLoc,
              "unconditional branch cannot precede another branch in packet");

      if (HexagonMCErrInfo::CHECK_ERROR_NEWP & Err ||
          HexagonMCErrInfo::CHECK_ERROR_NEWV & Err)
        Error(IDLoc, "register `" + R +
                         "' used with `.new' "
                         "but not validly modified in the same packet");

      if (HexagonMCErrInfo::CHECK_ERROR_REGISTERS & Err)
        Error(IDLoc, "register `" + R + "' modified more than once");

      if (HexagonMCErrInfo::CHECK_ERROR_READONLY & Err)
        Error(IDLoc, "cannot write to read-only register `" + R + "'");

      if (HexagonMCErrInfo::CHECK_ERROR_LOOP & Err)
        Error(IDLoc, "loop-setup and some branch instructions "
                     "cannot be in the same packet");

      if (HexagonMCErrInfo::CHECK_ERROR_ENDLOOP & Err) {
        Twine N(HexagonMCInstrInfo::isInnerLoop(MCB) ? '0' : '1');
        Error(IDLoc, "packet marked with `:endloop" + N + "' " +
                         "cannot contain instructions that modify register " +
                         "`" + R + "'");
      }

      if (HexagonMCErrInfo::CHECK_ERROR_SOLO & Err)
        Error(IDLoc,
              "instruction cannot appear in packet with other instructions");

      if (HexagonMCErrInfo::CHECK_ERROR_NOSLOTS & Err)
        Error(IDLoc, "too many slots used in packet");

      if (Err & HexagonMCErrInfo::CHECK_ERROR_SHUFFLE) {
        uint64_t Erm = Check.getShuffleError();

        if (HexagonShuffler::SHUFFLE_ERROR_INVALID == Erm)
          Error(IDLoc, "invalid instruction packet");
        else if (HexagonShuffler::SHUFFLE_ERROR_STORES == Erm)
          Error(IDLoc, "invalid instruction packet: too many stores");
        else if (HexagonShuffler::SHUFFLE_ERROR_LOADS == Erm)
          Error(IDLoc, "invalid instruction packet: too many loads");
        else if (HexagonShuffler::SHUFFLE_ERROR_BRANCHES == Erm)
          Error(IDLoc, "too many branches in packet");
        else if (HexagonShuffler::SHUFFLE_ERROR_NOSLOTS == Erm)
          Error(IDLoc, "invalid instruction packet: out of slots");
        else if (HexagonShuffler::SHUFFLE_ERROR_SLOTS == Erm)
          Error(IDLoc, "invalid instruction packet: slot error");
        else if (HexagonShuffler::SHUFFLE_ERROR_ERRATA2 == Erm)
          Error(IDLoc, "v60 packet violation");
        else if (HexagonShuffler::SHUFFLE_ERROR_STORE_LOAD_CONFLICT == Erm)
          Error(IDLoc, "slot 0 instruction does not allow slot 1 store");
        else
          Error(IDLoc, "unknown error in instruction packet");
      }
    }

    unsigned Warn = Check.getWarning();
    if (Warn != HexagonMCErrInfo::CHECK_SUCCESS) {
      if (HexagonMCErrInfo::CHECK_WARN_CURRENT & Warn)
        Warning(IDLoc, "register `" + R + "' used with `.cur' "
                                          "but not used in the same packet");
      else if (HexagonMCErrInfo::CHECK_WARN_TEMPORARY & Warn)
        Warning(IDLoc, "register `" + R + "' used with `.tmp' "
                                          "but not used in the same packet");
    }
  }

  if (CheckOk) {
    MCB.setLoc(IDLoc);
    if (HexagonMCInstrInfo::bundleSize(MCB) == 0) {
      assert(!HexagonMCInstrInfo::isInnerLoop(MCB));
      assert(!HexagonMCInstrInfo::isOuterLoop(MCB));
      // Empty packets are valid yet aren't emitted
      return false;
    }
    Out.EmitInstruction(MCB, getSTI());
  } else {
    // If compounding and duplexing didn't reduce the size below
    // 4 or less we have a packet that is too big.
    if (HexagonMCInstrInfo::bundleSize(MCB) > HEXAGON_PACKET_SIZE) {
      Error(IDLoc, "invalid instruction packet: out of slots");
      return true; // Error
    }
  }

  return false; // No error
}

bool HexagonAsmParser::matchBundleOptions() {
  MCAsmParser &Parser = getParser();
  while (true) {
    if (!Parser.getTok().is(AsmToken::Colon))
      return false;
    Lex();
    StringRef Option = Parser.getTok().getString();
    if (Option.compare_lower("endloop0") == 0)
      HexagonMCInstrInfo::setInnerLoop(MCB);
    else if (Option.compare_lower("endloop1") == 0)
      HexagonMCInstrInfo::setOuterLoop(MCB);
    else if (Option.compare_lower("mem_noshuf") == 0)
      HexagonMCInstrInfo::setMemReorderDisabled(MCB);
    else if (Option.compare_lower("mem_shuf") == 0)
      HexagonMCInstrInfo::setMemStoreReorderEnabled(MCB);
    else
      return true;
    Lex();
  }
}

// For instruction aliases, immediates are generated rather than
// MCConstantExpr.  Convert them for uniform MCExpr.
// Also check for signed/unsigned mismatches and warn
void HexagonAsmParser::canonicalizeImmediates(MCInst &MCI) {
  MCInst NewInst;
  NewInst.setOpcode(MCI.getOpcode());
  for (MCOperand &I : MCI)
    if (I.isImm()) {
      int64_t Value (I.getImm());
      NewInst.addOperand(MCOperand::createExpr(HexagonMCExpr::create(
          MCConstantExpr::create(Value, getContext()), getContext())));
    }
    else {
      if (I.isExpr() && cast<HexagonMCExpr>(I.getExpr())->signMismatch() &&
          WarnSignedMismatch)
        Warning (MCI.getLoc(), "Signed/Unsigned mismatch");
      NewInst.addOperand(I);
    }
  MCI = NewInst;
}

bool HexagonAsmParser::matchOneInstruction(MCInst &MCI, SMLoc IDLoc,
                                           OperandVector &InstOperands,
                                           uint64_t &ErrorInfo,
                                           bool MatchingInlineAsm) {
  // Perform matching with tablegen asmmatcher generated function
  int result =
      MatchInstructionImpl(InstOperands, MCI, ErrorInfo, MatchingInlineAsm);
  if (result == Match_Success) {
    MCI.setLoc(IDLoc);
    canonicalizeImmediates(MCI);
    result = processInstruction(MCI, InstOperands, IDLoc);

    DEBUG(dbgs() << "Insn:");
    DEBUG(MCI.dump_pretty(dbgs()));
    DEBUG(dbgs() << "\n\n");

    MCI.setLoc(IDLoc);
  }

  // Create instruction operand for bundle instruction
  //   Break this into a separate function Code here is less readable
  //   Think about how to get an instruction error to report correctly.
  //   SMLoc will return the "{"
  switch (result) {
  default:
    break;
  case Match_Success:
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "invalid instruction");
  case Match_MnemonicFail:
    return Error(IDLoc, "unrecognized instruction");
  case Match_InvalidOperand:
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= InstOperands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = (static_cast<HexagonOperand *>(InstOperands[ErrorInfo].get()))
                     ->getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }
  llvm_unreachable("Implement any new match types added!");
}

bool HexagonAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                               OperandVector &Operands,
                                               MCStreamer &Out,
                                               uint64_t &ErrorInfo,
                                               bool MatchingInlineAsm) {
  if (!InBrackets) {
    MCB.clear();
    MCB.addOperand(MCOperand::createImm(0));
  }
  HexagonOperand &FirstOperand = static_cast<HexagonOperand &>(*Operands[0]);
  if (FirstOperand.isToken() && FirstOperand.getToken() == "{") {
    assert(Operands.size() == 1 && "Brackets should be by themselves");
    if (InBrackets) {
      getParser().Error(IDLoc, "Already in a packet");
      return true;
    }
    InBrackets = true;
    return false;
  }
  if (FirstOperand.isToken() && FirstOperand.getToken() == "}") {
    assert(Operands.size() == 1 && "Brackets should be by themselves");
    if (!InBrackets) {
      getParser().Error(IDLoc, "Not in a packet");
      return true;
    }
    InBrackets = false;
    if (matchBundleOptions())
      return true;
    return finishBundle(IDLoc, Out);
  }
  MCInst *SubInst = new (getParser().getContext()) MCInst;
  if (matchOneInstruction(*SubInst, IDLoc, Operands, ErrorInfo,
                          MatchingInlineAsm))
    return true;
  HexagonMCInstrInfo::extendIfNeeded(
      getParser().getContext(), MCII, MCB, *SubInst);
  MCB.addOperand(MCOperand::createInst(SubInst));
  if (!InBrackets)
    return finishBundle(IDLoc, Out);
  return false;
}

/// ParseDirective parses the Hexagon specific directives
bool HexagonAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if ((IDVal.lower() == ".word") || (IDVal.lower() == ".4byte"))
    return ParseDirectiveValue(4, DirectiveID.getLoc());
  if (IDVal.lower() == ".short" || IDVal.lower() == ".hword" ||
      IDVal.lower() == ".half")
    return ParseDirectiveValue(2, DirectiveID.getLoc());
  if (IDVal.lower() == ".falign")
    return ParseDirectiveFalign(256, DirectiveID.getLoc());
  if ((IDVal.lower() == ".lcomm") || (IDVal.lower() == ".lcommon"))
    return ParseDirectiveComm(true, DirectiveID.getLoc());
  if ((IDVal.lower() == ".comm") || (IDVal.lower() == ".common"))
    return ParseDirectiveComm(false, DirectiveID.getLoc());
  if (IDVal.lower() == ".subsection")
    return ParseDirectiveSubsection(DirectiveID.getLoc());

  return true;
}
bool HexagonAsmParser::ParseDirectiveSubsection(SMLoc L) {
  const MCExpr *Subsection = 0;
  int64_t Res;

  assert((getLexer().isNot(AsmToken::EndOfStatement)) &&
         "Invalid subsection directive");
  getParser().parseExpression(Subsection);

  if (!Subsection->evaluateAsAbsolute(Res))
    return Error(L, "Cannot evaluate subsection number");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  // 0-8192 is the hard-coded range in MCObjectStreamper.cpp, this keeps the
  // negative subsections together and in the same order but at the opposite
  // end of the section.  Only legacy hexagon-gcc created assembly code
  // used negative subsections.
  if ((Res < 0) && (Res > -8193))
    Subsection = HexagonMCExpr::create(
        MCConstantExpr::create(8192 + Res, getContext()), getContext());

  getStreamer().SubSection(Subsection);
  return false;
}

///  ::= .falign [expression]
bool HexagonAsmParser::ParseDirectiveFalign(unsigned Size, SMLoc L) {

  int64_t MaxBytesToFill = 15;

  // if there is an arguement
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    const MCExpr *Value;
    SMLoc ExprLoc = L;

    // Make sure we have a number (false is returned if expression is a number)
    if (getParser().parseExpression(Value) == false) {
      // Make sure this is a number that is in range
      const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value);
      uint64_t IntValue = MCE->getValue();
      if (!isUIntN(Size, IntValue) && !isIntN(Size, IntValue))
        return Error(ExprLoc, "literal value out of range (256) for falign");
      MaxBytesToFill = IntValue;
      Lex();
    } else {
      return Error(ExprLoc, "not a valid expression for falign directive");
    }
  }

  getTargetStreamer().emitFAlign(16, MaxBytesToFill);
  Lex();

  return false;
}

///  ::= .word [ expression (, expression)* ]
bool HexagonAsmParser::ParseDirectiveValue(unsigned Size, SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {

    for (;;) {
      const MCExpr *Value;
      SMLoc ExprLoc = L;
      if (getParser().parseExpression(Value))
        return true;

      // Special case constant expressions to match code generator.
      if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value)) {
        assert(Size <= 8 && "Invalid size");
        uint64_t IntValue = MCE->getValue();
        if (!isUIntN(8 * Size, IntValue) && !isIntN(8 * Size, IntValue))
          return Error(ExprLoc, "literal value out of range for directive");
        getStreamer().EmitIntValue(IntValue, Size);
      } else
        getStreamer().EmitValue(Value, Size);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();
    }
  }

  Lex();
  return false;
}

// This is largely a copy of AsmParser's ParseDirectiveComm extended to
// accept a 3rd argument, AccessAlignment which indicates the smallest
// memory access made to the symbol, expressed in bytes.  If no
// AccessAlignment is specified it defaults to the Alignment Value.
// Hexagon's .lcomm:
//   .lcomm Symbol, Length, Alignment, AccessAlignment
bool HexagonAsmParser::ParseDirectiveComm(bool IsLocal, SMLoc Loc) {
  // FIXME: need better way to detect if AsmStreamer (upstream removed
  // getKind())
  if (getStreamer().hasRawTextSupport())
    return true; // Only object file output requires special treatment.

  StringRef Name;
  if (getParser().parseIdentifier(Name))
    return TokError("expected identifier in directive");
  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (getParser().parseAbsoluteExpression(Size))
    return true;

  int64_t ByteAlignment = 1;
  SMLoc ByteAlignmentLoc;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    ByteAlignmentLoc = getLexer().getLoc();
    if (getParser().parseAbsoluteExpression(ByteAlignment))
      return true;
    if (!isPowerOf2_64(ByteAlignment))
      return Error(ByteAlignmentLoc, "alignment must be a power of 2");
  }

  int64_t AccessAlignment = 0;
  if (getLexer().is(AsmToken::Comma)) {
    // The optional access argument specifies the size of the smallest memory
    //   access to be made to the symbol, expressed in bytes.
    SMLoc AccessAlignmentLoc;
    Lex();
    AccessAlignmentLoc = getLexer().getLoc();
    if (getParser().parseAbsoluteExpression(AccessAlignment))
      return true;

    if (!isPowerOf2_64(AccessAlignment))
      return Error(AccessAlignmentLoc, "access alignment must be a power of 2");
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.comm' or '.lcomm' directive");

  Lex();

  // NOTE: a size of zero for a .comm should create a undefined symbol
  // but a size of .lcomm creates a bss symbol of size zero.
  if (Size < 0)
    return Error(SizeLoc, "invalid '.comm' or '.lcomm' directive size, can't "
                          "be less than zero");

  // NOTE: The alignment in the directive is a power of 2 value, the assembler
  // may internally end up wanting an alignment in bytes.
  // FIXME: Diagnose overflow.
  if (ByteAlignment < 0)
    return Error(ByteAlignmentLoc, "invalid '.comm' or '.lcomm' directive "
                                   "alignment, can't be less than zero");

  if (!Sym->isUndefined())
    return Error(Loc, "invalid symbol redefinition");

  HexagonMCELFStreamer &HexagonELFStreamer =
      static_cast<HexagonMCELFStreamer &>(getStreamer());
  if (IsLocal) {
    HexagonELFStreamer.HexagonMCEmitLocalCommonSymbol(Sym, Size, ByteAlignment,
                                                      AccessAlignment);
    return false;
  }

  HexagonELFStreamer.HexagonMCEmitCommonSymbol(Sym, Size, ByteAlignment,
                                               AccessAlignment);
  return false;
}

// validate register against architecture
bool HexagonAsmParser::RegisterMatchesArch(unsigned MatchNum) const {
  return true;
}

// extern "C" void LLVMInitializeHexagonAsmLexer();

/// Force static initialization.
extern "C" void LLVMInitializeHexagonAsmParser() {
  RegisterMCAsmParser<HexagonAsmParser> X(TheHexagonTarget);
}

#define GET_MATCHER_IMPLEMENTATION
#define GET_REGISTER_MATCHER
#include "HexagonGenAsmMatcher.inc"

namespace {
bool previousEqual(OperandVector &Operands, size_t Index, StringRef String) {
  if (Index >= Operands.size())
    return false;
  MCParsedAsmOperand &Operand = *Operands[Operands.size() - Index - 1];
  if (!Operand.isToken())
    return false;
  return static_cast<HexagonOperand &>(Operand).getToken().equals_lower(String);
}
bool previousIsLoop(OperandVector &Operands, size_t Index) {
  return previousEqual(Operands, Index, "loop0") ||
         previousEqual(Operands, Index, "loop1") ||
         previousEqual(Operands, Index, "sp1loop0") ||
         previousEqual(Operands, Index, "sp2loop0") ||
         previousEqual(Operands, Index, "sp3loop0");
}
}

bool HexagonAsmParser::splitIdentifier(OperandVector &Operands) {
  AsmToken const &Token = getParser().getTok();
  StringRef String = Token.getString();
  SMLoc Loc = Token.getLoc();
  Lex();
  do {
    std::pair<StringRef, StringRef> HeadTail = String.split('.');
    if (!HeadTail.first.empty())
      Operands.push_back(HexagonOperand::CreateToken(HeadTail.first, Loc));
    if (!HeadTail.second.empty())
      Operands.push_back(HexagonOperand::CreateToken(
          String.substr(HeadTail.first.size(), 1), Loc));
    String = HeadTail.second;
  } while (!String.empty());
  return false;
}

bool HexagonAsmParser::parseOperand(OperandVector &Operands) {
  unsigned Register;
  SMLoc Begin;
  SMLoc End;
  MCAsmLexer &Lexer = getLexer();
  if (!ParseRegister(Register, Begin, End)) {
    if (!ErrorMissingParenthesis)
      switch (Register) {
      default:
        break;
      case Hexagon::P0:
      case Hexagon::P1:
      case Hexagon::P2:
      case Hexagon::P3:
        if (previousEqual(Operands, 0, "if")) {
          if (WarnMissingParenthesis)
            Warning (Begin, "Missing parenthesis around predicate register");
          static char const *LParen = "(";
          static char const *RParen = ")";
          Operands.push_back(HexagonOperand::CreateToken(LParen, Begin));
          Operands.push_back(HexagonOperand::CreateReg(Register, Begin, End));
          const AsmToken &MaybeDotNew = Lexer.getTok();
          if (MaybeDotNew.is(AsmToken::TokenKind::Identifier) &&
              MaybeDotNew.getString().equals_lower(".new"))
            splitIdentifier(Operands);
          Operands.push_back(HexagonOperand::CreateToken(RParen, Begin));
          return false;
        }
        if (previousEqual(Operands, 0, "!") &&
            previousEqual(Operands, 1, "if")) {
          if (WarnMissingParenthesis)
            Warning (Begin, "Missing parenthesis around predicate register");
          static char const *LParen = "(";
          static char const *RParen = ")";
          Operands.insert(Operands.end () - 1,
                          HexagonOperand::CreateToken(LParen, Begin));
          Operands.push_back(HexagonOperand::CreateReg(Register, Begin, End));
          const AsmToken &MaybeDotNew = Lexer.getTok();
          if (MaybeDotNew.is(AsmToken::TokenKind::Identifier) &&
              MaybeDotNew.getString().equals_lower(".new"))
            splitIdentifier(Operands);
          Operands.push_back(HexagonOperand::CreateToken(RParen, Begin));
          return false;
        }
        break;
      }
    Operands.push_back(HexagonOperand::CreateReg(
        Register, Begin, End));
    return false;
  }
  return splitIdentifier(Operands);
}

bool HexagonAsmParser::isLabel(AsmToken &Token) {
  MCAsmLexer &Lexer = getLexer();
  AsmToken const &Second = Lexer.getTok();
  AsmToken Third = Lexer.peekTok();  
  StringRef String = Token.getString();
  if (Token.is(AsmToken::TokenKind::LCurly) ||
      Token.is(AsmToken::TokenKind::RCurly))
    return false;
  if (!Token.is(AsmToken::TokenKind::Identifier))
    return true;
  if (!matchRegister(String.lower()))
    return true;
  (void)Second;
  assert(Second.is(AsmToken::Colon));
  StringRef Raw (String.data(), Third.getString().data() - String.data() +
                 Third.getString().size());
  std::string Collapsed = Raw;
  Collapsed.erase(std::remove_if(Collapsed.begin(), Collapsed.end(), isspace),
                  Collapsed.end());
  StringRef Whole = Collapsed;
  std::pair<StringRef, StringRef> DotSplit = Whole.split('.');
  if (!matchRegister(DotSplit.first.lower()))
    return true;
  return false;
}

bool HexagonAsmParser::handleNoncontigiousRegister(bool Contigious, SMLoc &Loc) {
  if (!Contigious && ErrorNoncontigiousRegister) {
    Error(Loc, "Register name is not contigious");
    return true;
  }
  if (!Contigious && WarnNoncontigiousRegister)
    Warning(Loc, "Register name is not contigious");
  return false;
}

bool HexagonAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) {
  MCAsmLexer &Lexer = getLexer();
  StartLoc = getLexer().getLoc();
  SmallVector<AsmToken, 5> Lookahead;
  StringRef RawString(Lexer.getTok().getString().data(), 0);
  bool Again = Lexer.is(AsmToken::Identifier);
  bool NeededWorkaround = false;
  while (Again) {
    AsmToken const &Token = Lexer.getTok();
    RawString = StringRef(RawString.data(),
                          Token.getString().data() - RawString.data () +
                          Token.getString().size());
    Lookahead.push_back(Token);
    Lexer.Lex();
    bool Contigious = Lexer.getTok().getString().data() ==
                      Lookahead.back().getString().data() +
                      Lookahead.back().getString().size();
    bool Type = Lexer.is(AsmToken::Identifier) || Lexer.is(AsmToken::Dot) ||
                Lexer.is(AsmToken::Integer) || Lexer.is(AsmToken::Real) ||
                Lexer.is(AsmToken::Colon);
    bool Workaround = Lexer.is(AsmToken::Colon) ||
                      Lookahead.back().is(AsmToken::Colon);
    Again = (Contigious && Type) || (Workaround && Type);
    NeededWorkaround = NeededWorkaround || (Again && !(Contigious && Type));
  }
  std::string Collapsed = RawString;
  Collapsed.erase(std::remove_if(Collapsed.begin(), Collapsed.end(), isspace),
                  Collapsed.end());
  StringRef FullString = Collapsed;
  std::pair<StringRef, StringRef> DotSplit = FullString.split('.');
  unsigned DotReg = matchRegister(DotSplit.first.lower());
  if (DotReg != Hexagon::NoRegister && RegisterMatchesArch(DotReg)) {
    if (DotSplit.second.empty()) {
      RegNo = DotReg;
      EndLoc = Lexer.getLoc();
      if (handleNoncontigiousRegister(!NeededWorkaround, StartLoc))
        return true;
      return false;
    } else {
      RegNo = DotReg;
      size_t First = RawString.find('.');
      StringRef DotString (RawString.data() + First, RawString.size() - First);
      Lexer.UnLex(AsmToken(AsmToken::Identifier, DotString));
      EndLoc = Lexer.getLoc();
      if (handleNoncontigiousRegister(!NeededWorkaround, StartLoc))
        return true;
      return false;
    }
  }
  std::pair<StringRef, StringRef> ColonSplit = StringRef(FullString).split(':');
  unsigned ColonReg = matchRegister(ColonSplit.first.lower());
  if (ColonReg != Hexagon::NoRegister && RegisterMatchesArch(DotReg)) {
    Lexer.UnLex(Lookahead.back());
    Lookahead.pop_back();
    Lexer.UnLex(Lookahead.back());
    Lookahead.pop_back();
    RegNo = ColonReg;
    EndLoc = Lexer.getLoc();
    if (handleNoncontigiousRegister(!NeededWorkaround, StartLoc))
      return true;
    return false;
  }
  while (!Lookahead.empty()) {
    Lexer.UnLex(Lookahead.back());
    Lookahead.pop_back();
  }
  return true;
}

bool HexagonAsmParser::implicitExpressionLocation(OperandVector &Operands) {
  if (previousEqual(Operands, 0, "call"))
    return true;
  if (previousEqual(Operands, 0, "jump"))
    if (!getLexer().getTok().is(AsmToken::Colon))
      return true;
  if (previousEqual(Operands, 0, "(") && previousIsLoop(Operands, 1))
    return true;
  if (previousEqual(Operands, 1, ":") && previousEqual(Operands, 2, "jump") &&
      (previousEqual(Operands, 0, "nt") || previousEqual(Operands, 0, "t")))
    return true;
  return false;
}

bool HexagonAsmParser::parseExpression(MCExpr const *& Expr) {
  llvm::SmallVector<AsmToken, 4> Tokens;
  MCAsmLexer &Lexer = getLexer();
  bool Done = false;
  static char const * Comma = ",";
  do {
    Tokens.emplace_back (Lexer.getTok());
    Lex();
    switch (Tokens.back().getKind())
    {
    case AsmToken::TokenKind::Hash:
      if (Tokens.size () > 1)
        if ((Tokens.end () - 2)->getKind() == AsmToken::TokenKind::Plus) {
          Tokens.insert(Tokens.end() - 2,
                        AsmToken(AsmToken::TokenKind::Comma, Comma));
          Done = true;
        }
      break;
    case AsmToken::TokenKind::RCurly:
    case AsmToken::TokenKind::EndOfStatement:
    case AsmToken::TokenKind::Eof:
      Done = true;
      break;
    default:
      break;
    }
  } while (!Done);
  while (!Tokens.empty()) {
    Lexer.UnLex(Tokens.back());
    Tokens.pop_back();
  }
  return getParser().parseExpression(Expr);
}

bool HexagonAsmParser::parseExpressionOrOperand(OperandVector &Operands) {
  if (implicitExpressionLocation(Operands)) {
    MCAsmParser &Parser = getParser();
    SMLoc Loc = Parser.getLexer().getLoc();
    MCExpr const *Expr = nullptr;
    bool Error = parseExpression(Expr);
    Expr = HexagonMCExpr::create(Expr, getContext());
    if (!Error)
      Operands.push_back(HexagonOperand::CreateImm(Expr, Loc, Loc));
    return Error;
  }
  return parseOperand(Operands);
}

/// Parse an instruction.
bool HexagonAsmParser::parseInstruction(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  MCAsmLexer &Lexer = getLexer();
  while (true) {
    AsmToken const &Token = Parser.getTok();
    switch (Token.getKind()) {
    case AsmToken::EndOfStatement: {
      Lex();
      return false;
    }
    case AsmToken::LCurly: {
      if (!Operands.empty())
        return true;
      Operands.push_back(
          HexagonOperand::CreateToken(Token.getString(), Token.getLoc()));
      Lex();
      return false;
    }
    case AsmToken::RCurly: {
      if (Operands.empty()) {
        Operands.push_back(
            HexagonOperand::CreateToken(Token.getString(), Token.getLoc()));
        Lex();
      }
      return false;
    }
    case AsmToken::Comma: {
      Lex();
      continue;
    }
    case AsmToken::EqualEqual:
    case AsmToken::ExclaimEqual:
    case AsmToken::GreaterEqual:
    case AsmToken::GreaterGreater:
    case AsmToken::LessEqual:
    case AsmToken::LessLess: {
      Operands.push_back(HexagonOperand::CreateToken(
          Token.getString().substr(0, 1), Token.getLoc()));
      Operands.push_back(HexagonOperand::CreateToken(
          Token.getString().substr(1, 1), Token.getLoc()));
      Lex();
      continue;
    }
    case AsmToken::Hash: {
      bool MustNotExtend = false;
      bool ImplicitExpression = implicitExpressionLocation(Operands);
      SMLoc ExprLoc = Lexer.getLoc();
      if (!ImplicitExpression)
        Operands.push_back(
          HexagonOperand::CreateToken(Token.getString(), Token.getLoc()));
      Lex();
      bool MustExtend = false;
      bool HiOnly = false;
      bool LoOnly = false;
      if (Lexer.is(AsmToken::Hash)) {
        Lex();
        MustExtend = true;
      } else if (ImplicitExpression)
        MustNotExtend = true;
      AsmToken const &Token = Parser.getTok();
      if (Token.is(AsmToken::Identifier)) {
        StringRef String = Token.getString();
        if (String.lower() == "hi") {
          HiOnly = true;
        } else if (String.lower() == "lo") {
          LoOnly = true;
        }
        if (HiOnly || LoOnly) {
          AsmToken LParen = Lexer.peekTok();
          if (!LParen.is(AsmToken::LParen)) {
            HiOnly = false;
            LoOnly = false;
          } else {
            Lex();
          }
        }
      }
      MCExpr const *Expr = nullptr;
      if (parseExpression(Expr))
        return true;
      int64_t Value;
      MCContext &Context = Parser.getContext();
      assert(Expr != nullptr);
      if (Expr->evaluateAsAbsolute(Value)) {
        if (HiOnly)
          Expr = MCBinaryExpr::createLShr(
              Expr,  MCConstantExpr::create(16, Context), Context);
        if (HiOnly || LoOnly)
          Expr = MCBinaryExpr::createAnd(Expr,
              MCConstantExpr::create(0xffff, Context),
                                    Context);
      } else {
        MCValue Value;
        if (Expr->evaluateAsRelocatable(Value, nullptr, nullptr)) {
          if (!Value.isAbsolute()) {
            switch(Value.getAccessVariant()) {
            case MCSymbolRefExpr::VariantKind::VK_TPREL:
            case MCSymbolRefExpr::VariantKind::VK_DTPREL:
              // Don't lazy extend these expression variants
              MustNotExtend = !MustExtend;
              break;
            default:
              break;
            }
          }
        }
      }
      Expr = HexagonMCExpr::create(Expr, Context);
      HexagonMCInstrInfo::setMustNotExtend(*Expr, MustNotExtend);
      HexagonMCInstrInfo::setMustExtend(*Expr, MustExtend);
      std::unique_ptr<HexagonOperand> Operand =
          HexagonOperand::CreateImm(Expr, ExprLoc, ExprLoc);
      Operands.push_back(std::move(Operand));
      continue;
    }
    default:
      break;
    }
    if (parseExpressionOrOperand(Operands))
      return true;
  }
}

bool HexagonAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                        StringRef Name,
                                        AsmToken ID,
                                        OperandVector &Operands) {
  getLexer().UnLex(ID);
  return parseInstruction(Operands);
}

namespace {
MCInst makeCombineInst(int opCode, MCOperand &Rdd,
                       MCOperand &MO1, MCOperand &MO2) {
  MCInst TmpInst;
  TmpInst.setOpcode(opCode);
  TmpInst.addOperand(Rdd);
  TmpInst.addOperand(MO1);
  TmpInst.addOperand(MO2);

  return TmpInst;
}
}

// Define this matcher function after the auto-generated include so we
// have the match class enum definitions.
unsigned HexagonAsmParser::validateTargetOperandClass(MCParsedAsmOperand &AsmOp,
                                                      unsigned Kind) {
  HexagonOperand *Op = static_cast<HexagonOperand *>(&AsmOp);

  switch (Kind) {
  case MCK_0: {
    int64_t Value;
    return Op->isImm() && Op->Imm.Val->evaluateAsAbsolute(Value) && Value == 0
               ? Match_Success
               : Match_InvalidOperand;
  }
  case MCK_1: {
    int64_t Value;
    return Op->isImm() && Op->Imm.Val->evaluateAsAbsolute(Value) && Value == 1
               ? Match_Success
               : Match_InvalidOperand;
  }
  case MCK__MINUS_1: {
    int64_t Value;
    return Op->isImm() && Op->Imm.Val->evaluateAsAbsolute(Value) && Value == -1
               ? Match_Success
               : Match_InvalidOperand;
  }
  }
  if (Op->Kind == HexagonOperand::Token && Kind != InvalidMatchClass) {
    StringRef myStringRef = StringRef(Op->Tok.Data, Op->Tok.Length);
    if (matchTokenString(myStringRef.lower()) == (MatchClassKind)Kind)
      return Match_Success;
    if (matchTokenString(myStringRef.upper()) == (MatchClassKind)Kind)
      return Match_Success;
  }

  DEBUG(dbgs() << "Unmatched Operand:");
  DEBUG(Op->dump());
  DEBUG(dbgs() << "\n");

  return Match_InvalidOperand;
}

void HexagonAsmParser::OutOfRange(SMLoc IDLoc, long long Val, long long Max) {
  std::string errStr;
  raw_string_ostream ES(errStr);
  ES << "value " << Val << "(" << format_hex(Val, 0) << ") out of range: ";
  if (Max >= 0)
    ES << "0-" << Max;
  else
    ES << Max << "-" << (-Max - 1);
  Error(IDLoc, ES.str().c_str());
}

int HexagonAsmParser::processInstruction(MCInst &Inst,
                                         OperandVector const &Operands,
                                         SMLoc IDLoc) {
  MCContext &Context = getParser().getContext();
  const MCRegisterInfo *RI = getContext().getRegisterInfo();
  std::string r = "r";
  std::string v = "v";
  std::string Colon = ":";

  bool is32bit = false; // used to distinguish between CONST32 and CONST64
  switch (Inst.getOpcode()) {
  default:
    break;

  case Hexagon::A2_iconst: {
    Inst.setOpcode(Hexagon::A2_addi);
    MCOperand Reg = Inst.getOperand(0);
    MCOperand S16 = Inst.getOperand(1);
    HexagonMCInstrInfo::setMustNotExtend(*S16.getExpr());
    HexagonMCInstrInfo::setS23_2_reloc(*S16.getExpr());
    Inst.clear();
    Inst.addOperand(Reg);
    Inst.addOperand(MCOperand::createReg(Hexagon::R0));
    Inst.addOperand(S16);
    break;
  }
  case Hexagon::M4_mpyrr_addr:
  case Hexagon::S4_addi_asl_ri:
  case Hexagon::S4_addi_lsr_ri:
  case Hexagon::S4_andi_asl_ri:
  case Hexagon::S4_andi_lsr_ri:
  case Hexagon::S4_ori_asl_ri:
  case Hexagon::S4_ori_lsr_ri:
  case Hexagon::S4_or_andix:
  case Hexagon::S4_subi_asl_ri:
  case Hexagon::S4_subi_lsr_ri: {
    MCOperand &Ry = Inst.getOperand(0);
    MCOperand &src = Inst.getOperand(2);
    if (RI->getEncodingValue(Ry.getReg()) != RI->getEncodingValue(src.getReg()))
      return Match_InvalidOperand;
    break;
  }

  case Hexagon::C2_cmpgei: {
    MCOperand &MO = Inst.getOperand(2);
    MO.setExpr(HexagonMCExpr::create(MCBinaryExpr::createSub(
        MO.getExpr(), MCConstantExpr::create(1, Context), Context), Context));
    Inst.setOpcode(Hexagon::C2_cmpgti);
    break;
  }

  case Hexagon::C2_cmpgeui: {
    MCOperand &MO = Inst.getOperand(2);
    int64_t Value;
    bool Success = MO.getExpr()->evaluateAsAbsolute(Value);
    (void)Success;
    assert(Success && "Assured by matcher");
    if (Value == 0) {
      MCInst TmpInst;
      MCOperand &Pd = Inst.getOperand(0);
      MCOperand &Rt = Inst.getOperand(1);
      TmpInst.setOpcode(Hexagon::C2_cmpeq);
      TmpInst.addOperand(Pd);
      TmpInst.addOperand(Rt);
      TmpInst.addOperand(Rt);
      Inst = TmpInst;
    } else {
      MO.setExpr(HexagonMCExpr::create(MCBinaryExpr::createSub(
          MO.getExpr(), MCConstantExpr::create(1, Context), Context), Context));
      Inst.setOpcode(Hexagon::C2_cmpgtui);
    }
    break;
  }

  // Translate a "$Rdd = $Rss" to "$Rdd = combine($Rs, $Rt)"
  case Hexagon::A2_tfrp: {
    MCOperand &MO = Inst.getOperand(1);
    unsigned int RegPairNum = RI->getEncodingValue(MO.getReg());
    std::string R1 = r + llvm::utostr(RegPairNum + 1);
    StringRef Reg1(R1);
    MO.setReg(matchRegister(Reg1));
    // Add a new operand for the second register in the pair.
    std::string R2 = r + llvm::utostr(RegPairNum);
    StringRef Reg2(R2);
    Inst.addOperand(MCOperand::createReg(matchRegister(Reg2)));
    Inst.setOpcode(Hexagon::A2_combinew);
    break;
  }

  case Hexagon::A2_tfrpt:
  case Hexagon::A2_tfrpf: {
    MCOperand &MO = Inst.getOperand(2);
    unsigned int RegPairNum = RI->getEncodingValue(MO.getReg());
    std::string R1 = r + llvm::utostr(RegPairNum + 1);
    StringRef Reg1(R1);
    MO.setReg(matchRegister(Reg1));
    // Add a new operand for the second register in the pair.
    std::string R2 = r + llvm::utostr(RegPairNum);
    StringRef Reg2(R2);
    Inst.addOperand(MCOperand::createReg(matchRegister(Reg2)));
    Inst.setOpcode((Inst.getOpcode() == Hexagon::A2_tfrpt)
                       ? Hexagon::C2_ccombinewt
                       : Hexagon::C2_ccombinewf);
    break;
  }
  case Hexagon::A2_tfrptnew:
  case Hexagon::A2_tfrpfnew: {
    MCOperand &MO = Inst.getOperand(2);
    unsigned int RegPairNum = RI->getEncodingValue(MO.getReg());
    std::string R1 = r + llvm::utostr(RegPairNum + 1);
    StringRef Reg1(R1);
    MO.setReg(matchRegister(Reg1));
    // Add a new operand for the second register in the pair.
    std::string R2 = r + llvm::utostr(RegPairNum);
    StringRef Reg2(R2);
    Inst.addOperand(MCOperand::createReg(matchRegister(Reg2)));
    Inst.setOpcode((Inst.getOpcode() == Hexagon::A2_tfrptnew)
                       ? Hexagon::C2_ccombinewnewt
                       : Hexagon::C2_ccombinewnewf);
    break;
  }

  // Translate a "$Vdd = $Vss" to "$Vdd = vcombine($Vs, $Vt)"
  case Hexagon::HEXAGON_V6_vassignpair: {
    MCOperand &MO = Inst.getOperand(1);
    unsigned int RegPairNum = RI->getEncodingValue(MO.getReg());
    std::string R1 = v + llvm::utostr(RegPairNum + 1);
    MO.setReg(MatchRegisterName(R1));
    // Add a new operand for the second register in the pair.
    std::string R2 = v + llvm::utostr(RegPairNum);
    Inst.addOperand(MCOperand::createReg(MatchRegisterName(R2)));
    Inst.setOpcode(Hexagon::V6_vcombine);
    break;
  }

  // Translate a "$Rx =  CONST32(#imm)" to "$Rx = memw(gp+#LABEL) "
  case Hexagon::CONST32:
  case Hexagon::CONST32_Float_Real:
  case Hexagon::CONST32_Int_Real:
  case Hexagon::FCONST32_nsdata:
    is32bit = true;
  // Translate a "$Rx:y =  CONST64(#imm)" to "$Rx:y = memd(gp+#LABEL) "
  case Hexagon::CONST64_Float_Real:
  case Hexagon::CONST64_Int_Real:

    // FIXME: need better way to detect AsmStreamer (upstream removed getKind())
    if (!Parser.getStreamer().hasRawTextSupport()) {
      MCELFStreamer *MES = static_cast<MCELFStreamer *>(&Parser.getStreamer());
      MCOperand &MO_1 = Inst.getOperand(1);
      MCOperand &MO_0 = Inst.getOperand(0);

      // push section onto section stack
      MES->PushSection();

      std::string myCharStr;
      MCSectionELF *mySection;

      // check if this as an immediate or a symbol
      int64_t Value;
      bool Absolute = MO_1.getExpr()->evaluateAsAbsolute(Value);
      if (Absolute) {
        // Create a new section - one for each constant
        // Some or all of the zeros are replaced with the given immediate.
        if (is32bit) {
          std::string myImmStr = utohexstr(static_cast<uint32_t>(Value));
          myCharStr = StringRef(".gnu.linkonce.l4.CONST_00000000")
                          .drop_back(myImmStr.size())
                          .str() +
                      myImmStr;
        } else {
          std::string myImmStr = utohexstr(Value);
          myCharStr = StringRef(".gnu.linkonce.l8.CONST_0000000000000000")
                          .drop_back(myImmStr.size())
                          .str() +
                      myImmStr;
        }

        mySection = getContext().getELFSection(myCharStr, ELF::SHT_PROGBITS,
                                               ELF::SHF_ALLOC | ELF::SHF_WRITE);
      } else if (MO_1.isExpr()) {
        // .lita - for expressions
        myCharStr = ".lita";
        mySection = getContext().getELFSection(myCharStr, ELF::SHT_PROGBITS,
                                               ELF::SHF_ALLOC | ELF::SHF_WRITE);
      } else
        llvm_unreachable("unexpected type of machine operand!");

      MES->SwitchSection(mySection);
      unsigned byteSize = is32bit ? 4 : 8;
      getStreamer().EmitCodeAlignment(byteSize, byteSize);

      MCSymbol *Sym;

      // for symbols, get rid of prepended ".gnu.linkonce.lx."

      // emit symbol if needed
      if (Absolute) {
        Sym = getContext().getOrCreateSymbol(StringRef(myCharStr.c_str() + 16));
        if (Sym->isUndefined()) {
          getStreamer().EmitLabel(Sym);
          getStreamer().EmitSymbolAttribute(Sym, MCSA_Global);
          getStreamer().EmitIntValue(Value, byteSize);
        }
      } else if (MO_1.isExpr()) {
        const char *StringStart = 0;
        const char *StringEnd = 0;
        if (*Operands[4]->getStartLoc().getPointer() == '#') {
          StringStart = Operands[5]->getStartLoc().getPointer();
          StringEnd = Operands[6]->getStartLoc().getPointer();
        } else { // no pound
          StringStart = Operands[4]->getStartLoc().getPointer();
          StringEnd = Operands[5]->getStartLoc().getPointer();
        }

        unsigned size = StringEnd - StringStart;
        std::string DotConst = ".CONST_";
        Sym = getContext().getOrCreateSymbol(DotConst +
                                             StringRef(StringStart, size));

        if (Sym->isUndefined()) {
          // case where symbol is not yet defined: emit symbol
          getStreamer().EmitLabel(Sym);
          getStreamer().EmitSymbolAttribute(Sym, MCSA_Local);
          getStreamer().EmitValue(MO_1.getExpr(), 4);
        }
      } else
        llvm_unreachable("unexpected type of machine operand!");

      MES->PopSection();

      if (Sym) {
        MCInst TmpInst;
        if (is32bit) // 32 bit
          TmpInst.setOpcode(Hexagon::L2_loadrigp);
        else // 64 bit
          TmpInst.setOpcode(Hexagon::L2_loadrdgp);

        TmpInst.addOperand(MO_0);
        TmpInst.addOperand(
            MCOperand::createExpr(MCSymbolRefExpr::create(Sym, getContext())));
        Inst = TmpInst;
      }
    }
    break;

  // Translate a "$Rdd = #-imm" to "$Rdd = combine(#[-1,0], #-imm)"
  case Hexagon::A2_tfrpi: {
    MCOperand &Rdd = Inst.getOperand(0);
    MCOperand &MO = Inst.getOperand(1);
    int64_t Value;
    int sVal = (MO.getExpr()->evaluateAsAbsolute(Value) && Value < 0) ? -1 : 0;
    MCOperand imm(MCOperand::createExpr(
        HexagonMCExpr::create(MCConstantExpr::create(sVal, Context), Context)));
    Inst = makeCombineInst(Hexagon::A2_combineii, Rdd, imm, MO);
    break;
  }

  // Translate a "$Rdd = [#]#imm" to "$Rdd = combine(#, [#]#imm)"
  case Hexagon::TFRI64_V4: {
    MCOperand &Rdd = Inst.getOperand(0);
    MCOperand &MO = Inst.getOperand(1);
    int64_t Value;
    if (MO.getExpr()->evaluateAsAbsolute(Value)) {
      int s8 = Hi_32(Value);
      if (!isInt<8>(s8))
        OutOfRange(IDLoc, s8, -128);
      MCOperand imm(MCOperand::createExpr(HexagonMCExpr::create(
          MCConstantExpr::create(s8, Context), Context))); // upper 32
      auto Expr = HexagonMCExpr::create(
          MCConstantExpr::create(Lo_32(Value), Context), Context);
      HexagonMCInstrInfo::setMustExtend(*Expr, HexagonMCInstrInfo::mustExtend(*MO.getExpr()));
      MCOperand imm2(MCOperand::createExpr(Expr)); // lower 32
      Inst = makeCombineInst(Hexagon::A4_combineii, Rdd, imm, imm2);
    } else {
      MCOperand imm(MCOperand::createExpr(HexagonMCExpr::create(
          MCConstantExpr::create(0, Context), Context))); // upper 32
      Inst = makeCombineInst(Hexagon::A4_combineii, Rdd, imm, MO);
    }
    break;
  }

  // Handle $Rdd = combine(##imm, #imm)"
  case Hexagon::TFRI64_V2_ext: {
    MCOperand &Rdd = Inst.getOperand(0);
    MCOperand &MO1 = Inst.getOperand(1);
    MCOperand &MO2 = Inst.getOperand(2);
    int64_t Value;
    if (MO2.getExpr()->evaluateAsAbsolute(Value)) {
      int s8 = Value;
      if (s8 < -128 || s8 > 127)
        OutOfRange(IDLoc, s8, -128);
    }
    Inst = makeCombineInst(Hexagon::A2_combineii, Rdd, MO1, MO2);
    break;
  }

  // Handle $Rdd = combine(#imm, ##imm)"
  case Hexagon::A4_combineii: {
    MCOperand &Rdd = Inst.getOperand(0);
    MCOperand &MO1 = Inst.getOperand(1);
    int64_t Value;
    if (MO1.getExpr()->evaluateAsAbsolute(Value)) {
      int s8 = Value;
      if (s8 < -128 || s8 > 127)
        OutOfRange(IDLoc, s8, -128);
    }
    MCOperand &MO2 = Inst.getOperand(2);
    Inst = makeCombineInst(Hexagon::A4_combineii, Rdd, MO1, MO2);
    break;
  }

  case Hexagon::S2_tableidxb_goodsyntax: {
    Inst.setOpcode(Hexagon::S2_tableidxb);
    break;
  }

  case Hexagon::S2_tableidxh_goodsyntax: {
    MCInst TmpInst;
    MCOperand &Rx = Inst.getOperand(0);
    MCOperand &_dst_ = Inst.getOperand(1);
    MCOperand &Rs = Inst.getOperand(2);
    MCOperand &Imm4 = Inst.getOperand(3);
    MCOperand &Imm6 = Inst.getOperand(4);
    Imm6.setExpr(HexagonMCExpr::create(MCBinaryExpr::createSub(
        Imm6.getExpr(), MCConstantExpr::create(1, Context), Context), Context));
    TmpInst.setOpcode(Hexagon::S2_tableidxh);
    TmpInst.addOperand(Rx);
    TmpInst.addOperand(_dst_);
    TmpInst.addOperand(Rs);
    TmpInst.addOperand(Imm4);
    TmpInst.addOperand(Imm6);
    Inst = TmpInst;
    break;
  }

  case Hexagon::S2_tableidxw_goodsyntax: {
    MCInst TmpInst;
    MCOperand &Rx = Inst.getOperand(0);
    MCOperand &_dst_ = Inst.getOperand(1);
    MCOperand &Rs = Inst.getOperand(2);
    MCOperand &Imm4 = Inst.getOperand(3);
    MCOperand &Imm6 = Inst.getOperand(4);
    Imm6.setExpr(HexagonMCExpr::create(MCBinaryExpr::createSub(
        Imm6.getExpr(), MCConstantExpr::create(2, Context), Context), Context));
    TmpInst.setOpcode(Hexagon::S2_tableidxw);
    TmpInst.addOperand(Rx);
    TmpInst.addOperand(_dst_);
    TmpInst.addOperand(Rs);
    TmpInst.addOperand(Imm4);
    TmpInst.addOperand(Imm6);
    Inst = TmpInst;
    break;
  }

  case Hexagon::S2_tableidxd_goodsyntax: {
    MCInst TmpInst;
    MCOperand &Rx = Inst.getOperand(0);
    MCOperand &_dst_ = Inst.getOperand(1);
    MCOperand &Rs = Inst.getOperand(2);
    MCOperand &Imm4 = Inst.getOperand(3);
    MCOperand &Imm6 = Inst.getOperand(4);
    Imm6.setExpr(HexagonMCExpr::create(MCBinaryExpr::createSub(
        Imm6.getExpr(), MCConstantExpr::create(3, Context), Context), Context));
    TmpInst.setOpcode(Hexagon::S2_tableidxd);
    TmpInst.addOperand(Rx);
    TmpInst.addOperand(_dst_);
    TmpInst.addOperand(Rs);
    TmpInst.addOperand(Imm4);
    TmpInst.addOperand(Imm6);
    Inst = TmpInst;
    break;
  }

  case Hexagon::M2_mpyui: {
    Inst.setOpcode(Hexagon::M2_mpyi);
    break;
  }
  case Hexagon::M2_mpysmi: {
    MCInst TmpInst;
    MCOperand &Rd = Inst.getOperand(0);
    MCOperand &Rs = Inst.getOperand(1);
    MCOperand &Imm = Inst.getOperand(2);
    int64_t Value;
    MCExpr const &Expr = *Imm.getExpr();
    bool Absolute = Expr.evaluateAsAbsolute(Value);
    assert(Absolute);
    (void)Absolute;
    if (!HexagonMCInstrInfo::mustExtend(Expr)) {
      if (Value < 0 && Value > -256) {
        Imm.setExpr(HexagonMCExpr::create(
            MCConstantExpr::create(Value * -1, Context), Context));
        TmpInst.setOpcode(Hexagon::M2_mpysin);
      } else if (Value < 256 && Value >= 0)
        TmpInst.setOpcode(Hexagon::M2_mpysip);
      else
        return Match_InvalidOperand;
    } else {
      if (Value >= 0)
        TmpInst.setOpcode(Hexagon::M2_mpysip);
      else
        return Match_InvalidOperand;
    }
    TmpInst.addOperand(Rd);
    TmpInst.addOperand(Rs);
    TmpInst.addOperand(Imm);
    Inst = TmpInst;
    break;
  }

  case Hexagon::S2_asr_i_r_rnd_goodsyntax: {
    MCOperand &Imm = Inst.getOperand(2);
    MCInst TmpInst;
    int64_t Value;
    bool Absolute = Imm.getExpr()->evaluateAsAbsolute(Value);
    assert(Absolute);
    (void)Absolute;
    if (Value == 0) { // convert to $Rd = $Rs
      TmpInst.setOpcode(Hexagon::A2_tfr);
      MCOperand &Rd = Inst.getOperand(0);
      MCOperand &Rs = Inst.getOperand(1);
      TmpInst.addOperand(Rd);
      TmpInst.addOperand(Rs);
    } else {
      Imm.setExpr(HexagonMCExpr::create(
          MCBinaryExpr::createSub(Imm.getExpr(),
                                  MCConstantExpr::create(1, Context), Context),
          Context));
      TmpInst.setOpcode(Hexagon::S2_asr_i_r_rnd);
      MCOperand &Rd = Inst.getOperand(0);
      MCOperand &Rs = Inst.getOperand(1);
      TmpInst.addOperand(Rd);
      TmpInst.addOperand(Rs);
      TmpInst.addOperand(Imm);
    }
    Inst = TmpInst;
    break;
  }

  case Hexagon::S2_asr_i_p_rnd_goodsyntax: {
    MCOperand &Rdd = Inst.getOperand(0);
    MCOperand &Rss = Inst.getOperand(1);
    MCOperand &Imm = Inst.getOperand(2);
    int64_t Value;
    bool Absolute = Imm.getExpr()->evaluateAsAbsolute(Value);
    assert(Absolute);
    (void)Absolute;
    if (Value == 0) { // convert to $Rdd = combine ($Rs[0], $Rs[1])
      MCInst TmpInst;
      unsigned int RegPairNum = RI->getEncodingValue(Rss.getReg());
      std::string R1 = r + llvm::utostr(RegPairNum + 1);
      StringRef Reg1(R1);
      Rss.setReg(matchRegister(Reg1));
      // Add a new operand for the second register in the pair.
      std::string R2 = r + llvm::utostr(RegPairNum);
      StringRef Reg2(R2);
      TmpInst.setOpcode(Hexagon::A2_combinew);
      TmpInst.addOperand(Rdd);
      TmpInst.addOperand(Rss);
      TmpInst.addOperand(MCOperand::createReg(matchRegister(Reg2)));
      Inst = TmpInst;
    } else {
      Imm.setExpr(HexagonMCExpr::create(
          MCBinaryExpr::createSub(Imm.getExpr(),
                                  MCConstantExpr::create(1, Context), Context),
          Context));
      Inst.setOpcode(Hexagon::S2_asr_i_p_rnd);
    }
    break;
  }

  case Hexagon::A4_boundscheck: {
    MCOperand &Rs = Inst.getOperand(1);
    unsigned int RegNum = RI->getEncodingValue(Rs.getReg());
    if (RegNum & 1) { // Odd mapped to raw:hi, regpair is rodd:odd-1, like r3:2
      Inst.setOpcode(Hexagon::A4_boundscheck_hi);
      std::string Name =
          r + llvm::utostr(RegNum) + Colon + llvm::utostr(RegNum - 1);
      StringRef RegPair = Name;
      Rs.setReg(matchRegister(RegPair));
    } else { // raw:lo
      Inst.setOpcode(Hexagon::A4_boundscheck_lo);
      std::string Name =
          r + llvm::utostr(RegNum + 1) + Colon + llvm::utostr(RegNum);
      StringRef RegPair = Name;
      Rs.setReg(matchRegister(RegPair));
    }
    break;
  }

  case Hexagon::A2_addsp: {
    MCOperand &Rs = Inst.getOperand(1);
    unsigned int RegNum = RI->getEncodingValue(Rs.getReg());
    if (RegNum & 1) { // Odd mapped to raw:hi
      Inst.setOpcode(Hexagon::A2_addsph);
      std::string Name =
          r + llvm::utostr(RegNum) + Colon + llvm::utostr(RegNum - 1);
      StringRef RegPair = Name;
      Rs.setReg(matchRegister(RegPair));
    } else { // Even mapped raw:lo
      Inst.setOpcode(Hexagon::A2_addspl);
      std::string Name =
          r + llvm::utostr(RegNum + 1) + Colon + llvm::utostr(RegNum);
      StringRef RegPair = Name;
      Rs.setReg(matchRegister(RegPair));
    }
    break;
  }

  case Hexagon::M2_vrcmpys_s1: {
    MCOperand &Rt = Inst.getOperand(2);
    unsigned int RegNum = RI->getEncodingValue(Rt.getReg());
    if (RegNum & 1) { // Odd mapped to sat:raw:hi
      Inst.setOpcode(Hexagon::M2_vrcmpys_s1_h);
      std::string Name =
          r + llvm::utostr(RegNum) + Colon + llvm::utostr(RegNum - 1);
      StringRef RegPair = Name;
      Rt.setReg(matchRegister(RegPair));
    } else { // Even mapped sat:raw:lo
      Inst.setOpcode(Hexagon::M2_vrcmpys_s1_l);
      std::string Name =
          r + llvm::utostr(RegNum + 1) + Colon + llvm::utostr(RegNum);
      StringRef RegPair = Name;
      Rt.setReg(matchRegister(RegPair));
    }
    break;
  }

  case Hexagon::M2_vrcmpys_acc_s1: {
    MCInst TmpInst;
    MCOperand &Rxx = Inst.getOperand(0);
    MCOperand &Rss = Inst.getOperand(2);
    MCOperand &Rt = Inst.getOperand(3);
    unsigned int RegNum = RI->getEncodingValue(Rt.getReg());
    if (RegNum & 1) { // Odd mapped to sat:raw:hi
      TmpInst.setOpcode(Hexagon::M2_vrcmpys_acc_s1_h);
      std::string Name =
          r + llvm::utostr(RegNum) + Colon + llvm::utostr(RegNum - 1);
      StringRef RegPair = Name;
      Rt.setReg(matchRegister(RegPair));
    } else { // Even mapped sat:raw:lo
      TmpInst.setOpcode(Hexagon::M2_vrcmpys_acc_s1_l);
      std::string Name =
          r + llvm::utostr(RegNum + 1) + Colon + llvm::utostr(RegNum);
      StringRef RegPair = Name;
      Rt.setReg(matchRegister(RegPair));
    }
    // Registers are in different positions
    TmpInst.addOperand(Rxx);
    TmpInst.addOperand(Rxx);
    TmpInst.addOperand(Rss);
    TmpInst.addOperand(Rt);
    Inst = TmpInst;
    break;
  }

  case Hexagon::M2_vrcmpys_s1rp: {
    MCOperand &Rt = Inst.getOperand(2);
    unsigned int RegNum = RI->getEncodingValue(Rt.getReg());
    if (RegNum & 1) { // Odd mapped to rnd:sat:raw:hi
      Inst.setOpcode(Hexagon::M2_vrcmpys_s1rp_h);
      std::string Name =
          r + llvm::utostr(RegNum) + Colon + llvm::utostr(RegNum - 1);
      StringRef RegPair = Name;
      Rt.setReg(matchRegister(RegPair));
    } else { // Even mapped rnd:sat:raw:lo
      Inst.setOpcode(Hexagon::M2_vrcmpys_s1rp_l);
      std::string Name =
          r + llvm::utostr(RegNum + 1) + Colon + llvm::utostr(RegNum);
      StringRef RegPair = Name;
      Rt.setReg(matchRegister(RegPair));
    }
    break;
  }

  case Hexagon::S5_asrhub_rnd_sat_goodsyntax: {
    MCOperand &Imm = Inst.getOperand(2);
    int64_t Value;
    bool Absolute = Imm.getExpr()->evaluateAsAbsolute(Value);
    assert(Absolute);
    (void)Absolute;
    if (Value == 0)
      Inst.setOpcode(Hexagon::S2_vsathub);
    else {
      Imm.setExpr(HexagonMCExpr::create(
          MCBinaryExpr::createSub(Imm.getExpr(),
                                  MCConstantExpr::create(1, Context), Context),
          Context));
      Inst.setOpcode(Hexagon::S5_asrhub_rnd_sat);
    }
    break;
  }

  case Hexagon::S5_vasrhrnd_goodsyntax: {
    MCOperand &Rdd = Inst.getOperand(0);
    MCOperand &Rss = Inst.getOperand(1);
    MCOperand &Imm = Inst.getOperand(2);
    int64_t Value;
    bool Absolute = Imm.getExpr()->evaluateAsAbsolute(Value);
    assert(Absolute);
    (void)Absolute;
    if (Value == 0) {
      MCInst TmpInst;
      unsigned int RegPairNum = RI->getEncodingValue(Rss.getReg());
      std::string R1 = r + llvm::utostr(RegPairNum + 1);
      StringRef Reg1(R1);
      Rss.setReg(matchRegister(Reg1));
      // Add a new operand for the second register in the pair.
      std::string R2 = r + llvm::utostr(RegPairNum);
      StringRef Reg2(R2);
      TmpInst.setOpcode(Hexagon::A2_combinew);
      TmpInst.addOperand(Rdd);
      TmpInst.addOperand(Rss);
      TmpInst.addOperand(MCOperand::createReg(matchRegister(Reg2)));
      Inst = TmpInst;
    } else {
      Imm.setExpr(HexagonMCExpr::create(
          MCBinaryExpr::createSub(Imm.getExpr(),
                                  MCConstantExpr::create(1, Context), Context),
          Context));
      Inst.setOpcode(Hexagon::S5_vasrhrnd);
    }
    break;
  }

  case Hexagon::A2_not: {
    MCInst TmpInst;
    MCOperand &Rd = Inst.getOperand(0);
    MCOperand &Rs = Inst.getOperand(1);
    TmpInst.setOpcode(Hexagon::A2_subri);
    TmpInst.addOperand(Rd);
    TmpInst.addOperand(MCOperand::createExpr(
        HexagonMCExpr::create(MCConstantExpr::create(-1, Context), Context)));
    TmpInst.addOperand(Rs);
    Inst = TmpInst;
    break;
  }
  } // switch

  return Match_Success;
}


unsigned HexagonAsmParser::matchRegister(StringRef Name) {
  if (unsigned Reg = MatchRegisterName(Name))
    return Reg;
  return MatchRegisterAltName(Name);
}
