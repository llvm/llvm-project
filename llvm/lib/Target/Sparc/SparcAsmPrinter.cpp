//===-- SparcAsmPrinter.cpp - Sparc LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format SPARC assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "Sparc.h"
#include "SparcInstrInfo.h"
#include "SparcTargetMachine.h"
#include "MCTargetDesc/SparcBaseInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/Mangler.h"
using namespace llvm;

namespace {
  class SparcAsmPrinter : public AsmPrinter {
  public:
    explicit SparcAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : AsmPrinter(TM, Streamer) {}

    virtual const char *getPassName() const {
      return "Sparc Assembly Printer";
    }

    void printOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);
    void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &OS,
                         const char *Modifier = 0);
    void printCCOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);

    virtual void EmitFunctionBodyStart();
    virtual void EmitInstruction(const MachineInstr *MI) {
      SmallString<128> Str;
      raw_svector_ostream OS(Str);
      printInstruction(MI, OS);
      OutStreamer.EmitRawText(OS.str());
    }
    void printInstruction(const MachineInstr *MI, raw_ostream &OS);// autogen'd.
    static const char *getRegisterName(unsigned RegNo);

    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode,
                         raw_ostream &O);
    bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &O);

    bool printGetPCX(const MachineInstr *MI, unsigned OpNo, raw_ostream &OS);

    virtual bool isBlockOnlyReachableByFallthrough(const MachineBasicBlock *MBB)
                       const;
    void EmitGlobalRegisterDecl(unsigned reg) {
      SmallString<128> Str;
      raw_svector_ostream OS(Str);
      OS << "\t.register "
         << "%" << StringRef(getRegisterName(reg)).lower()
         << ", "
         << ((reg == SP::G6 || reg == SP::G7)? "#ignore" : "#scratch");
      OutStreamer.EmitRawText(OS.str());
    }

  };
} // end of anonymous namespace

#include "SparcGenAsmWriter.inc"

void SparcAsmPrinter::EmitFunctionBodyStart() {
  if (!TM.getSubtarget<SparcSubtarget>().is64Bit())
    return;

  const MachineRegisterInfo &MRI = MF->getRegInfo();
  const unsigned globalRegs[] = { SP::G2, SP::G3, SP::G6, SP::G7, 0 };
  for (unsigned i = 0; globalRegs[i] != 0; ++i) {
    unsigned reg = globalRegs[i];
    if (MRI.use_empty(reg))
      continue;
    EmitGlobalRegisterDecl(reg);
  }
}

void SparcAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand (opNum);
  unsigned TF = MO.getTargetFlags();
#ifndef NDEBUG
  // Verify the target flags.
  if (MO.isGlobal() || MO.isSymbol() || MO.isCPI()) {
    if (MI->getOpcode() == SP::CALL)
      assert(TF == SPII::MO_NO_FLAG &&
             "Cannot handle target flags on call address");
    else if (MI->getOpcode() == SP::SETHIi)
      assert((TF == SPII::MO_HI || TF == SPII::MO_H44 || TF == SPII::MO_HH
              || TF == SPII::MO_TLS_GD_HI22
              || TF == SPII::MO_TLS_LDM_HI22
              || TF == SPII::MO_TLS_LDO_HIX22
              || TF == SPII::MO_TLS_IE_HI22
              || TF == SPII::MO_TLS_LE_HIX22) &&
             "Invalid target flags for address operand on sethi");
    else if (MI->getOpcode() == SP::TLS_CALL)
      assert((TF == SPII::MO_NO_FLAG
              || TF == SPII::MO_TLS_GD_CALL
              || TF == SPII::MO_TLS_LDM_CALL) &&
             "Cannot handle target flags on tls call address");
    else if (MI->getOpcode() == SP::TLS_ADDrr)
      assert((TF == SPII::MO_TLS_GD_ADD || TF == SPII::MO_TLS_LDM_ADD
              || TF == SPII::MO_TLS_LDO_ADD || TF == SPII::MO_TLS_IE_ADD) &&
             "Cannot handle target flags on add for TLS");
    else if (MI->getOpcode() == SP::TLS_LDrr)
      assert(TF == SPII::MO_TLS_IE_LD &&
             "Cannot handle target flags on ld for TLS");
    else if (MI->getOpcode() == SP::TLS_LDXrr)
      assert(TF == SPII::MO_TLS_IE_LDX &&
             "Cannot handle target flags on ldx for TLS");
    else if (MI->getOpcode() == SP::XORri)
      assert((TF == SPII::MO_TLS_LDO_LOX10 || TF == SPII::MO_TLS_LE_LOX10) &&
             "Cannot handle target flags on xor for TLS");
    else
      assert((TF == SPII::MO_LO || TF == SPII::MO_M44 || TF == SPII::MO_L44
              || TF == SPII::MO_HM
              || TF == SPII::MO_TLS_GD_LO10
              || TF == SPII::MO_TLS_LDM_LO10
              || TF == SPII::MO_TLS_IE_LO10 ) &&
             "Invalid target flags for small address operand");
  }
#endif

  bool CloseParen = true;
  switch (TF) {
  default:
      llvm_unreachable("Unknown target flags on operand");
  case SPII::MO_NO_FLAG:
    CloseParen = false;
    break;
  case SPII::MO_LO:  O << "%lo(";  break;
  case SPII::MO_HI:  O << "%hi(";  break;
  case SPII::MO_H44: O << "%h44("; break;
  case SPII::MO_M44: O << "%m44("; break;
  case SPII::MO_L44: O << "%l44("; break;
  case SPII::MO_HH:  O << "%hh(";  break;
  case SPII::MO_HM:  O << "%hm(";  break;
  case SPII::MO_TLS_GD_HI22:   O << "%tgd_hi22(";   break;
  case SPII::MO_TLS_GD_LO10:   O << "%tgd_lo10(";   break;
  case SPII::MO_TLS_GD_ADD:    O << "%tgd_add(";    break;
  case SPII::MO_TLS_GD_CALL:   O << "%tgd_call(";   break;
  case SPII::MO_TLS_LDM_HI22:  O << "%tldm_hi22(";  break;
  case SPII::MO_TLS_LDM_LO10:  O << "%tldm_lo10(";  break;
  case SPII::MO_TLS_LDM_ADD:   O << "%tldm_add(";   break;
  case SPII::MO_TLS_LDM_CALL:  O << "%tldm_call(";  break;
  case SPII::MO_TLS_LDO_HIX22: O << "%tldo_hix22("; break;
  case SPII::MO_TLS_LDO_LOX10: O << "%tldo_lox10("; break;
  case SPII::MO_TLS_LDO_ADD:   O << "%tldo_add(";   break;
  case SPII::MO_TLS_IE_HI22:   O << "%tie_hi22(";   break;
  case SPII::MO_TLS_IE_LO10:   O << "%tie_lo10(";   break;
  case SPII::MO_TLS_IE_LD:     O << "%tie_ld(";     break;
  case SPII::MO_TLS_IE_LDX:    O << "%tie_ldx(";    break;
  case SPII::MO_TLS_IE_ADD:    O << "%tie_add(";    break;
  case SPII::MO_TLS_LE_HIX22:  O << "%tle_hix22(";  break;
  case SPII::MO_TLS_LE_LOX10:  O << "%tle_lox10(";   break;
  }

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << "%" << StringRef(getRegisterName(MO.getReg())).lower();
    break;

  case MachineOperand::MO_Immediate:
    O << (int)MO.getImm();
    break;
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_GlobalAddress:
    O << *getSymbol(MO.getGlobal());
    break;
  case MachineOperand::MO_BlockAddress:
    O <<  GetBlockAddressSymbol(MO.getBlockAddress())->getName();
    break;
  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    O << MAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << "_"
      << MO.getIndex();
    break;
  default:
    llvm_unreachable("<unknown operand type>");
  }
  if (CloseParen) O << ")";
}

void SparcAsmPrinter::printMemOperand(const MachineInstr *MI, int opNum,
                                      raw_ostream &O, const char *Modifier) {
  printOperand(MI, opNum, O);

  // If this is an ADD operand, emit it like normal operands.
  if (Modifier && !strcmp(Modifier, "arith")) {
    O << ", ";
    printOperand(MI, opNum+1, O);
    return;
  }

  if (MI->getOperand(opNum+1).isReg() &&
      MI->getOperand(opNum+1).getReg() == SP::G0)
    return;   // don't print "+%g0"
  if (MI->getOperand(opNum+1).isImm() &&
      MI->getOperand(opNum+1).getImm() == 0)
    return;   // don't print "+0"

  O << "+";
  printOperand(MI, opNum+1, O);
}

bool SparcAsmPrinter::printGetPCX(const MachineInstr *MI, unsigned opNum,
                                  raw_ostream &O) {
  std::string operand = "";
  const MachineOperand &MO = MI->getOperand(opNum);
  switch (MO.getType()) {
  default: llvm_unreachable("Operand is not a register");
  case MachineOperand::MO_Register:
    assert(TargetRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           "Operand is not a physical register ");
    assert(MO.getReg() != SP::O7 &&
           "%o7 is assigned as destination for getpcx!");
    operand = "%" + StringRef(getRegisterName(MO.getReg())).lower();
    break;
  }

  unsigned mfNum = MI->getParent()->getParent()->getFunctionNumber();
  unsigned bbNum = MI->getParent()->getNumber();

  O << '\n' << ".LLGETPCH" << mfNum << '_' << bbNum << ":\n";
  O << "\tcall\t.LLGETPC" << mfNum << '_' << bbNum << '\n' ;

  O << "\t  sethi\t"
    << "%hi(_GLOBAL_OFFSET_TABLE_+(.-.LLGETPCH" << mfNum << '_' << bbNum
    << ")), "  << operand << '\n' ;

  O << ".LLGETPC" << mfNum << '_' << bbNum << ":\n" ;
  O << "\tor\t" << operand
    << ", %lo(_GLOBAL_OFFSET_TABLE_+(.-.LLGETPCH" << mfNum << '_' << bbNum
    << ")), " << operand << '\n';
  O << "\tadd\t" << operand << ", %o7, " << operand << '\n';

  return true;
}

void SparcAsmPrinter::printCCOperand(const MachineInstr *MI, int opNum,
                                     raw_ostream &O) {
  int CC = (int)MI->getOperand(opNum).getImm();
  O << SPARCCondCodeToString((SPCC::CondCodes)CC);
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool SparcAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      unsigned AsmVariant,
                                      const char *ExtraCode,
                                      raw_ostream &O) {
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNo, AsmVariant, ExtraCode, O);
    case 'r':
     break;
    }
  }

  printOperand(MI, OpNo, O);

  return false;
}

bool SparcAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNo, unsigned AsmVariant,
                                            const char *ExtraCode,
                                            raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true;  // Unknown modifier

  O << '[';
  printMemOperand(MI, OpNo, O);
  O << ']';

  return false;
}

/// isBlockOnlyReachableByFallthough - Return true if the basic block has
/// exactly one predecessor and the control transfer mechanism between
/// the predecessor and this block is a fall-through.
///
/// This overrides AsmPrinter's implementation to handle delay slots.
bool SparcAsmPrinter::
isBlockOnlyReachableByFallthrough(const MachineBasicBlock *MBB) const {
  // If this is a landing pad, it isn't a fall through.  If it has no preds,
  // then nothing falls through to it.
  if (MBB->isLandingPad() || MBB->pred_empty())
    return false;

  // If there isn't exactly one predecessor, it can't be a fall through.
  MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(), PI2 = PI;
  ++PI2;
  if (PI2 != MBB->pred_end())
    return false;

  // The predecessor has to be immediately before this block.
  const MachineBasicBlock *Pred = *PI;

  if (!Pred->isLayoutSuccessor(MBB))
    return false;

  // Check if the last terminator is an unconditional branch.
  MachineBasicBlock::const_iterator I = Pred->end();
  while (I != Pred->begin() && !(--I)->isTerminator())
    ; // Noop
  return I == Pred->end() || !I->isBarrier();
}

// Force static initialization.
extern "C" void LLVMInitializeSparcAsmPrinter() {
  RegisterAsmPrinter<SparcAsmPrinter> X(TheSparcTarget);
  RegisterAsmPrinter<SparcAsmPrinter> Y(TheSparcV9Target);
}
