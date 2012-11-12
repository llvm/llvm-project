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
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
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

    virtual MachineLocation getDebugValueLocation(const MachineInstr *MI) const;
  };
} // end of anonymous namespace

#include "SparcGenAsmWriter.inc"

void SparcAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand (opNum);
  bool CloseParen = false;
  if (MI->getOpcode() == SP::SETHIi && !MO.isReg() && !MO.isImm()) {
    O << "%hi(";
    CloseParen = true;
  } else if ((MI->getOpcode() == SP::ORri || MI->getOpcode() == SP::ADDri) &&
             !MO.isReg() && !MO.isImm()) {
    O << "%lo(";
    CloseParen = true;
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
    O << *Mang->getSymbol(MO.getGlobal());
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
  if (MI->getOperand(opNum+1).isGlobal() ||
      MI->getOperand(opNum+1).isCPI()) {
    O << "%lo(";
    printOperand(MI, opNum+1, O);
    O << ")";
  } else {
    printOperand(MI, opNum+1, O);
  }
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

MachineLocation SparcAsmPrinter::
getDebugValueLocation(const MachineInstr *MI) const {
  assert(MI->getNumOperands() == 4 && "Invalid number of operands!");
  assert(MI->getOperand(0).isReg() && MI->getOperand(1).isImm() &&
         "Unexpected MachineOperand types");
  return MachineLocation(MI->getOperand(0).getReg(),
                         MI->getOperand(1).getImm());
}

// Force static initialization.
extern "C" void LLVMInitializeSparcAsmPrinter() { 
  RegisterAsmPrinter<SparcAsmPrinter> X(TheSparcTarget);
  RegisterAsmPrinter<SparcAsmPrinter> Y(TheSparcV9Target);
}
