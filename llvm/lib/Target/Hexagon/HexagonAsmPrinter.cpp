//===-- HexagonAsmPrinter.cpp - Print machine instrs to Hexagon assembly --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to Hexagon assembly language. This printer is
// the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonAsmPrinter.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "MCTargetDesc/HexagonInstPrinter.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCShuffler.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

static cl::opt<bool> AlignCalls(
         "hexagon-align-calls", cl::Hidden, cl::init(true),
          cl::desc("Insert falign after call instruction for Hexagon target"));

HexagonAsmPrinter::HexagonAsmPrinter(TargetMachine &TM,
                                     std::unique_ptr<MCStreamer> Streamer)
    : AsmPrinter(TM, std::move(Streamer)), Subtarget(nullptr) {}

void HexagonAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo,
                                    raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNo);

  switch (MO.getType()) {
  default: llvm_unreachable ("<unknown operand type>");
  case MachineOperand::MO_Register:
    O << HexagonInstPrinter::getRegisterName(MO.getReg());
    return;
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    MO.getMBB()->getSymbol()->print(O, MAI);
    return;
  case MachineOperand::MO_ConstantPoolIndex:
    GetCPISymbol(MO.getIndex())->print(O, MAI);
    return;
  case MachineOperand::MO_GlobalAddress:
    // Computing the address of a global symbol, not calling it.
    getSymbol(MO.getGlobal())->print(O, MAI);
    printOffset(MO.getOffset(), O);
    return;
  }
}

//
// isBlockOnlyReachableByFallthrough - We need to override this since the
// default AsmPrinter does not print labels for any basic block that
// is only reachable by a fall through. That works for all cases except
// for the case in which the basic block is reachable by a fall through but
// through an indirect from a jump table. In this case, the jump table
// will contain a label not defined by AsmPrinter.
//
bool HexagonAsmPrinter::
isBlockOnlyReachableByFallthrough(const MachineBasicBlock *MBB) const {
  if (MBB->hasAddressTaken()) {
    return false;
  }
  return AsmPrinter::isBlockOnlyReachableByFallthrough(MBB);
}


/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool HexagonAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                        unsigned AsmVariant,
                                        const char *ExtraCode,
                                        raw_ostream &OS) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNo, AsmVariant, ExtraCode, OS);
    case 'c': // Don't print "$" before a global var name or constant.
      // Hexagon never has a prefix.
      printOperand(MI, OpNo, OS);
      return false;
    case 'L': // Write second word of DImode reference.
      // Verify that this operand has two consecutive registers.
      if (!MI->getOperand(OpNo).isReg() ||
          OpNo+1 == MI->getNumOperands() ||
          !MI->getOperand(OpNo+1).isReg())
        return true;
      ++OpNo;   // Return the high-part.
      break;
    case 'I':
      // Write 'i' if an integer constant, otherwise nothing.  Used to print
      // addi vs add, etc.
      if (MI->getOperand(OpNo).isImm())
        OS << "i";
      return false;
    }
  }

  printOperand(MI, OpNo, OS);
  return false;
}

bool HexagonAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNo, unsigned AsmVariant,
                                            const char *ExtraCode,
                                            raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.

  const MachineOperand &Base  = MI->getOperand(OpNo);
  const MachineOperand &Offset = MI->getOperand(OpNo+1);

  if (Base.isReg())
    printOperand(MI, OpNo, O);
  else
    llvm_unreachable("Unimplemented");

  if (Offset.isImm()) {
    if (Offset.getImm())
      O << " + #" << Offset.getImm();
  }
  else
    llvm_unreachable("Unimplemented");

  return false;
}


/// printMachineInstruction -- Print out a single Hexagon MI in Darwin syntax to
/// the current output stream.
///
void HexagonAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  MCInst MCB;
  MCB.setOpcode(Hexagon::BUNDLE);
  MCB.addOperand(MCOperand::createImm(0));

  if (MI->isBundle()) {
    const MachineBasicBlock* MBB = MI->getParent();
    MachineBasicBlock::const_instr_iterator MII = MI->getIterator();
    unsigned IgnoreCount = 0;

    for (++MII; MII != MBB->instr_end() && MII->isInsideBundle(); ++MII) {
      if (MII->getOpcode() == TargetOpcode::DBG_VALUE ||
          MII->getOpcode() == TargetOpcode::IMPLICIT_DEF)
        ++IgnoreCount;
      else {
        HexagonLowerToMC(&*MII, MCB, *this);
      }
    }
  }
  else {
    HexagonLowerToMC(MI, MCB, *this);
    HexagonMCInstrInfo::padEndloop(OutStreamer->getContext(), MCB);
  }
  // Examine the packet and try to find instructions that can be converted
  // to compounds.
  HexagonMCInstrInfo::tryCompound(*Subtarget->getInstrInfo(),
                                  OutStreamer->getContext(), MCB);
  // Examine the packet and convert pairs of instructions to duplex
  // instructions when possible.
  SmallVector<DuplexCandidate, 8> possibleDuplexes;
  possibleDuplexes = HexagonMCInstrInfo::getDuplexPossibilties(
      *Subtarget->getInstrInfo(), MCB);
  HexagonMCShuffle(*Subtarget->getInstrInfo(), *Subtarget,
                   OutStreamer->getContext(), MCB, possibleDuplexes);
  EmitToStreamer(*OutStreamer, MCB);
}

extern "C" void LLVMInitializeHexagonAsmPrinter() {
  RegisterAsmPrinter<HexagonAsmPrinter> X(TheHexagonTarget);
}
