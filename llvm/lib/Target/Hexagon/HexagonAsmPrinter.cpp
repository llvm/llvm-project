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

#define DEBUG_TYPE "asm-printer"
#include "Hexagon.h"
#include "HexagonAsmPrinter.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonMCInst.h"
#include "HexagonTargetMachine.h"
#include "HexagonSubtarget.h"
#include "InstPrinter/HexagonInstPrinter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/Mangler.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

static cl::opt<bool> AlignCalls(
         "hexagon-align-calls", cl::Hidden, cl::init(true),
          cl::desc("Insert falign after call instruction for Hexagon target"));

void HexagonAsmPrinter::EmitAlignment(unsigned NumBits,
                                      const GlobalValue *GV) const {
  // For basic block level alignment, use ".falign".
  if (!GV) {
    OutStreamer.EmitRawText(StringRef("\t.falign"));
    return;
  }

  AsmPrinter::EmitAlignment(NumBits, GV);
}

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
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_JumpTableIndex:
    O << *GetJTISymbol(MO.getIndex());
    // FIXME: PIC relocation model.
    return;
  case MachineOperand::MO_ConstantPoolIndex:
    O << *GetCPISymbol(MO.getIndex());
    return;
  case MachineOperand::MO_ExternalSymbol:
    O << *GetExternalSymbolSymbol(MO.getSymbolName());
    return;
  case MachineOperand::MO_GlobalAddress:
    // Computing the address of a global symbol, not calling it.
    O << *Mang->getSymbol(MO.getGlobal());
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

void HexagonAsmPrinter::printPredicateOperand(const MachineInstr *MI,
                                              unsigned OpNo,
                                              raw_ostream &O) {
  llvm_unreachable("Unimplemented");
}


/// printMachineInstruction -- Print out a single Hexagon MI in Darwin syntax to
/// the current output stream.
///
void HexagonAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  if (MI->isBundle()) {
    std::vector<const MachineInstr*> BundleMIs;

    const MachineBasicBlock *MBB = MI->getParent();
    MachineBasicBlock::const_instr_iterator MII = MI;
    ++MII;
    unsigned int IgnoreCount = 0;
    while (MII != MBB->end() && MII->isInsideBundle()) {
      const MachineInstr *MInst = MII;
      if (MInst->getOpcode() == TargetOpcode::DBG_VALUE ||
          MInst->getOpcode() == TargetOpcode::IMPLICIT_DEF) {
          IgnoreCount++;
          ++MII;
          continue;
      }
      //BundleMIs.push_back(&*MII);
      BundleMIs.push_back(MInst);
      ++MII;
    }
    unsigned Size = BundleMIs.size();
    assert((Size+IgnoreCount) == MI->getBundleSize() && "Corrupt Bundle!");
    for (unsigned Index = 0; Index < Size; Index++) {
      HexagonMCInst MCI;
      MCI.setStartPacket(Index == 0);
      MCI.setEndPacket(Index == (Size-1));

      HexagonLowerToMC(BundleMIs[Index], MCI, *this);
      OutStreamer.EmitInstruction(MCI);
    }
  }
  else {
    HexagonMCInst MCI;
    if (MI->getOpcode() == Hexagon::ENDLOOP0) {
      MCI.setStartPacket(true);
      MCI.setEndPacket(true);
    }
    HexagonLowerToMC(MI, MCI, *this);
    OutStreamer.EmitInstruction(MCI);
  }

  return;
}

/// PrintUnmangledNameSafely - Print out the printable characters in the name.
/// Don't print things like \n or \0.
// static void PrintUnmangledNameSafely(const Value *V, raw_ostream &OS) {
//   for (const char *Name = V->getNameStart(), *E = Name+V->getNameLen();
//        Name != E; ++Name)
//     if (isprint(*Name))
//       OS << *Name;
// }


void HexagonAsmPrinter::printAddrModeBasePlusOffset(const MachineInstr *MI,
                                                    int OpNo, raw_ostream &O) {
  const MachineOperand &MO1 = MI->getOperand(OpNo);
  const MachineOperand &MO2 = MI->getOperand(OpNo+1);

  O << HexagonInstPrinter::getRegisterName(MO1.getReg())
    << " + #"
    << MO2.getImm();
}


void HexagonAsmPrinter::printGlobalOperand(const MachineInstr *MI, int OpNo,
                                           raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  assert( (MO.getType() == MachineOperand::MO_GlobalAddress) &&
         "Expecting global address");

  O << *Mang->getSymbol(MO.getGlobal());
  if (MO.getOffset() != 0) {
    O << " + ";
    O << MO.getOffset();
  }
}

void HexagonAsmPrinter::printJumpTable(const MachineInstr *MI, int OpNo,
                                       raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  assert( (MO.getType() == MachineOperand::MO_JumpTableIndex) && 
           "Expecting jump table index");

  // Hexagon_TODO: Do we need name mangling?
  O << *GetJTISymbol(MO.getIndex());
}

void HexagonAsmPrinter::printConstantPool(const MachineInstr *MI, int OpNo,
                                       raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  assert( (MO.getType() == MachineOperand::MO_ConstantPoolIndex) &&
          "Expecting constant pool index");

  // Hexagon_TODO: Do we need name mangling?
  O << *GetCPISymbol(MO.getIndex());
}

static MCInstPrinter *createHexagonMCInstPrinter(const Target &T,
                                                 unsigned SyntaxVariant,
                                                 const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI,
                                                 const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return(new HexagonInstPrinter(MAI, MII, MRI));
  else
   return NULL;
}

extern "C" void LLVMInitializeHexagonAsmPrinter() {
  RegisterAsmPrinter<HexagonAsmPrinter> X(TheHexagonTarget);

  TargetRegistry::RegisterMCInstPrinter(TheHexagonTarget,
                                        createHexagonMCInstPrinter);
}
