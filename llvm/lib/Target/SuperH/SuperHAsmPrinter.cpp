//===-- SuperHAsmPrinter.cpp - SH LLVM Assembly Printer ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format SuperH assembly language.
//
//===-----------------------------------------------------------------------===//

#include "MCTargetDesc/SuperHBaseInfo.h"
#include "MCTargetDesc/SuperHInstPrinter.h"
#include "MCTargetDesc/SuperHMCAsmInfo.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "MCTargetDesc/SuperHTargetStreamer.h"
#include "SuperH.h"
#include "SuperHConstantPoolValue.h"
#include "SuperHMCInstLower.h"
#include "TargetInfo/SuperHTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {

class SuperHAsmPrinter : public AsmPrinter {
public:
  static char ID;

private:

  /// MCP - Keep a pointer to constantpool entries of the current
  /// MachineFunction.
  const MachineConstantPool *MCP;

  /// InConstantPool - Maintain state when emitting a sequence of constant
  /// pool entries so we can properly mark them as data regions.
  bool InConstantPool = false;

	SuperHTargetStreamer &getTargetStreamer() {
		return static_cast<SuperHTargetStreamer &>(
			*OutStreamer->getTargetStreamer());
	}

public:
  explicit SuperHAsmPrinter(TargetMachine &TM,
                           std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID), MCP(nullptr) {}

  StringRef getPassName() const override { return "SuperH Assembly Printer"; }
  bool runOnMachineFunction(MachineFunction &F) override;

  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);
  void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);

  void emitFunctionBodyStart() override;
  void emitFunctionBodyEnd() override;
  void emitInstruction(const MachineInstr *MI) override;

  // We emit them ourselves.
  void emitConstantPool() override { }
  void emitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) override;

  static const char *getRegisterName(MCRegister Reg) {
    return SuperHInstPrinter::getRegisterName(Reg);
  }

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &O) override;
};

} // namespace

bool SuperHAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  MCP = MF.getConstantPool();
  return AsmPrinter::runOnMachineFunction(MF);
}




//===----------------------------------------------------------------------===//
//                                   Utilities
//===----------------------------------------------------------------------===//

// Convert a SuperH-specific constant pool modifier into the associated
// specifier.
static uint8_t getSpecifierFromModifier(SHCP::SHCPModifier Modifier) {
  switch (Modifier) {
  case SHCP::SHCPModifier::DIR:
  case SHCP::SHCPModifier::no_modifier:
    return SHII::MO_DIR;
  default:
    return SHII::MO_DIR;
  }
}




//===----------------------------------------------------------------------===//
//                                  Operands
//===----------------------------------------------------------------------===//

void SuperHAsmPrinter::printOperand(const MachineInstr *MI, int OpNo, raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNo);

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << StringRef(getRegisterName(MO.getReg())).lower();
    break;
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    break;
  case MachineOperand::MO_GlobalAddress:
    O << getSymbol(MO.getGlobal());
    break;
  case MachineOperand::MO_ExternalSymbol:
    O << *GetExternalSymbolSymbol(MO.getSymbolName());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    break;
  default:
    llvm_unreachable("Not implemented yet!");
  }
}

bool SuperHAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) {
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, O))
    return false;

  const MachineOperand &MO = MI->getOperand(OpNo);
  if (MO.getType() == MachineOperand::MO_GlobalAddress)
    PrintSymbolOperand(MO, O); // Print global symbols.
  else
    printOperand(MI, OpNo, O); // Fallback to ordinary cases.

  return false;
}

bool SuperHAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier

  const MachineOperand &MO = MI->getOperand(OpNo);

  // Print direct memory operands.
  if (MO.isGlobal() || MO.isSymbol() || MO.isMCSymbol()) {
    PrintSymbolOperand(MO, O);
    return false;
  }
  return false;
}




//===----------------------------------------------------------------------===//
//                                Constant Pool
//===----------------------------------------------------------------------===//


void SuperHAsmPrinter::emitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  const DataLayout &DL = getDataLayout();
  int Size = DL.getTypeAllocSize(MCPV->getType());

  MCSymbol *MCSym;
  SuperHConstantPoolValue *SCPV = static_cast<SuperHConstantPoolValue*>(MCPV);
  if (SCPV->isBlockAddress()) {
    const BlockAddress *BA =
      cast<SuperHConstantPoolConstant>(SCPV)->getBlockAddress();
    MCSym = GetBlockAddressSymbol(BA);
  } else if (SCPV->isGlobalValue()) {
    const GlobalValue *GV = cast<SuperHConstantPoolConstant>(SCPV)->getGV();
    MCSym = getSymbolPreferLocal(*GV);
  } else {
    assert(SCPV->isExtSymbol() && "unrecognized constant pool value");
    auto Sym = cast<SuperHConstantPoolSymbol>(SCPV)->getSymbol();
    MCSym = GetExternalSymbolSymbol(Sym);
  }

  // Create an MCSymbol for the reference.
  const MCExpr *Expr = MCSymbolRefExpr::create(
    MCSym, 
    OutContext
  );
  OutStreamer->emitValue(Expr, Size);
}

void SuperHAsmPrinter::emitFunctionBodyStart() {
  AsmPrinter::emitFunctionBodyStart();
}

void SuperHAsmPrinter::emitFunctionBodyEnd() {

  // Make sure to terminate any constant pools that were at the end
  // of the function.
  if (!InConstantPool)
    return;

  InConstantPool = false;
  OutStreamer->emitDataRegion(MCDR_DataRegionEnd);
}


void SuperHAsmPrinter::emitInstruction(const MachineInstr *MI) {
  const SuperHSubtarget &STI = MF->getSubtarget<SuperHSubtarget>();
  const DataLayout &DL = getDataLayout();
  MCTargetStreamer &TS = *OutStreamer->getTargetStreamer();
  SuperHTargetStreamer &STS = static_cast<SuperHTargetStreamer &>(TS);

  // If we just ended a constant pool, mark it as such.
  if (InConstantPool && MI->getOpcode() != SH::CONSTPOOL_ENTRY) {
    OutStreamer->emitDataRegion(MCDR_DataRegionEnd);
    InConstantPool = false;
  }

  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  default: {
    SuperHMCInstLower MCInstLowering(OutContext, *this);

    MCInst I;
    MCInstLowering.lowerInstruction(*MI, I);
    EmitToStreamer(*OutStreamer, I);
    return;
  }

  case SH::CONSTPOOL_ENTRY: {
    unsigned LabelId = (unsigned)MI->getOperand(0).getImm();
    unsigned CPIdx   = (unsigned)MI->getOperand(1).getIndex();

    // If this is the first entry of the pool, mark it.
    if (!InConstantPool) {
      OutStreamer->emitDataRegion(MCDR_DataRegion);
      InConstantPool = true;
    }

    OutStreamer->emitLabel(GetCPISymbol(LabelId));

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPIdx];
    if (MCPE.isMachineConstantPoolEntry())
      emitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    else
      emitGlobalConstant(DL, MCPE.Val.ConstVal);
    return;
  }
  }
}

char SuperHAsmPrinter::ID = 0;
INITIALIZE_PASS(SuperHAsmPrinter, "sh-asm-printer", "SuperH Assembly Printer", false, false)

// Force static initialization.
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeSuperHAsmPrinter() {
  RegisterAsmPrinter<SuperHAsmPrinter> X(getTheSuperHTarget());
  RegisterAsmPrinter<SuperHAsmPrinter> Y(getTheSuperHLETarget());
}
