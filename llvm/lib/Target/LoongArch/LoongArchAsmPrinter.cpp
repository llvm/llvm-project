//===- LoongArchAsmPrinter.cpp - LoongArch LLVM Assembly Printer -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format LoongArch assembly language.
//
//===----------------------------------------------------------------------===//

#include "LoongArchAsmPrinter.h"
#include "LoongArch.h"
#include "LoongArchMachineFunctionInfo.h"
#include "MCTargetDesc/LoongArchInstPrinter.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "TargetInfo/LoongArchTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-asm-printer"

cl::opt<bool> LArchAnnotateTableJump(
    "loongarch-annotate-tablejump", cl::Hidden,
    cl::desc(
        "Annotate table jump instruction to correlate it with the jump table."),
    cl::init(false));

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "LoongArchGenMCPseudoLowering.inc"

void LoongArchAsmPrinter::emitInstruction(const MachineInstr *MI) {
  LoongArch_MC::verifyInstructionPredicates(
      MI->getOpcode(), getSubtargetInfo().getFeatureBits());

  // Do any auto-generated pseudo lowerings.
  if (MCInst OutInst; lowerPseudoInstExpansion(MI, OutInst)) {
    EmitToStreamer(*OutStreamer, OutInst);
    return;
  }

  switch (MI->getOpcode()) {
  case TargetOpcode::STATEPOINT:
    LowerSTATEPOINT(*MI);
    return;
  case TargetOpcode::PATCHABLE_FUNCTION_ENTER:
    LowerPATCHABLE_FUNCTION_ENTER(*MI);
    return;
  case TargetOpcode::PATCHABLE_FUNCTION_EXIT:
    LowerPATCHABLE_FUNCTION_EXIT(*MI);
    return;
  case TargetOpcode::PATCHABLE_TAIL_CALL:
    LowerPATCHABLE_TAIL_CALL(*MI);
    return;
  }

  MCInst TmpInst;
  if (!lowerLoongArchMachineInstrToMCInst(MI, TmpInst, *this))
    EmitToStreamer(*OutStreamer, TmpInst);
}

bool LoongArchAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                          const char *ExtraCode,
                                          raw_ostream &OS) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
    return false;

  const MachineOperand &MO = MI->getOperand(OpNo);
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      return true; // Unknown modifier.
    case 'z':      // Print $zero register if zero, regular printing otherwise.
      if (MO.isImm() && MO.getImm() == 0) {
        OS << '$' << LoongArchInstPrinter::getRegisterName(LoongArch::R0);
        return false;
      }
      break;
    case 'u': // Print LASX registers.
    case 'w': // Print LSX registers.
    {
      // If the operand is an LASX, LSX or floating point register, print the
      // name of LASX or LSX register with the same index in that register
      // class.
      unsigned RegID = MO.getReg().id(), FirstReg;
      if (RegID >= LoongArch::XR0 && RegID <= LoongArch::XR31)
        FirstReg = LoongArch::XR0;
      else if (RegID >= LoongArch::VR0 && RegID <= LoongArch::VR31)
        FirstReg = LoongArch::VR0;
      else if (RegID >= LoongArch::F0_64 && RegID <= LoongArch::F31_64)
        FirstReg = LoongArch::F0_64;
      else if (RegID >= LoongArch::F0 && RegID <= LoongArch::F31)
        FirstReg = LoongArch::F0;
      else
        return true;
      OS << '$'
         << LoongArchInstPrinter::getRegisterName(
                RegID - FirstReg +
                (ExtraCode[0] == 'u' ? LoongArch::XR0 : LoongArch::VR0));
      return false;
    }
      // TODO: handle other extra codes if any.
    }
  }

  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    OS << MO.getImm();
    return false;
  case MachineOperand::MO_Register:
    OS << '$' << LoongArchInstPrinter::getRegisterName(MO.getReg());
    return false;
  case MachineOperand::MO_GlobalAddress:
    PrintSymbolOperand(MO, OS);
    return false;
  default:
    llvm_unreachable("not implemented");
  }

  return true;
}

bool LoongArchAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                                unsigned OpNo,
                                                const char *ExtraCode,
                                                raw_ostream &OS) {
  // TODO: handle extra code.
  if (ExtraCode)
    return true;

  // We only support memory operands like "Base + Offset", where base must be a
  // register, and offset can be a register or an immediate value.
  const MachineOperand &BaseMO = MI->getOperand(OpNo);
  // Base address must be a register.
  if (!BaseMO.isReg())
    return true;
  // Print the base address register.
  OS << "$" << LoongArchInstPrinter::getRegisterName(BaseMO.getReg());
  // Print the offset operand.
  const MachineOperand &OffsetMO = MI->getOperand(OpNo + 1);
  MCOperand MCO;
  if (!lowerOperand(OffsetMO, MCO))
    return true;
  if (OffsetMO.isReg())
    OS << ", $" << LoongArchInstPrinter::getRegisterName(OffsetMO.getReg());
  else if (OffsetMO.isImm())
    OS << ", " << OffsetMO.getImm();
  else if (OffsetMO.isGlobal() || OffsetMO.isBlockAddress() ||
           OffsetMO.isMCSymbol()) {
    OS << ", ";
    MAI->printExpr(OS, *MCO.getExpr());
  } else
    return true;

  return false;
}

void LoongArchAsmPrinter::LowerSTATEPOINT(const MachineInstr &MI) {
  StatepointOpers SOpers(&MI);
  if (unsigned PatchBytes = SOpers.getNumPatchBytes()) {
    assert(PatchBytes % 4 == 0 && "Invalid number of NOP bytes requested!");
    emitNops(PatchBytes / 4);
  } else {
    // Lower call target and choose correct opcode.
    const MachineOperand &CallTarget = SOpers.getCallTarget();
    MCOperand CallTargetMCOp;
    switch (CallTarget.getType()) {
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
      lowerOperand(CallTarget, CallTargetMCOp);
      EmitToStreamer(*OutStreamer,
                     MCInstBuilder(LoongArch::BL).addOperand(CallTargetMCOp));
      break;
    case MachineOperand::MO_Immediate:
      CallTargetMCOp = MCOperand::createImm(CallTarget.getImm());
      EmitToStreamer(*OutStreamer,
                     MCInstBuilder(LoongArch::BL).addOperand(CallTargetMCOp));
      break;
    case MachineOperand::MO_Register:
      CallTargetMCOp = MCOperand::createReg(CallTarget.getReg());
      EmitToStreamer(*OutStreamer, MCInstBuilder(LoongArch::JIRL)
                                       .addReg(LoongArch::R1)
                                       .addOperand(CallTargetMCOp)
                                       .addImm(0));
      break;
    default:
      llvm_unreachable("Unsupported operand type in statepoint call target");
      break;
    }
  }

  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);
  SM.recordStatepoint(*MILabel, MI);
}

void LoongArchAsmPrinter::LowerPATCHABLE_FUNCTION_ENTER(
    const MachineInstr &MI) {
  const Function &F = MF->getFunction();
  if (F.hasFnAttribute("patchable-function-entry")) {
    unsigned Num;
    if (F.getFnAttribute("patchable-function-entry")
            .getValueAsString()
            .getAsInteger(10, Num))
      return;
    emitNops(Num);
    return;
  }

  emitSled(MI, SledKind::FUNCTION_ENTER);
}

void LoongArchAsmPrinter::LowerPATCHABLE_FUNCTION_EXIT(const MachineInstr &MI) {
  emitSled(MI, SledKind::FUNCTION_EXIT);
}

void LoongArchAsmPrinter::LowerPATCHABLE_TAIL_CALL(const MachineInstr &MI) {
  emitSled(MI, SledKind::TAIL_CALL);
}

void LoongArchAsmPrinter::emitSled(const MachineInstr &MI, SledKind Kind) {
  // For loongarch64 we want to emit the following pattern:
  //
  // .Lxray_sled_beginN:
  //   B .Lxray_sled_endN
  //   11 NOPs (44 bytes)
  // .Lxray_sled_endN:
  //
  // We need the extra bytes because at runtime they may be used for the
  // actual pattern defined at compiler-rt/lib/xray/xray_loongarch64.cpp.
  // The count here should be adjusted accordingly if the implementation
  // changes.
  const int8_t NoopsInSledCount = 11;
  OutStreamer->emitCodeAlignment(Align(4), &getSubtargetInfo());
  MCSymbol *BeginOfSled = OutContext.createTempSymbol("xray_sled_begin");
  MCSymbol *EndOfSled = OutContext.createTempSymbol("xray_sled_end");
  OutStreamer->emitLabel(BeginOfSled);
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(LoongArch::B)
                     .addExpr(MCSymbolRefExpr::create(EndOfSled, OutContext)));
  emitNops(NoopsInSledCount);
  OutStreamer->emitLabel(EndOfSled);
  recordSled(BeginOfSled, MI, Kind, 2);
}

void LoongArchAsmPrinter::emitJumpTableInfo() {
  AsmPrinter::emitJumpTableInfo();

  if (!LArchAnnotateTableJump)
    return;

  assert(TM.getTargetTriple().isOSBinFormatELF());

  auto *LAFI = MF->getInfo<LoongArchMachineFunctionInfo>();
  unsigned EntrySize = LAFI->getJumpInfoSize();
  auto JTI = MF->getJumpTableInfo();

  if (!JTI || 0 == EntrySize)
    return;

  unsigned Size = getDataLayout().getPointerSize();
  auto JT = JTI->getJumpTables();

  // Emit an additional section to store the correlation info as pairs of
  // addresses, each pair contains the address of a jump instruction (jr) and
  // the address of the jump table.
  OutStreamer->switchSection(MMI->getContext().getELFSection(
      ".discard.tablejump_annotate", ELF::SHT_PROGBITS, 0));

  for (unsigned Idx = 0; Idx < EntrySize; ++Idx) {
    int JTIIdx = LAFI->getJumpInfoJTIIndex(Idx);
    if (JT[JTIIdx].MBBs.empty())
      continue;
    OutStreamer->emitValue(
        MCSymbolRefExpr::create(LAFI->getJumpInfoJrMI(Idx)->getPreInstrSymbol(),
                                OutContext),
        Size);
    OutStreamer->emitValue(
        MCSymbolRefExpr::create(GetJTISymbol(JTIIdx), OutContext), Size);
  }
}

bool LoongArchAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  AsmPrinter::runOnMachineFunction(MF);
  // Emit the XRay table for this function.
  emitXRayTable();
  return true;
}

char LoongArchAsmPrinter::ID = 0;

INITIALIZE_PASS(LoongArchAsmPrinter, "loongarch-asm-printer",
                "LoongArch Assembly Printer", false, false)

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeLoongArchAsmPrinter() {
  RegisterAsmPrinter<LoongArchAsmPrinter> X(getTheLoongArch32Target());
  RegisterAsmPrinter<LoongArchAsmPrinter> Y(getTheLoongArch64Target());
}
