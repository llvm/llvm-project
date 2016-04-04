//===-- SystemZAsmPrinter.cpp - SystemZ LLVM assembly printer -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Streams SystemZ assembly language and associated data, in the form of
// MCInsts and MCExprs respectively.
//
//===----------------------------------------------------------------------===//

#include "SystemZAsmPrinter.h"
#include "InstPrinter/SystemZInstPrinter.h"
#include "SystemZConstantPoolValue.h"
#include "SystemZMCInstLower.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

// Return an RI instruction like MI with opcode Opcode, but with the
// GR64 register operands turned into GR32s.
static MCInst lowerRILow(const MachineInstr *MI, unsigned Opcode) {
  if (MI->isCompare())
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(1).getImm());
  else
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(1).getReg()))
      .addImm(MI->getOperand(2).getImm());
}

// Return an RI instruction like MI with opcode Opcode, but with the
// GR64 register operands turned into GRH32s.
static MCInst lowerRIHigh(const MachineInstr *MI, unsigned Opcode) {
  if (MI->isCompare())
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(1).getImm());
  else
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(1).getReg()))
      .addImm(MI->getOperand(2).getImm());
}

// Return an RI instruction like MI with opcode Opcode, but with the
// R2 register turned into a GR64.
static MCInst lowerRIEfLow(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
    .addReg(MI->getOperand(0).getReg())
    .addReg(MI->getOperand(1).getReg())
    .addReg(SystemZMC::getRegAsGR64(MI->getOperand(2).getReg()))
    .addImm(MI->getOperand(3).getImm())
    .addImm(MI->getOperand(4).getImm())
    .addImm(MI->getOperand(5).getImm());
}

static const MCSymbolRefExpr *getTLSGetOffset(MCContext &Context) {
  StringRef Name = "__tls_get_offset";
  return MCSymbolRefExpr::create(Context.getOrCreateSymbol(Name),
                                 MCSymbolRefExpr::VK_PLT,
                                 Context);
}

static const MCSymbolRefExpr *getGlobalOffsetTable(MCContext &Context) {
  StringRef Name = "_GLOBAL_OFFSET_TABLE_";
  return MCSymbolRefExpr::create(Context.getOrCreateSymbol(Name),
                                 MCSymbolRefExpr::VK_None,
                                 Context);
}

// MI loads the high part of a vector from memory.  Return an instruction
// that uses replicating vector load Opcode to do the same thing.
static MCInst lowerSubvectorLoad(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
    .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
    .addReg(MI->getOperand(1).getReg())
    .addImm(MI->getOperand(2).getImm())
    .addReg(MI->getOperand(3).getReg());
}

// MI stores the high part of a vector to memory.  Return an instruction
// that uses elemental vector store Opcode to do the same thing.
static MCInst lowerSubvectorStore(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
    .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
    .addReg(MI->getOperand(1).getReg())
    .addImm(MI->getOperand(2).getImm())
    .addReg(MI->getOperand(3).getReg())
    .addImm(0);
}

void SystemZAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  SystemZMCInstLower Lower(MF->getContext(), *this);
  MCInst LoweredMI;
  switch (MI->getOpcode()) {
  case SystemZ::Return:
    LoweredMI = MCInstBuilder(SystemZ::BR).addReg(SystemZ::R14D);
    break;

  case SystemZ::CallBRASL:
    LoweredMI = MCInstBuilder(SystemZ::BRASL)
      .addReg(SystemZ::R14D)
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_PLT));
    break;

  case SystemZ::CallBASR:
    LoweredMI = MCInstBuilder(SystemZ::BASR)
      .addReg(SystemZ::R14D)
      .addReg(MI->getOperand(0).getReg());
    break;

  case SystemZ::CallJG:
    LoweredMI = MCInstBuilder(SystemZ::JG)
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_PLT));
    break;

  case SystemZ::CallBR:
    LoweredMI = MCInstBuilder(SystemZ::BR).addReg(SystemZ::R1D);
    break;

  case SystemZ::TLS_GDCALL:
    LoweredMI = MCInstBuilder(SystemZ::BRASL)
      .addReg(SystemZ::R14D)
      .addExpr(getTLSGetOffset(MF->getContext()))
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_TLSGD));
    break;

  case SystemZ::TLS_LDCALL:
    LoweredMI = MCInstBuilder(SystemZ::BRASL)
      .addReg(SystemZ::R14D)
      .addExpr(getTLSGetOffset(MF->getContext()))
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_TLSLDM));
    break;

  case SystemZ::GOT:
    LoweredMI = MCInstBuilder(SystemZ::LARL)
      .addReg(MI->getOperand(0).getReg())
      .addExpr(getGlobalOffsetTable(MF->getContext()));
    break;

  case SystemZ::IILF64:
    LoweredMI = MCInstBuilder(SystemZ::IILF)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(2).getImm());
    break;

  case SystemZ::IIHF64:
    LoweredMI = MCInstBuilder(SystemZ::IIHF)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(2).getImm());
    break;

  case SystemZ::RISBHH:
  case SystemZ::RISBHL:
    LoweredMI = lowerRIEfLow(MI, SystemZ::RISBHG);
    break;

  case SystemZ::RISBLH:
  case SystemZ::RISBLL:
    LoweredMI = lowerRIEfLow(MI, SystemZ::RISBLG);
    break;

  case SystemZ::VLVGP32:
    LoweredMI = MCInstBuilder(SystemZ::VLVGP)
      .addReg(MI->getOperand(0).getReg())
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(1).getReg()))
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(2).getReg()));
    break;

  case SystemZ::VLR32:
  case SystemZ::VLR64:
    LoweredMI = MCInstBuilder(SystemZ::VLR)
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(1).getReg()));
    break;

  case SystemZ::VL32:
    LoweredMI = lowerSubvectorLoad(MI, SystemZ::VLREPF);
    break;

  case SystemZ::VL64:
    LoweredMI = lowerSubvectorLoad(MI, SystemZ::VLREPG);
    break;

  case SystemZ::VST32:
    LoweredMI = lowerSubvectorStore(MI, SystemZ::VSTEF);
    break;

  case SystemZ::VST64:
    LoweredMI = lowerSubvectorStore(MI, SystemZ::VSTEG);
    break;

  case SystemZ::LFER:
    LoweredMI = MCInstBuilder(SystemZ::VLGVF)
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(1).getReg()))
      .addReg(0).addImm(0);
    break;

  case SystemZ::LEFR:
    LoweredMI = MCInstBuilder(SystemZ::VLVGF)
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(MI->getOperand(1).getReg())
      .addReg(0).addImm(0);
    break;

#define LOWER_LOW(NAME)                                                 \
  case SystemZ::NAME##64: LoweredMI = lowerRILow(MI, SystemZ::NAME); break

  LOWER_LOW(IILL);
  LOWER_LOW(IILH);
  LOWER_LOW(TMLL);
  LOWER_LOW(TMLH);
  LOWER_LOW(NILL);
  LOWER_LOW(NILH);
  LOWER_LOW(NILF);
  LOWER_LOW(OILL);
  LOWER_LOW(OILH);
  LOWER_LOW(OILF);
  LOWER_LOW(XILF);

#undef LOWER_LOW

#define LOWER_HIGH(NAME) \
  case SystemZ::NAME##64: LoweredMI = lowerRIHigh(MI, SystemZ::NAME); break

  LOWER_HIGH(IIHL);
  LOWER_HIGH(IIHH);
  LOWER_HIGH(TMHL);
  LOWER_HIGH(TMHH);
  LOWER_HIGH(NIHL);
  LOWER_HIGH(NIHH);
  LOWER_HIGH(NIHF);
  LOWER_HIGH(OIHL);
  LOWER_HIGH(OIHH);
  LOWER_HIGH(OIHF);
  LOWER_HIGH(XIHF);

#undef LOWER_HIGH

  case SystemZ::Serialize:
    if (MF->getSubtarget<SystemZSubtarget>().hasFastSerialization())
      LoweredMI = MCInstBuilder(SystemZ::AsmBCR)
        .addImm(14).addReg(SystemZ::R0D);
    else
      LoweredMI = MCInstBuilder(SystemZ::AsmBCR)
        .addImm(15).addReg(SystemZ::R0D);
    break;

  // Emit nothing here but a comment if we can.
  case SystemZ::MemBarrier:
    OutStreamer->emitRawComment("MEMBARRIER");
    return;

  default:
    Lower.lower(MI, LoweredMI);
    break;
  }
  EmitToStreamer(*OutStreamer, LoweredMI);
}

// Convert a SystemZ-specific constant pool modifier into the associated
// MCSymbolRefExpr variant kind.
static MCSymbolRefExpr::VariantKind
getModifierVariantKind(SystemZCP::SystemZCPModifier Modifier) {
  switch (Modifier) {
  case SystemZCP::TLSGD: return MCSymbolRefExpr::VK_TLSGD;
  case SystemZCP::TLSLDM: return MCSymbolRefExpr::VK_TLSLDM;
  case SystemZCP::DTPOFF: return MCSymbolRefExpr::VK_DTPOFF;
  case SystemZCP::NTPOFF: return MCSymbolRefExpr::VK_NTPOFF;
  }
  llvm_unreachable("Invalid SystemCPModifier!");
}

void SystemZAsmPrinter::
EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  auto *ZCPV = static_cast<SystemZConstantPoolValue*>(MCPV);

  const MCExpr *Expr =
    MCSymbolRefExpr::create(getSymbol(ZCPV->getGlobalValue()),
                            getModifierVariantKind(ZCPV->getModifier()),
                            OutContext);
  uint64_t Size = getDataLayout().getTypeAllocSize(ZCPV->getType());

  OutStreamer->EmitValue(Expr, Size);
}

bool SystemZAsmPrinter::PrintAsmOperand(const MachineInstr *MI,
                                        unsigned OpNo,
                                        unsigned AsmVariant,
                                        const char *ExtraCode,
                                        raw_ostream &OS) {
  if (ExtraCode && *ExtraCode == 'n') {
    if (!MI->getOperand(OpNo).isImm())
      return true;
    OS << -int64_t(MI->getOperand(OpNo).getImm());
  } else {
    SystemZMCInstLower Lower(MF->getContext(), *this);
    MCOperand MO(Lower.lowerOperand(MI->getOperand(OpNo)));
    SystemZInstPrinter::printOperand(MO, MAI, OS);
  }
  return false;
}

bool SystemZAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                              unsigned OpNo,
                                              unsigned AsmVariant,
                                              const char *ExtraCode,
                                              raw_ostream &OS) {
  SystemZInstPrinter::printAddress(MI->getOperand(OpNo).getReg(),
                                   MI->getOperand(OpNo + 1).getImm(),
                                   MI->getOperand(OpNo + 2).getReg(), OS);
  return false;
}

// Force static initialization.
extern "C" void LLVMInitializeSystemZAsmPrinter() {
  RegisterAsmPrinter<SystemZAsmPrinter> X(TheSystemZTarget);
}
