//===-- LX32MCTargetDesc.cpp - LX32 MC Target Description ----------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines MC-layer registration and factories for LX32.
// It is organized into the following sections:
//
//   Section 0 — Generated descriptor imports
//   Section 1 — Minimal MCInstPrinter implementation
//   Section 2 — Factory functions (MCAsmInfo/MCInstrInfo/MCSubtargetInfo/...)
//   Section 3 — Target registration entry point
//
//===----------------------------------------------------------------------===//

#include "LX32MCTargetDesc.h"
#include "LX32MCAsmInfo.h"
#include "../TargetInfo/LX32TargetInfo.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDwarf.h" // for MCCFIInstruction
#include "llvm/MC/MCExpr.h"  // for MCExpr::print
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

// Pull in TableGen-generated enums and init helpers.
//
// The generated MC descriptor tables reference symbols like LX32::X1 and
// LX32::GPRRegClassID. Those are emitted by the *_ENUM sections.
// If we skip them, the compiler will fail with:
//   error: use of undeclared identifier 'LX32'

#define GET_REGINFO_ENUM
#include "../TableGen/LX32GenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "../TableGen/LX32GenInstrInfo.inc"

// Now pull in the generated Init* functions and createLX32MCSubtargetInfoImpl.
#define GET_INSTRINFO_MC_DESC
#include "../TableGen/LX32GenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "../TableGen/LX32GenRegisterInfo.inc"

// Emits: createLX32MCSubtargetInfoImpl(), LX32WriteProcResTable, etc.
#define GET_SUBTARGETINFO_MC_DESC
#include "../TableGen/LX32GenSubtargetInfo.inc"

using namespace llvm;

//===----------------------------------------------------------------------===//
// MCInstPrinter — minimal implementation
//===----------------------------------------------------------------------===//
//
// LLVM requires a registered MCInstPrinter before it can create an
// AsmStreamer. The printer is responsible for converting an MCInst to human-
// readable text. A skeleton implementation that prints register/immediate
// operands is sufficient to pass the assertion and produce readable (if
// incomplete) assembly output.

namespace {

class LX32InstPrinter : public MCInstPrinter {
public:
  LX32InstPrinter(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                  const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI) {}

  // Print a full instruction. For now we fall back to the raw opcode number
  // and operand list so that the pipeline produces _something_ instead of
  // crashing. This will be replaced by the TableGen-generated AsmWriter once
  // LX32AsmWriter.inc is available.
  void printInst(const MCInst *MI, uint64_t Address, StringRef Annot,
                 const MCSubtargetInfo &STI, raw_ostream &OS) override {
    // Try to get the instruction name from the generated table.
    StringRef Name = MII.getName(MI->getOpcode());
    if (!Name.empty())
      OS << "\t" << Name;
    else
      OS << "\t<opcode:" << MI->getOpcode() << ">";

    // Print operands separated by ", ".
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      OS << (i == 0 ? "\t" : ", ");
      const MCOperand &Op = MI->getOperand(i);
      if (Op.isReg())
        // Use the ABI register name (alt name index 0 = ABIRegAltName).
        OS << getRegisterName(Op.getReg());
      else if (Op.isImm())
        OS << Op.getImm();
      else if (Op.isExpr())
        // Some LLVM versions keep MCExpr::print() private and do not provide an
        // operator<< overload. Until we hook up a real asm writer, print a
        // stable placeholder instead of failing to compile.
        OS << "<expr>";
      else
        OS << "<unknown operand>";
    }

    // Emit any inline annotation (e.g. branch target comment).
    printAnnotation(OS, Annot);
  }

  // Required: return the register name. We use ABIRegAltName (index 0)
  // so that x10 prints as "a0", x1 as "ra", etc.
  static const char *getRegisterName(MCRegister Reg) {
    // IMPORTANT:
    //   Returning nullptr here is UB for callers and has been observed to
    //   trigger asserts/crashes when the MC layer tries to print instructions.
    //
    // Until we hook up the TableGen-generated AsmWriter (which provides proper
    // ABI names like "sp", "ra", "a0", ...), we provide a tiny but safe
    // fallback that prints registers as "x<N>".
    // TableGen allocates registers starting from ID 1, so we subtract 1.
    static thread_local char Buf[16];
    unsigned R = static_cast<unsigned>(Reg.id());
    if (R > 0) R -= 1;
    (void)snprintf(Buf, sizeof(Buf), "x%u", R);
    return Buf;
  }

  // Required pure virtual from MCInstPrinter — return the instruction mnemonic.
  std::pair<const char *, uint64_t>
  getMnemonic(const MCInst &MI) const override {
    StringRef Name = MII.getName(MI.getOpcode());
    return {Name.data(), 0};
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Factory functions
//===----------------------------------------------------------------------===//

static MCInstrInfo *createLX32MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitLX32MCInstrInfo(X); // generated by GET_INSTRINFO_MC_DESC
  return X;
}

static MCRegisterInfo *createLX32MCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  // X1 = ra (return address register — used for DWARF CFA).
  InitLX32MCRegisterInfo(X, LX32::X1); // generated by GET_REGINFO_MC_DESC
  return X;
}

static MCAsmInfo *createLX32MCAsmInfo(const MCRegisterInfo &MRI,
                                       const Triple &TT,
                                       const MCTargetOptions &Options) {
  if (!TT.isOSBinFormatELF())
    llvm::report_fatal_error("lx32: only ELF object format is supported");

  MCAsmInfo *MAI = new LX32MCAsmInfo(TT);

  // Set the initial frame state so DWARF unwinding works.
  // SP (x2) is register number 2 in the DWARF mapping.
  unsigned SP = MRI.getDwarfRegNum(LX32::X2, /*isEH=*/true);
  MCCFIInstruction Inst = MCCFIInstruction::cfiDefCfa(nullptr, SP, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCSubtargetInfo *createLX32MCSubtargetInfo(const Triple &TT,
                                                   StringRef CPU,
                                                   StringRef FS) {
  // Use the TableGen-generated factory. This ensures the processor table
  // (LX32SubTypeKV) is populated and "generic" is recognised.
  // The previous manual construction with empty arrays was the root cause of:
  //   'generic' is not a recognized processor for this target (ignoring processor)
  if (CPU.empty())
    CPU = "generic";
  return createLX32MCSubtargetInfoImpl(TT, CPU, /*TuneCPU=*/CPU, FS);
}

static MCInstPrinter *createLX32MCInstPrinter(const Triple &TT,
                                               unsigned SyntaxVariant,
                                               const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI) {
  // Only one syntax variant is defined for lx32.
  if (SyntaxVariant != 0)
    return nullptr;
  return new LX32InstPrinter(MAI, MII, MRI);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeLX32TargetMC() {
  Target &T = getTheLX32TargetInfo();

  TargetRegistry::RegisterMCAsmInfo(T,        createLX32MCAsmInfo);
  TargetRegistry::RegisterMCInstrInfo(T,      createLX32MCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(T,        createLX32MCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(T,  createLX32MCSubtargetInfo);
  TargetRegistry::RegisterMCInstPrinter(T,    createLX32MCInstPrinter);

  TargetRegistry::RegisterMCCodeEmitter(T,    createLX32MCCodeEmitter);
  TargetRegistry::RegisterMCAsmBackend(T,     createLX32AsmBackend);
}