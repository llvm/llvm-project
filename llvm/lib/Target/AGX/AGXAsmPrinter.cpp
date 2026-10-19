//===-- AGXAsmPrinter.cpp - AGX assembly printer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits AGX machine code. Each kernel is preceded by comments describing the
// resources the driver must allocate for it (uniform registers, general
// purpose registers, threadgroup memory and stack), and followed by trap
// padding, since the instruction prefetcher reads past the end of the
// program. The initial contents of the global data buffer are emitted in the
// .agx.globals section.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXFrameLowering.h"
#include "AGXMCInstLower.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "TargetInfo/AGXTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

#define DEBUG_TYPE "agx-asm-printer"

namespace {

class AGXAsmPrinter : public AsmPrinter {
  bool ReportedMissingEncoding = false;

public:
  static char ID;

  explicit AGXAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID) {}

  StringRef getPassName() const override { return "AGX Assembly Printer"; }

  bool runOnMachineFunction(MachineFunction &MF) override {
    ReportedMissingEncoding = false;
    return AsmPrinter::runOnMachineFunction(MF);
  }

  void emitGlobalVariable(const GlobalVariable *GV) override;
  void emitEndOfAsmFile(Module &M) override;
  void emitFunctionBodyStart() override;
  void emitFunctionBodyEnd() override;
  void emitInstruction(const MachineInstr *MI) override;
};

} // namespace

char AGXAsmPrinter::ID = 0;

INITIALIZE_PASS(AGXAsmPrinter, "agx-asm-printer", "AGX Assembly Printer",
                false, false)

/// The number of 32-bit general purpose registers used, counting from r0.
static unsigned getGPRCount(const MachineFunction &MF) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned Count = 1; // r0l is always used for the execution mask.
  for (unsigned I = 0; I < 128; ++I)
    if (MRI.isPhysRegUsed(AGX::GPR32RegClass.getRegister(I)))
      Count = I + 1;
  return Count;
}

void AGXAsmPrinter::emitFunctionBodyStart() {
  if (!isVerbose())
    return;
  const auto *MFI = MF->getInfo<AGXMachineFunctionInfo>();
  OutStreamer->emitRawComment(" uniform registers: " +
                              Twine(MFI->getUniformHalfCount()) +
                              " x 16-bit (threadgroups_per_grid at u" +
                              Twine(MFI->getImplicitArgHalfIndex() / 2) + ")");
  OutStreamer->emitRawComment(" general purpose registers: " +
                              Twine(getGPRCount(*MF)) + " x 32-bit");
  if (MFI->getGlobalDataVReg())
    OutStreamer->emitRawComment(
        " global data buffer at u" + Twine(MFI->getGlobalDataHalfIndex() / 2) +
        "_u" + Twine(MFI->getGlobalDataHalfIndex() / 2 + 1));
  OutStreamer->emitRawComment(" threadgroup memory: " +
                              Twine(MFI->getThreadgroupSize()) + " bytes");
  OutStreamer->emitRawComment(" stack size: " +
                              Twine(AGXFrameLowering::getStackAdjustUnits(*MF) *
                                    4) +
                              " bytes");
}

void AGXAsmPrinter::emitGlobalVariable(const GlobalVariable *GV) {
  // Threadgroup variables are allocated per kernel, and device variables are
  // emitted together by emitEndOfAsmFile.
  if (GV->getAddressSpace() == AGX::THREADGROUP_ADDRESS ||
      AGX::isGlobalDataVariable(*GV))
    return;
  AsmPrinter::emitGlobalVariable(GV);
}

void AGXAsmPrinter::emitEndOfAsmFile(Module &M) {
  AGX::GlobalDataLayout Layout = AGX::computeGlobalDataLayout(M);
  if (Layout.empty())
    return;
  const DataLayout &DL = M.getDataLayout();
  OutStreamer->switchSection(OutContext.getELFSection(
      ".agx.globals", ELF::SHT_PROGBITS, ELF::SHF_ALLOC | ELF::SHF_WRITE));
  OutStreamer->emitValueToAlignment(Layout.MaxAlign);
  uint64_t End = 0;
  for (const auto &[GV, Offset] : Layout.Offsets) {
    OutStreamer->emitZeros(Offset - End);
    // Typed, sized symbols with the variable's linkage, so that a runtime
    // can find a variable's offset in the buffer by name.
    MCSymbol *Sym = getSymbol(GV);
    uint64_t Size = DL.getTypeAllocSize(GV->getValueType());
    emitLinkage(GV, Sym);
    emitVisibility(Sym, GV->getVisibility(), !GV->isDeclaration());
    OutStreamer->emitSymbolAttribute(Sym, MCSA_ELF_TypeObject);
    OutStreamer->emitELFSize(Sym, MCConstantExpr::create(Size, OutContext));
    OutStreamer->emitLabel(Sym);
    const Constant *Init = GV->hasInitializer() ? GV->getInitializer() : nullptr;
    if (Init && !isa<UndefValue>(Init))
      emitGlobalConstant(DL, Init);
    else
      OutStreamer->emitZeros(Size);
    End = Offset + Size;
  }
}

void AGXAsmPrinter::emitFunctionBodyEnd() {
  const auto &ST = MF->getSubtarget<AGXSubtarget>();
  if (ST.getEncodingFamily() != AGXSubtarget::AGX2)
    return;
  for (unsigned I = 0; I < 8; ++I) {
    MCInst Trap;
    Trap.setOpcode(AGX::TRAP_g13);
    EmitToStreamer(*OutStreamer, Trap);
  }
}

void AGXAsmPrinter::emitInstruction(const MachineInstr *MI) {
  const auto &ST = MF->getSubtarget<AGXSubtarget>();
  AGXMCInstLower Lower(OutContext, ST);
  MCInst Inst;
  if (!Lower.lower(MI, Inst)) {
    if (!ReportedMissingEncoding) {
      OutContext.reportError(
          SMLoc(), "no AGX instruction encoding for '" +
                       Twine(ST.getInstrInfo()->getName(MI->getOpcode())) +
                       "' on " + ST.getCPU() +
                       (ST.getEncodingFamily() == AGXSubtarget::Apple9
                            ? " (the Apple9 encoding is not yet supported)"
                            : ""));
      ReportedMissingEncoding = true;
    }
    return;
  }
  EmitToStreamer(*OutStreamer, Inst);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeAGXAsmPrinter() {
  RegisterAsmPrinter<AGXAsmPrinter> X(getTheAGXTarget());
}
