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
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

#define DEBUG_TYPE "agx-asm-printer"

namespace {

class AGXAsmPrinter : public AsmPrinter {
  bool ReportedMissingEncoding = false;
  /// Encodes instructions printed as text in mesa-isa mode, so that they are
  /// checked against Mesa's forms as object output is.
  std::unique_ptr<MCCodeEmitter> MesaChecker;
  /// The .agx.kernels record of each kernel, emitted at the end of the file.
  SmallVector<std::pair<std::string, SmallVector<uint32_t, 16>>, 4> Kernels;

  void recordKernel();

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
  unsigned GPRs = getGPRCount(*MF);
  OutStreamer->emitRawComment(
      " general purpose registers: " + Twine(GPRs) + " x 32-bit (up to " +
      Twine(AGXSubtarget::maxThreadsForHalves(2 * GPRs)) +
      " threads per threadgroup)");
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

/// The .agx.kernels record of the current kernel. Each record is a sequence
/// of little-endian 32-bit words:
///
///   size (of the record in bytes), version (1), name length, the name
///   (padded with zeros to 4 bytes), uniform halves, first half of
///   threadgroups_per_grid, first half of the global data buffer address
///   (0xffffffff if none), general purpose registers, threadgroup memory
///   bytes, stack bytes per thread, argument count, and for each argument
///   of the source kernel: kind (0 = its bytes are in the uniforms, 1 = a
///   64-bit pointer to its bytes is), its size in bytes, and its first half.
///   Optional tagged extensions follow, up to the record's size; readers skip
///   the ones they do not know:
///     1, x, y, z: the kernel runs clusters of x * y * z CTAs as one
///     threadgroup (function attribute agx-cluster-dims): the launch divides
///     the grid by (x, y, z) and multiplies the threadgroup's z by x * y * z.
///
/// Argument values are packed little-endian from their first half in 32-bit
/// words (16-bit arguments in one half).
void AGXAsmPrinter::recordKernel() {
  const Function &F = MF->getFunction();
  if (F.getCallingConv() != CallingConv::AGX_KERNEL)
    return;
  const auto *MFI = MF->getInfo<AGXMachineFunctionInfo>();
  const DataLayout &DL = F.getDataLayout();
  SmallVector<uint32_t, 16> R;
  R.push_back(MFI->getUniformHalfCount());
  R.push_back(MFI->getImplicitArgHalfIndex());
  R.push_back(MFI->getGlobalDataVReg() ? MFI->getGlobalDataHalfIndex()
                                       : ~0u);
  R.push_back(getGPRCount(*MF));
  R.push_back(MFI->getThreadgroupSize());
  R.push_back(AGXFrameLowering::getStackAdjustUnits(*MF) * 4);
  auto Half = [&](unsigned Arg) {
    return Arg < MFI->ArgHalves.size() ? MFI->ArgHalves[Arg] : ~0u;
  };
  SmallVector<uint32_t, 16> Args;
  Attribute Desc = F.getFnAttribute("agx-kernel-args");
  if (Desc.isValid()) {
    // Written by AGXPromoteKernelArgs: <v|p><bytes>:<first argument>.
    SmallVector<StringRef, 8> Items;
    Desc.getValueAsString().split(Items, ',');
    for (StringRef Item : Items) {
      auto [Size, First] = Item.drop_front().split(':');
      unsigned Bytes = 0, Arg = 0;
      Size.getAsInteger(10, Bytes);
      First.getAsInteger(10, Arg);
      Args.append({Item.front() == 'p' ? 1u : 0u, Bytes, Half(Arg)});
    }
  } else {
    for (const Argument &A : F.args()) {
      Type *T = A.getType()->isPointerTy() ? A.getParamByRefType() : nullptr;
      Args.append({T ? 1u : 0u,
                   uint32_t(T ? DL.getTypeAllocSize(T)
                              : DL.getTypeStoreSize(A.getType())),
                   Half(A.getArgNo())});
    }
  }
  R.push_back(Args.size() / 3);
  R.append(Args);
  Attribute Cluster = F.getFnAttribute("agx-cluster-dims");
  if (Cluster.isValid()) {
    SmallVector<StringRef, 3> Dims;
    Cluster.getValueAsString().split(Dims, ',');
    R.push_back(1);
    for (unsigned I = 0; I < 3; ++I) {
      unsigned D = 1;
      if (I < Dims.size() && Dims[I].trim().getAsInteger(10, D))
        D = 1;
      R.push_back(std::max(D, 1u));
    }
  }
  Kernels.push_back({std::string(getSymbol(&F)->getName()), std::move(R)});
}

void AGXAsmPrinter::emitEndOfAsmFile(Module &M) {
  if (!Kernels.empty()) {
    OutStreamer->switchSection(
        OutContext.getELFSection(".agx.kernels", ELF::SHT_PROGBITS, 0));
    OutStreamer->emitValueToAlignment(Align(4));
    for (const auto &[Name, R] : Kernels) {
      uint32_t NameWords = divideCeil(Name.size(), 4);
      OutStreamer->emitInt32(4 * (3 + NameWords + R.size()));
      OutStreamer->emitInt32(1);
      OutStreamer->emitInt32(Name.size());
      OutStreamer->emitBytes(Name);
      OutStreamer->emitZeros(4 * NameWords - Name.size());
      for (uint32_t W : R)
        OutStreamer->emitInt32(W);
    }
  }
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
  recordKernel();
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
  if (ST.hasMesaISA() && OutStreamer->hasRawTextSupport()) {
    if (!MesaChecker)
      MesaChecker.reset(
          TM.getTarget().createMCCodeEmitter(*TM.getMCInstrInfo(), OutContext));
    SmallVector<char, 16> Bytes;
    SmallVector<MCFixup, 2> Fixups;
    MesaChecker->encodeInstruction(Inst, Bytes, Fixups, ST);
  }
  EmitToStreamer(*OutStreamer, Inst);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeAGXAsmPrinter() {
  RegisterAsmPrinter<AGXAsmPrinter> X(getTheAGXTarget());
}
