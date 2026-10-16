//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
using namespace llvm;

cl::opt<bool> llvm::NoKernelInfoEndLTO(
    "no-kernel-info-end-lto",
    cl::desc("remove the kernel-info pass at the end of the full LTO pipeline"),
    cl::init(false), cl::Hidden);

//---------------------------------------------------------------------------
// TargetMachine Class
//

TargetMachine::TargetMachine(const Target &T, StringRef DataLayoutString,
                             const Triple &TT, StringRef CPU, StringRef FS,
                             const TargetOptions &Options)
    : TheTarget(T), DL(DataLayoutString), TargetTriple(TT),
      TargetCPU(std::string(CPU)), TargetFS(std::string(FS)), AsmInfo(nullptr),
      MRI(nullptr), MII(nullptr), STI(nullptr), RequireStructuredCFG(false),
      O0WantsFastISel(false), EnableTiedFastRegAlloc(true),
      SupportsDefaultOutlining(false), SupportsDebugEntryValues(false),
      Options(Options) {}

TargetMachine::~TargetMachine() = default;

/// NOTE: There are targets that still do not support the debug entry values
/// production and that is being controlled with the SupportsDebugEntryValues.
/// In addition, SCE debugger does not have the feature implemented, so prefer
/// not to emit the debug entry values in that case.
/// The EnableDebugEntryValues can be used for the testing purposes.
bool TargetMachine::shouldEmitDebugEntryValues() const {
  return (SupportsDebugEntryValues &&
          Options.DebuggerTuning != DebuggerKind::SCE) ||
         Options.EnableDebugEntryValues;
}

Expected<std::unique_ptr<MCStreamer>>
TargetMachine::createMCStreamer(raw_pwrite_stream &Out,
                                raw_pwrite_stream *DwoOut,
                                CodeGenFileType FileType, MCContext &Ctx) {
  return nullptr;
}

bool TargetMachine::isLargeDataSize(uint64_t Size) const {
  if (getTargetTriple().getArch() != Triple::x86_64)
    return false;

  if (!getTargetTriple().isOSBinFormatELF())
    return getCodeModel() == CodeModel::Large;

  if (getCodeModel() == CodeModel::Medium || getCodeModel() == CodeModel::Large)
    return Size == 0 || Size > LargeDataThreshold;

  return false;
}

bool TargetMachine::isPositionIndependent() const {
  return getRelocationModel() == Reloc::PIC_;
}

/// Returns the code generation relocation model. The choices are static, PIC,
/// and dynamic-no-pic.
Reloc::Model TargetMachine::getRelocationModel() const { return RM; }

uint64_t TargetMachine::getMaxCodeSize() const {
  switch (getCodeModel()) {
  case CodeModel::Tiny:
    return llvm::maxUIntN(10);
  case CodeModel::Small:
  case CodeModel::Kernel:
  case CodeModel::Medium:
    return llvm::maxUIntN(31);
  case CodeModel::Large:
    return llvm::maxUIntN(64);
  }
  llvm_unreachable("Unhandled CodeModel enum");
}

/// Get the IR-specified TLS model for Var.
static TLSModel::Model getSelectedTLSModel(const GlobalValue *GV) {
  switch (GV->getThreadLocalMode()) {
  case GlobalVariable::NotThreadLocal:
    llvm_unreachable("getSelectedTLSModel for non-TLS variable");
    break;
  case GlobalVariable::GeneralDynamicTLSModel:
    return TLSModel::GeneralDynamic;
  case GlobalVariable::LocalDynamicTLSModel:
    return TLSModel::LocalDynamic;
  case GlobalVariable::InitialExecTLSModel:
    return TLSModel::InitialExec;
  case GlobalVariable::LocalExecTLSModel:
    return TLSModel::LocalExec;
  }
  llvm_unreachable("invalid TLS model");
}

bool TargetMachine::shouldAssumeDSOLocal(const GlobalValue *GV) const {
  const Triple &TT = getTargetTriple();
  Reloc::Model RM = getRelocationModel();

  // According to the llvm language reference, we should be able to
  // just return false in here if we have a GV, as we know it is
  // dso_preemptable.  At this point in time, the various IR producers
  // have not been transitioned to always produce a dso_local when it
  // is possible to do so.
  //
  // As a result we still have some logic in here to improve the quality of the
  // generated code.
  if (!GV)
    return false;

  // If the IR producer requested that this GV be treated as dso local, obey.
  if (GV->isDSOLocal())
    return true;

  if (TT.isOSBinFormatCOFF()) {
    // DLLImport explicitly marks the GV as external.
    if (GV->hasDLLImportStorageClass())
      return false;

    // On MinGW, variables that haven't been declared with DLLImport may still
    // end up automatically imported by the linker. To make this feasible,
    // don't assume the variables to be DSO local unless we actually know
    // that for sure. This only has to be done for variables; for functions
    // the linker can insert thunks for calling functions from another DLL.
    if (TT.isOSCygMing() && GV->isDeclarationForLinker() &&
        isa<GlobalVariable>(GV))
      return false;

    // Don't mark 'extern_weak' symbols as DSO local. If these symbols remain
    // unresolved in the link, they can be resolved to zero, which is outside
    // the current DSO.
    if (GV->hasExternalWeakLinkage())
      return false;

    // Every other GV is local on COFF.
    return true;
  }

  if (TT.isOSBinFormatGOFF())
    return true;

  if (TT.isOSBinFormatMachO()) {
    if (RM == Reloc::Static)
      return true;
    return GV->isStrongDefinitionForLinker();
  }

  assert(TT.isOSBinFormatELF() || TT.isOSBinFormatWasm() ||
         TT.isOSBinFormatXCOFF());
  return false;
}

bool TargetMachine::useEmulatedTLS() const { return Options.EmulatedTLS; }
bool TargetMachine::useTLSDESC() const { return Options.EnableTLSDESC; }

TLSModel::Model TargetMachine::getTLSModel(const GlobalValue *GV) const {
  bool IsPIE = GV->getParent()->getPIELevel() != PIELevel::Default;
  Reloc::Model RM = getRelocationModel();
  bool IsSharedLibrary = RM == Reloc::PIC_ && !IsPIE;
  bool IsLocal = shouldAssumeDSOLocal(GV);

  TLSModel::Model Model;
  if (IsSharedLibrary) {
    if (IsLocal)
      Model = TLSModel::LocalDynamic;
    else
      Model = TLSModel::GeneralDynamic;
  } else {
    if (IsLocal)
      Model = TLSModel::LocalExec;
    else
      Model = TLSModel::InitialExec;
  }

  // If the user specified a more specific model, use that.
  TLSModel::Model SelectedModel = getSelectedTLSModel(GV);
  if (SelectedModel > Model)
    return SelectedModel;

  return Model;
}

TargetTransformInfo
TargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(F.getDataLayout());
}

void TargetMachine::getNameWithPrefix(SmallVectorImpl<char> &Name,
                                      const GlobalValue *GV, Mangler &Mang,
                                      bool MayAlwaysUsePrivate) const {
  if (MayAlwaysUsePrivate || !GV->hasPrivateLinkage()) {
    // Simple case: If GV is not private, it is not important to find out if
    // private labels are legal in this case or not.
    Mang.getNameWithPrefix(Name, GV, false);
    return;
  }
  const TargetLoweringObjectFile *TLOF = getObjFileLowering();
  TLOF->getNameWithPrefix(Name, GV, *this);
}

MCSymbol *TargetMachine::getSymbol(const GlobalValue *GV) const {
  const TargetLoweringObjectFile *TLOF = getObjFileLowering();
  // XCOFF symbols could have special naming convention.
  if (MCSymbol *TargetSymbol = TLOF->getTargetSymbol(GV, *this))
    return TargetSymbol;

  SmallString<128> NameStr;
  getNameWithPrefix(NameStr, GV, TLOF->getMangler());
  return TLOF->getContext().getOrCreateSymbol(NameStr);
}

TargetIRAnalysis TargetMachine::getTargetIRAnalysis() const {
  // Since Analysis can't depend on Target, use a std::function to invert the
  // dependency.
  return TargetIRAnalysis(
      [this](const Function &F) { return this->getTargetTransformInfo(F); });
}

StringRef TargetMachine::getTargetABIName(const Module &M) const {
  if (const auto *MD = cast_or_null<MDString>(M.getModuleFlag("target-abi")))
    return MD->getString();
  return Options.MCOptions.getABIName();
}

void TargetMachine::verifyOptionsConsistency(const Module &M) const {
  // The "target-abi" module flag must agree with the -target-abi option.
  StringRef OptionABI = Options.MCOptions.getABIName();
  if (!OptionABI.empty()) {
    if (const auto *MD =
            cast_or_null<MDString>(M.getModuleFlag("target-abi"))) {
      if (OptionABI != MD->getString())
        M.getContext().emitError(
            "-target-abi option != target-abi module flag");
    }
  }
}

const MCSubtargetInfo &TargetMachine::getMCSubtargetInfo(StringRef CPU,
                                                         StringRef FS) {
  if (CPU.empty() && FS.empty())
    return *STI;
  SmallString<128> Key = CPU;
  Key += '/';
  Key += FS;
  auto &Entry = MCSubtargetMap[Key];
  if (!Entry)
    Entry.reset(getTarget().createMCSubtargetInfo(getTargetTriple(), CPU, FS));
  return *Entry;
}
