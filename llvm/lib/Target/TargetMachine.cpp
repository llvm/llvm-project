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
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
using namespace llvm;

//---------------------------------------------------------------------------
// TargetMachine Class
//

TargetMachine::TargetMachine(const Target &T, StringRef DataLayoutString,
                             const Triple &TT, StringRef CPU, StringRef FS,
                             const TargetOptions &Options)
    : TheTarget(T), DL(DataLayoutString), TargetTriple(TT),
      TargetCPU(std::string(CPU)), TargetFS(std::string(FS)), AsmInfo(nullptr),
      MRI(nullptr), MII(nullptr), STI(nullptr), RequireStructuredCFG(false),
      O0WantsFastISel(false), DefaultOptions(Options), Options(Options) {}

TargetMachine::~TargetMachine() = default;

bool TargetMachine::isPositionIndependent() const {
  return getRelocationModel() == Reloc::PIC_;
}

/// Reset the target options based on the function's attributes.
/// setFunctionAttributes should have made the raw attribute value consistent
/// with the command line flag if used.
//
// FIXME: This function needs to go away for a number of reasons:
// a) global state on the TargetMachine is terrible in general,
// b) these target options should be passed only on the function
//    and not on the TargetMachine (via TargetOptions) at all.
void TargetMachine::resetTargetOptions(const Function &F) const {
#define RESET_OPTION(X, Y)                                              \
  do {                                                                  \
    Options.X = F.getFnAttribute(Y).getValueAsBool();     \
  } while (0)

  RESET_OPTION(UnsafeFPMath, "unsafe-fp-math");
  RESET_OPTION(NoInfsFPMath, "no-infs-fp-math");
  RESET_OPTION(NoNaNsFPMath, "no-nans-fp-math");
  RESET_OPTION(NoSignedZerosFPMath, "no-signed-zeros-fp-math");
  RESET_OPTION(ApproxFuncFPMath, "approx-func-fp-math");
}

/// Returns the code generation relocation model. The choices are static, PIC,
/// and dynamic-no-pic.
Reloc::Model TargetMachine::getRelocationModel() const { return RM; }

/// Returns the code model. The choices are small, kernel, medium, large, and
/// target default.
CodeModel::Model TargetMachine::getCodeModel() const { return CMModel; }

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

bool TargetMachine::shouldAssumeDSOLocal(const Module &M,
                                         const GlobalValue *GV) const {
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
    if (TT.isWindowsGNUEnvironment() && GV->isDeclarationForLinker() &&
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

bool TargetMachine::useEmulatedTLS() const {
  // Returns Options.EmulatedTLS if the -emulated-tls or -no-emulated-tls
  // was specified explicitly; otherwise uses target triple to decide default.
  if (Options.ExplicitEmulatedTLS)
    return Options.EmulatedTLS;
  return getTargetTriple().hasDefaultEmulatedTLS();
}

TLSModel::Model TargetMachine::getTLSModel(const GlobalValue *GV) const {
  bool IsPIE = GV->getParent()->getPIELevel() != PIELevel::Default;
  Reloc::Model RM = getRelocationModel();
  bool IsSharedLibrary = RM == Reloc::PIC_ && !IsPIE;
  bool IsLocal = shouldAssumeDSOLocal(*GV->getParent(), GV);

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

/// Returns the optimization level: None, Less, Default, or Aggressive.
CodeGenOpt::Level TargetMachine::getOptLevel() const { return OptLevel; }

void TargetMachine::setOptLevel(CodeGenOpt::Level Level) { OptLevel = Level; }

TargetTransformInfo TargetMachine::getTargetTransformInfo(const Function &F) {
  return TargetTransformInfo(F.getParent()->getDataLayout());
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

TargetIRAnalysis TargetMachine::getTargetIRAnalysis() {
  // Since Analysis can't depend on Target, use a std::function to invert the
  // dependency.
  return TargetIRAnalysis(
      [this](const Function &F) { return this->getTargetTransformInfo(F); });
}

std::pair<int, int> TargetMachine::parseBinutilsVersion(StringRef Version) {
  if (Version == "none")
    return {INT_MAX, INT_MAX}; // Make binutilsIsAtLeast() return true.
  std::pair<int, int> Ret;
  if (!Version.consumeInteger(10, Ret.first) && Version.consume_front("."))
    Version.consumeInteger(10, Ret.second);
  return Ret;
}
