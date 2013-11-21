//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

//---------------------------------------------------------------------------
// Command-line options that tend to be useful on more than one back-end.
//

namespace llvm {
  bool HasDivModLibcall;
  bool AsmVerbosityDefault(false);
}

static cl::opt<bool>
DataSections("fdata-sections",
  cl::desc("Emit data into separate sections"),
  cl::init(false));
static cl::opt<bool>
FunctionSections("ffunction-sections",
  cl::desc("Emit functions into separate sections"),
  cl::init(false));

//---------------------------------------------------------------------------
// TargetMachine Class
//

TargetMachine::TargetMachine(const Target &T,
                             StringRef TT, StringRef CPU, StringRef FS,
                             const TargetOptions &Options)
  : TheTarget(T), TargetTriple(TT), TargetCPU(CPU), TargetFS(FS),
    CodeGenInfo(0), AsmInfo(0),
    MCRelaxAll(false),
    MCNoExecStack(false),
    MCSaveTempLabels(false),
    MCUseLoc(true),
    MCUseCFI(true),
    MCUseDwarfDirectory(false),
    Options(Options) {
}

TargetMachine::~TargetMachine() {
  delete CodeGenInfo;
  delete AsmInfo;
}

/// \brief Reset the target options based on the function's attributes.
void TargetMachine::resetTargetOptions(const MachineFunction *MF) const {
  const Function *F = MF->getFunction();
  TargetOptions &TO = MF->getTarget().Options;
  
#define RESET_OPTION(X, Y)                                              \
  do {                                                                  \
    if (F->hasFnAttribute(Y))                                           \
      TO.X =                                                            \
        (F->getAttributes().                                            \
           getAttribute(AttributeSet::FunctionIndex,                    \
                        Y).getValueAsString() == "true");               \
  } while (0)

  RESET_OPTION(NoFramePointerElim, "no-frame-pointer-elim");
  RESET_OPTION(LessPreciseFPMADOption, "less-precise-fpmad");
  RESET_OPTION(UnsafeFPMath, "unsafe-fp-math");
  RESET_OPTION(NoInfsFPMath, "no-infs-fp-math");
  RESET_OPTION(NoNaNsFPMath, "no-nans-fp-math");
  RESET_OPTION(UseSoftFloat, "use-soft-float");
  RESET_OPTION(DisableTailCalls, "disable-tail-calls");
}

/// getRelocationModel - Returns the code generation relocation model. The
/// choices are static, PIC, and dynamic-no-pic, and target default.
Reloc::Model TargetMachine::getRelocationModel() const {
  if (!CodeGenInfo)
    return Reloc::Default;
  return CodeGenInfo->getRelocationModel();
}

/// getCodeModel - Returns the code model. The choices are small, kernel,
/// medium, large, and target default.
CodeModel::Model TargetMachine::getCodeModel() const {
  if (!CodeGenInfo)
    return CodeModel::Default;
  return CodeGenInfo->getCodeModel();
}

/// Get the IR-specified TLS model for Var.
static TLSModel::Model getSelectedTLSModel(const GlobalVariable *Var) {
  switch (Var->getThreadLocalMode()) {
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

TLSModel::Model TargetMachine::getTLSModel(const GlobalValue *GV) const {
  // If GV is an alias then use the aliasee for determining
  // thread-localness.
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
    GV = GA->resolveAliasedGlobal(false);
  const GlobalVariable *Var = cast<GlobalVariable>(GV);

  bool isLocal = Var->hasLocalLinkage();
  bool isDeclaration = Var->isDeclaration();
  bool isPIC = getRelocationModel() == Reloc::PIC_;
  bool isPIE = Options.PositionIndependentExecutable;
  // FIXME: what should we do for protected and internal visibility?
  // For variables, is internal different from hidden?
  bool isHidden = Var->hasHiddenVisibility();

  TLSModel::Model Model;
  if (isPIC && !isPIE) {
    if (isLocal || isHidden)
      Model = TLSModel::LocalDynamic;
    else
      Model = TLSModel::GeneralDynamic;
  } else {
    if (!isDeclaration || isHidden)
      Model = TLSModel::LocalExec;
    else
      Model = TLSModel::InitialExec;
  }

  // If the user specified a more specific model, use that.
  TLSModel::Model SelectedModel = getSelectedTLSModel(Var);
  if (SelectedModel > Model)
    return SelectedModel;

  return Model;
}

/// getOptLevel - Returns the optimization level: None, Less,
/// Default, or Aggressive.
CodeGenOpt::Level TargetMachine::getOptLevel() const {
  if (!CodeGenInfo)
    return CodeGenOpt::Default;
  return CodeGenInfo->getOptLevel();
}

bool TargetMachine::getAsmVerbosityDefault() {
  return AsmVerbosityDefault;
}

void TargetMachine::setAsmVerbosityDefault(bool V) {
  AsmVerbosityDefault = V;
}

bool TargetMachine::getFunctionSections() {
  return FunctionSections;
}

bool TargetMachine::getDataSections() {
  return DataSections;
}

void TargetMachine::setFunctionSections(bool V) {
  FunctionSections = V;
}

void TargetMachine::setDataSections(bool V) {
  DataSections = V;
}
