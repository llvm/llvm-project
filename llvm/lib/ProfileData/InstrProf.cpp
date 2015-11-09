//=-- InstrProf.cpp - Instrumented profiling format support -----------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for clang's instrumentation based PGO and
// coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

namespace {
class InstrProfErrorCategoryType : public std::error_category {
  const char *name() const LLVM_NOEXCEPT override { return "llvm.instrprof"; }
  std::string message(int IE) const override {
    instrprof_error E = static_cast<instrprof_error>(IE);
    switch (E) {
    case instrprof_error::success:
      return "Success";
    case instrprof_error::eof:
      return "End of File";
    case instrprof_error::bad_magic:
      return "Invalid profile data (bad magic)";
    case instrprof_error::bad_header:
      return "Invalid profile data (file header is corrupt)";
    case instrprof_error::unsupported_version:
      return "Unsupported profiling format version";
    case instrprof_error::unsupported_hash_type:
      return "Unsupported profiling hash";
    case instrprof_error::too_large:
      return "Too much profile data";
    case instrprof_error::truncated:
      return "Truncated profile data";
    case instrprof_error::malformed:
      return "Malformed profile data";
    case instrprof_error::unknown_function:
      return "No profile data available for function";
    case instrprof_error::hash_mismatch:
      return "Function hash mismatch";
    case instrprof_error::count_mismatch:
      return "Function count mismatch";
    case instrprof_error::counter_overflow:
      return "Counter overflow";
    case instrprof_error::value_site_count_mismatch:
      return "Function's value site counts mismatch";
    }
    llvm_unreachable("A value of instrprof_error has no message.");
  }
};
}

static ManagedStatic<InstrProfErrorCategoryType> ErrorCategory;

const std::error_category &llvm::instrprof_category() {
  return *ErrorCategory;
}

namespace llvm {

std::string getPGOFuncName(StringRef RawFuncName,
                           GlobalValue::LinkageTypes Linkage,
                           StringRef FileName) {

  // Function names may be prefixed with a binary '1' to indicate
  // that the backend should not modify the symbols due to any platform
  // naming convention. Do not include that '1' in the PGO profile name.
  if (RawFuncName[0] == '\1')
    RawFuncName = RawFuncName.substr(1);

  std::string FuncName = RawFuncName;
  if (llvm::GlobalValue::isLocalLinkage(Linkage)) {
    // For local symbols, prepend the main file name to distinguish them.
    // Do not include the full path in the file name since there's no guarantee
    // that it will stay the same, e.g., if the files are checked out from
    // version control in different locations.
    if (FileName.empty())
      FuncName = FuncName.insert(0, "<unknown>:");
    else
      FuncName = FuncName.insert(0, FileName.str() + ":");
  }
  return FuncName;
}

std::string getPGOFuncName(const Function &F) {
  return getPGOFuncName(F.getName(), F.getLinkage(), F.getParent()->getName());
}

GlobalVariable *createPGOFuncNameVar(Module &M,
                                     GlobalValue::LinkageTypes Linkage,
                                     StringRef FuncName) {

  // We generally want to match the function's linkage, but available_externally
  // and extern_weak both have the wrong semantics, and anything that doesn't
  // need to link across compilation units doesn't need to be visible at all.
  if (Linkage == GlobalValue::ExternalWeakLinkage)
    Linkage = GlobalValue::LinkOnceAnyLinkage;
  else if (Linkage == GlobalValue::AvailableExternallyLinkage)
    Linkage = GlobalValue::LinkOnceODRLinkage;
  else if (Linkage == GlobalValue::InternalLinkage ||
           Linkage == GlobalValue::ExternalLinkage)
    Linkage = GlobalValue::PrivateLinkage;

  auto *Value = ConstantDataArray::getString(M.getContext(), FuncName, false);
  auto FuncNameVar =
      new GlobalVariable(M, Value->getType(), true, Linkage, Value,
                         Twine(getInstrProfNameVarPrefix()) + FuncName);

  // Hide the symbol so that we correctly get a copy for each executable.
  if (!GlobalValue::isLocalLinkage(FuncNameVar->getLinkage()))
    FuncNameVar->setVisibility(GlobalValue::HiddenVisibility);

  return FuncNameVar;
}

GlobalVariable *createPGOFuncNameVar(Function &F, StringRef FuncName) {
  return createPGOFuncNameVar(*F.getParent(), F.getLinkage(), FuncName);
}
}
