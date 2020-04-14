//===-- PPCLowerMASSVEntries.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of MASSV (SIMD) entries for specific PowerPC
// subtargets.
// Following is an example of a conversion specific to Power9 subtarget:
// __sind2_massv ---> __sind2_P9
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCSubtarget.h"
#include "PPCTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "ppc-lower-massv-entries"

using namespace llvm;

namespace {

// Length of the suffix "massv", which is specific to IBM MASSV library entries.
const unsigned MASSVSuffixLength = 5;

static StringRef MASSVFuncs[] = {
#define TLI_DEFINE_MASSV_VECFUNCS_NAMES
#include "llvm/Analysis/VecFuncs.def"
};

class PPCLowerMASSVEntries : public ModulePass {
public:
  static char ID;

  PPCLowerMASSVEntries() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override { return "PPC Lower MASS Entries"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

private:
  static bool isMASSVFunc(StringRef Name);
  static StringRef getCPUSuffix(const PPCSubtarget *Subtarget);
  static std::string createMASSVFuncName(Function &Func,
                                         const PPCSubtarget *Subtarget);
  bool lowerMASSVCall(CallInst *CI, Function &Func, Module &M,
                      const PPCSubtarget *Subtarget);
};

} // namespace

/// Checks if the specified function name represents an entry in the MASSV
/// library.
bool PPCLowerMASSVEntries::isMASSVFunc(StringRef Name) {
  auto Iter = std::find(std::begin(MASSVFuncs), std::end(MASSVFuncs), Name);
  return Iter != std::end(MASSVFuncs);
}

// FIXME:
/// Returns a string corresponding to the specified PowerPC subtarget. e.g.:
/// "P8" for Power8, "P9" for Power9. The string is used as a suffix while
/// generating subtarget-specific MASSV library functions. Current support
/// includes  Power8 and Power9 subtargets.
StringRef PPCLowerMASSVEntries::getCPUSuffix(const PPCSubtarget *Subtarget) {
  // Assume Power8 when Subtarget is unavailable.
  if (!Subtarget)
    return "P8";
  if (Subtarget->hasP9Vector())
    return "P9";
  if (Subtarget->hasP8Vector())
    return "P8";

  report_fatal_error("Unsupported Subtarget: MASSV is supported only on "
                     "Power8 and Power9 subtargets.");
}

/// Creates PowerPC subtarget-specific name corresponding to the specified
/// generic MASSV function, and the PowerPC subtarget.
std::string
PPCLowerMASSVEntries::createMASSVFuncName(Function &Func,
                                          const PPCSubtarget *Subtarget) {
  StringRef Suffix = getCPUSuffix(Subtarget);
  auto GenericName = Func.getName().drop_back(MASSVSuffixLength).str();
  std::string MASSVEntryName = GenericName + Suffix.str();
  return MASSVEntryName;
}

/// Lowers generic MASSV entries to PowerPC subtarget-specific MASSV entries.
/// e.g.: __sind2_massv --> __sind2_P9 for a Power9 subtarget.
/// Both function prototypes and their callsites are updated during lowering.
bool PPCLowerMASSVEntries::lowerMASSVCall(CallInst *CI, Function &Func,
                                          Module &M,
                                          const PPCSubtarget *Subtarget) {
  if (CI->use_empty())
    return false;

  std::string MASSVEntryName = createMASSVFuncName(Func, Subtarget);
  FunctionCallee FCache = M.getOrInsertFunction(
      MASSVEntryName, Func.getFunctionType(), Func.getAttributes());

  CI->setCalledFunction(FCache);  

  return true;
}

bool PPCLowerMASSVEntries::runOnModule(Module &M) {
  bool Changed = false;

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return Changed;

  auto &TM = TPC->getTM<PPCTargetMachine>();
  const PPCSubtarget *Subtarget;

  for (Function &Func : M) {
    if (!Func.isDeclaration())
      continue;

    if (!isMASSVFunc(Func.getName()))
      continue;

    // Call to lowerMASSVCall() invalidates the iterator over users upon
    // replacing the users. Precomputing the current list of users allows us to
    // replace all the call sites.
    SmallVector<User *, 4> MASSVUsers;
    for (auto *User: Func.users())
      MASSVUsers.push_back(User);
    
    for (auto *User : MASSVUsers) {
      auto *CI = dyn_cast<CallInst>(User);
      if (!CI)
        continue;

      Subtarget = &TM.getSubtarget<PPCSubtarget>(*CI->getParent()->getParent());
      Changed |= lowerMASSVCall(CI, Func, M, Subtarget);
    }
  }

  return Changed;
}

char PPCLowerMASSVEntries::ID = 0;

char &llvm::PPCLowerMASSVEntriesID = PPCLowerMASSVEntries::ID;

INITIALIZE_PASS(PPCLowerMASSVEntries, DEBUG_TYPE, "Lower MASSV entries", false,
                false)

ModulePass *llvm::createPPCLowerMASSVEntriesPass() {
  return new PPCLowerMASSVEntries();
}
