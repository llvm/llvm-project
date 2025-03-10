#include "llvm/Analysis/StaticDataProfileInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/InstrProf.h"
#include <sys/types.h>

using namespace llvm;
void StaticDataProfileInfo::addConstantProfileCount(
    const Constant *C, std::optional<uint64_t> Count) {
  if (!Count) {
    ConstantWithoutCounts.insert(C);
    return;
  }
  uint64_t &OriginalCount = ConstantProfileCounts[C];
  OriginalCount += llvm::SaturatingAdd(*Count, OriginalCount);
  // Clamp the count to getInstrMaxCountValue. InstrFDO reserves a few
  // large values for special use.
  if (OriginalCount > getInstrMaxCountValue())
    OriginalCount = getInstrMaxCountValue();
}

std::optional<uint64_t>
StaticDataProfileInfo::getConstantProfileCount(const Constant *C) const {
  auto I = ConstantProfileCounts.find(C);
  if (I == ConstantProfileCounts.end())
    return std::nullopt;
  return I->second;
}

bool StaticDataProfileInfoWrapperPass::doInitialization(Module &M) {
  Info.reset(new StaticDataProfileInfo());
  return false;
}

bool StaticDataProfileInfoWrapperPass::doFinalization(Module &M) {
  Info.reset();
  return false;
}

INITIALIZE_PASS(StaticDataProfileInfoWrapperPass, "static-data-profile-info",
                "Static Data Profile Info", false, true)

StaticDataProfileInfoWrapperPass::StaticDataProfileInfoWrapperPass()
    : ImmutablePass(ID) {
  initializeStaticDataProfileInfoWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

char StaticDataProfileInfoWrapperPass::ID = 0;
