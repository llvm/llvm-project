#include "llvm/Analysis/StaticDataProfileInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/InstrProf.h"

#define DEBUG_TYPE "static-data-profile-info"

using namespace llvm;

extern cl::opt<bool> AnnotateStaticDataSectionPrefix;

bool llvm::IsReservedGlobalVariable(const GlobalVariable &GV) {
  return GV.getName().starts_with("llvm.");
}

void StaticDataProfileInfo::addConstantProfileCount(
    const Constant *C, std::optional<uint64_t> Count) {
  if (!Count) {
    ConstantWithoutCounts.insert(C);
    return;
  }
  uint64_t &OriginalCount = ConstantProfileCounts[C];
  OriginalCount = llvm::SaturatingAdd(*Count, OriginalCount);
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

StaticDataProfileInfo::StaticDataHotness
StaticDataProfileInfo::getSectionHotnessUsingProfileCount(
    const Constant *C, const ProfileSummaryInfo *PSI, uint64_t Count) const {
  // The accummulated counter shows the constant is hot. Return 'hot' whether
  // this variable is seen by unprofiled functions or not.
  if (PSI->isHotCount(Count))
    return StaticDataHotness::Hot;
  // The constant is not hot, and seen by unprofiled functions. We don't want to
  // assign it to unlikely sections, even if the counter says 'cold'. So return
  // an empty prefix before checking whether the counter is cold.
  if (ConstantWithoutCounts.count(C))
    return StaticDataHotness::LukewarmOrUnknown;
  // The accummulated counter shows the constant is cold. Return 'unlikely'.
  if (PSI->isColdCount(Count))
    return StaticDataHotness::Cold;

  return StaticDataHotness::LukewarmOrUnknown;
}

StringRef StaticDataProfileInfo::hotnessToStr(
    StaticDataProfileInfo::StaticDataHotness Hotness) const {
  switch (Hotness) {
  case StaticDataProfileInfo::StaticDataHotness::Cold:
    return "unlikely";
  case StaticDataProfileInfo::StaticDataHotness::Hot:
    return "hot";
  default:
    return "";
  }
}

StaticDataProfileInfo::StaticDataHotness
StaticDataProfileInfo::getSectionHotnessUsingDAP(
    std::optional<StringRef> MaybeSectionPrefix) const {
  if (!MaybeSectionPrefix)
    return StaticDataProfileInfo::StaticDataHotness::LukewarmOrUnknown;
  StringRef Prefix = *MaybeSectionPrefix;
  assert((Prefix == "hot" || Prefix == "unlikely") &&
         "Expect section_prefix to be one of hot or unlikely");
  return Prefix == "hot" ? StaticDataProfileInfo::StaticDataHotness::Hot
                         : StaticDataProfileInfo::StaticDataHotness::Cold;
}

StringRef StaticDataProfileInfo::getConstantSectionPrefix(
    const Constant *C, const ProfileSummaryInfo *PSI) const {
  std::optional<uint64_t> Count = getConstantProfileCount(C);

  if (HasDataAccessProf) {
    // Module flag `HasDataAccessProf` is 1 -> empty section prefix means
    // unknown hotness except for string literals.
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(C);
        GV && !IsReservedGlobalVariable(*GV) &&
        !GV->getName().starts_with(".str")) {
      auto HotnessFromDAP = getSectionHotnessUsingDAP(GV->getSectionPrefix());

      if (!Count) {
        // Use data access profiles to infer hotness when the profile counter
        // isn't computed.
        return hotnessToStr(HotnessFromDAP);
      }

      // Both DAP and PGO counters are available. Use the hotter one.
      auto HotnessFromPGO = getSectionHotnessUsingProfileCount(C, PSI, *Count);
      return hotnessToStr(std::max(HotnessFromDAP, HotnessFromPGO));
    }
  }

  if (!Count)
    return "";
  return hotnessToStr(getSectionHotnessUsingProfileCount(C, PSI, *Count));
}

bool StaticDataProfileInfoWrapperPass::doInitialization(Module &M) {
  bool HasDataAccessProf = false;
  if (auto *MD = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("HasDataAccessProf")))
    HasDataAccessProf = MD->getZExtValue();
  Info.reset(new StaticDataProfileInfo(HasDataAccessProf));
  return false;
}

bool StaticDataProfileInfoWrapperPass::doFinalization(Module &M) {
  Info.reset();
  return false;
}

INITIALIZE_PASS(StaticDataProfileInfoWrapperPass, "static-data-profile-info",
                "Static Data Profile Info", false, true)

StaticDataProfileInfoWrapperPass::StaticDataProfileInfoWrapperPass()
    : ImmutablePass(ID) {}

char StaticDataProfileInfoWrapperPass::ID = 0;
