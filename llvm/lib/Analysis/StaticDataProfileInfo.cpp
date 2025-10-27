#include "llvm/Analysis/StaticDataProfileInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/InstrProf.h"

#define DEBUG_TYPE "static-data-profile-info"

using namespace llvm;

namespace llvm {
namespace memprof {
// Returns true iff the global variable has custom section either by
// __attribute__((section("name")))
// (https://clang.llvm.org/docs/AttributeReference.html#section-declspec-allocate)
// or #pragma clang section directives
// (https://clang.llvm.org/docs/LanguageExtensions.html#specifying-section-names-for-global-objects-pragma-clang-section).
static bool hasExplicitSectionName(const GlobalVariable &GVar) {
  if (GVar.hasSection())
    return true;

  auto Attrs = GVar.getAttributes();
  if (Attrs.hasAttribute("bss-section") || Attrs.hasAttribute("data-section") ||
      Attrs.hasAttribute("relro-section") ||
      Attrs.hasAttribute("rodata-section"))
    return true;
  return false;
}

AnnotationKind getAnnotationKind(const GlobalVariable &GV) {
  if (GV.isDeclarationForLinker())
    return AnnotationKind::DeclForLinker;
  // Skip 'llvm.'-prefixed global variables conservatively because they are
  // often handled specially,
  StringRef Name = GV.getName();
  if (Name.starts_with("llvm."))
    return AnnotationKind::ReservedName;
  // Respect user-specified custom data sections.
  if (hasExplicitSectionName(GV))
    return AnnotationKind::ExplicitSection;
  return AnnotationKind::AnnotationOK;
}

bool IsAnnotationOK(const GlobalVariable &GV) {
  return getAnnotationKind(GV) == AnnotationKind::AnnotationOK;
}
} // namespace memprof
} // namespace llvm

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

StaticDataProfileInfo::StaticDataHotness
StaticDataProfileInfo::getConstantHotnessUsingProfileCount(
    const Constant *C, const ProfileSummaryInfo *PSI, uint64_t Count) const {
  // The accummulated counter shows the constant is hot. Return enum 'hot'
  // whether this variable is seen by unprofiled functions or not.
  if (PSI->isHotCount(Count))
    return StaticDataHotness::Hot;
  // The constant is not hot, and seen by unprofiled functions. We don't want to
  // assign it to unlikely sections, even if the counter says 'cold'. So return
  // enum 'LukewarmOrUnknown'.
  if (ConstantWithoutCounts.count(C))
    return StaticDataHotness::LukewarmOrUnknown;
  // The accummulated counter shows the constant is cold so return enum 'cold'.
  if (PSI->isColdCount(Count))
    return StaticDataHotness::Cold;

  return StaticDataHotness::LukewarmOrUnknown;
}

StaticDataProfileInfo::StaticDataHotness
StaticDataProfileInfo::getSectionHotnessUsingDataAccessProfile(
    std::optional<StringRef> MaybeSectionPrefix) const {
  if (!MaybeSectionPrefix)
    return StaticDataHotness::LukewarmOrUnknown;
  StringRef Prefix = *MaybeSectionPrefix;
  assert((Prefix == "hot" || Prefix == "unlikely") &&
         "Expect section_prefix to be one of hot or unlikely");
  return Prefix == "hot" ? StaticDataHotness::Hot : StaticDataHotness::Cold;
}

StringRef StaticDataProfileInfo::hotnessToStr(StaticDataHotness Hotness) const {
  switch (Hotness) {
  case StaticDataHotness::Cold:
    return "unlikely";
  case StaticDataHotness::Hot:
    return "hot";
  default:
    return "";
  }
}

std::optional<uint64_t>
StaticDataProfileInfo::getConstantProfileCount(const Constant *C) const {
  auto I = ConstantProfileCounts.find(C);
  if (I == ConstantProfileCounts.end())
    return std::nullopt;
  return I->second;
}

StringRef StaticDataProfileInfo::getConstantSectionPrefix(
    const Constant *C, const ProfileSummaryInfo *PSI) const {
  std::optional<uint64_t> Count = getConstantProfileCount(C);

#ifndef NDEBUG
  auto DbgPrintPrefix = [](StringRef Prefix) {
    return Prefix.empty() ? "<empty>" : Prefix;
  };
#endif

  if (EnableDataAccessProf) {
    // Module flag `HasDataAccessProf` is 1 -> empty section prefix means
    // unknown hotness except for string literals.
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(C);
        GV && llvm::memprof::IsAnnotationOK(*GV) &&
        !GV->getName().starts_with(".str")) {
      auto HotnessFromDataAccessProf =
          getSectionHotnessUsingDataAccessProfile(GV->getSectionPrefix());

      if (!Count) {
        StringRef Prefix = hotnessToStr(HotnessFromDataAccessProf);
        LLVM_DEBUG(dbgs() << GV->getName() << " has section prefix "
                          << DbgPrintPrefix(Prefix)
                          << ", solely from data access profiles\n");
        return Prefix;
      }

      // Both data access profiles and PGO counters are available. Use the
      // hotter one.
      auto HotnessFromPGO = getConstantHotnessUsingProfileCount(C, PSI, *Count);
      StaticDataHotness GlobalVarHotness = StaticDataHotness::LukewarmOrUnknown;
      if (HotnessFromDataAccessProf == StaticDataHotness::Hot ||
          HotnessFromPGO == StaticDataHotness::Hot) {
        GlobalVarHotness = StaticDataHotness::Hot;
      } else if (HotnessFromDataAccessProf ==
                     StaticDataHotness::LukewarmOrUnknown ||
                 HotnessFromPGO == StaticDataHotness::LukewarmOrUnknown) {
        GlobalVarHotness = StaticDataHotness::LukewarmOrUnknown;
      } else {
        GlobalVarHotness = StaticDataHotness::Cold;
      }
      StringRef Prefix = hotnessToStr(GlobalVarHotness);
      LLVM_DEBUG(
          dbgs() << GV->getName() << " has section prefix "
                 << DbgPrintPrefix(Prefix)
                 << ", the max from data access profiles as "
                 << DbgPrintPrefix(hotnessToStr(HotnessFromDataAccessProf))
                 << " and PGO counters as "
                 << DbgPrintPrefix(hotnessToStr(HotnessFromPGO)) << "\n");
      return Prefix;
    }
  }
  if (!Count)
    return "";
  return hotnessToStr(getConstantHotnessUsingProfileCount(C, PSI, *Count));
}

bool StaticDataProfileInfoWrapperPass::doInitialization(Module &M) {
  bool EnableDataAccessProf = false;
  if (auto *MD = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("EnableDataAccessProf")))
    EnableDataAccessProf = MD->getZExtValue();
  Info.reset(new StaticDataProfileInfo(EnableDataAccessProf));
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
