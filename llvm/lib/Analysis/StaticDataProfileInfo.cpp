#include "llvm/Analysis/StaticDataProfileInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/InstrProf.h"

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
  if (!Count)
    return "";
  return hotnessToStr(getConstantHotnessUsingProfileCount(C, PSI, *Count));
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
    : ImmutablePass(ID) {}

char StaticDataProfileInfoWrapperPass::ID = 0;
