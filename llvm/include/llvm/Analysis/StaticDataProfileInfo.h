#ifndef LLVM_ANALYSIS_STATICDATAPROFILEINFO_H
#define LLVM_ANALYSIS_STATICDATAPROFILEINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

namespace memprof {
// Represents the eligibility status of a global variable for section prefix
// annotation. Other than AnnotationOk, each enum value indicates a specific
// reason for ineligibility.
enum class AnnotationKind : uint8_t {
  AnnotationOK,
  DeclForLinker,
  ExplicitSection,
  ReservedName,
};
/// Returns the annotation kind of the global variable \p GV.
AnnotationKind getAnnotationKind(const GlobalVariable &GV);

/// Returns true if the annotation kind of the global variable \p GV is
/// AnnotationOK.
bool IsAnnotationOK(const GlobalVariable &GV);
} // namespace memprof

/// A class that holds the constants that represent static data and their
/// profile information and provides methods to operate on them.
class StaticDataProfileInfo {
public:
  /// A constant is tracked only if the following conditions are met.
  ///   1) It has local (i.e., private or internal) linkage.
  //    2) Its data kind is one of {.rodata, .data, .bss, .data.rel.ro}.
  //    3) It's eligible for section prefix annotation. See `AnnotationKind`
  //       above for ineligible reasons.
  DenseMap<const Constant *, uint64_t> ConstantProfileCounts;

  /// Keeps track of the constants that are seen at least once without profile
  /// counts.
  DenseSet<const Constant *> ConstantWithoutCounts;

  /// If \p C has a count, return it. Otherwise, return std::nullopt.
  LLVM_ABI std::optional<uint64_t>
  getConstantProfileCount(const Constant *C) const;

  /// Use signed enums for enum value comparison, and make 'LukewarmOrUnknown'
  /// as 0 so any accidentally uninitialized value will default to unknown.
  enum class StaticDataHotness : int8_t {
    Cold = -1,
    LukewarmOrUnknown = 0,
    Hot = 1,
  };

  /// Return the hotness of the constant \p C based on its profile count \p
  /// Count.
  LLVM_ABI StaticDataHotness getConstantHotnessUsingProfileCount(
      const Constant *C, const ProfileSummaryInfo *PSI, uint64_t Count) const;

  /// Return the hotness based on section prefix \p SectionPrefix.
  LLVM_ABI StaticDataHotness getSectionHotnessUsingDataAccessProfile(
      std::optional<StringRef> SectionPrefix) const;

  /// Return the string representation of the hotness enum \p Hotness.
  LLVM_ABI StringRef hotnessToStr(StaticDataHotness Hotness) const;

  bool EnableDataAccessProf = false;

public:
  StaticDataProfileInfo(bool EnableDataAccessProf)
      : EnableDataAccessProf(EnableDataAccessProf) {}

  /// If \p Count is not nullopt, add it to the profile count of the constant \p
  /// C in a saturating way, and clamp the count to \p getInstrMaxCountValue if
  /// the result exceeds it. Otherwise, mark the constant as having no profile
  /// count.
  LLVM_ABI void addConstantProfileCount(const Constant *C,
                                        std::optional<uint64_t> Count);

  /// Given a constant \p C, returns a section prefix.
  /// If \p C is a global variable, the section prefix is the bigger one
  /// between its existing section prefix and its use profile count. Otherwise,
  /// the section prefix is based on its use profile count.
  LLVM_ABI StringRef getConstantSectionPrefix(
      const Constant *C, const ProfileSummaryInfo *PSI) const;
};

/// This wraps the StaticDataProfileInfo object as an immutable pass, for a
/// backend pass to operate on.
class LLVM_ABI StaticDataProfileInfoWrapperPass : public ImmutablePass {
public:
  static char ID;
  StaticDataProfileInfoWrapperPass();
  bool doInitialization(Module &M) override;
  bool doFinalization(Module &M) override;

  StaticDataProfileInfo &getStaticDataProfileInfo() { return *Info; }
  const StaticDataProfileInfo &getStaticDataProfileInfo() const {
    return *Info;
  }

  /// This pass provides StaticDataProfileInfo for reads/writes but does not
  /// modify \p M or other analysis. All analysis are preserved.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

private:
  std::unique_ptr<StaticDataProfileInfo> Info;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_STATICDATAPROFILEINFO_H
