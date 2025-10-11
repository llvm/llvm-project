#ifndef LLVM_ANALYSIS_STATICDATAPROFILEINFO_H
#define LLVM_ANALYSIS_STATICDATAPROFILEINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

bool IsReservedGlobalVariable(const GlobalVariable &GV);

/// A class that holds the constants that represent static data and their
/// profile information and provides methods to operate on them.
class StaticDataProfileInfo {
public:
  /// A constant and its profile count.
  /// A constant is tracked if both conditions are met:
  ///   1) It has local (i.e., private or internal) linkage.
  //    2) Its data kind is one of {.rodata, .data, .bss, .data.rel.ro}.
  DenseMap<const Constant *, uint64_t> ConstantProfileCounts;

  /// Keeps track of the constants that are seen at least once without profile
  /// counts.
  DenseSet<const Constant *> ConstantWithoutCounts;

  /// If \p C has a count, return it. Otherwise, return std::nullopt.
  LLVM_ABI std::optional<uint64_t>
  getConstantProfileCount(const Constant *C) const;

  enum class StaticDataHotness : uint8_t {
    Cold = 0,
    LukewarmOrUnknown = 1,
    Hot = 2,
  };

  LLVM_ABI StaticDataHotness getSectionHotnessUsingProfileCount(
      const Constant *C, const ProfileSummaryInfo *PSI, uint64_t Count) const;
  LLVM_ABI StaticDataHotness
  getSectionHotnessUsingDAP(std::optional<StringRef> SectionPrefix) const;

  LLVM_ABI StringRef hotnessToStr(StaticDataHotness Hotness) const;

  bool HasDataAccessProf = false;

public:
  StaticDataProfileInfo(bool HasDataAccessProf)
      : HasDataAccessProf(HasDataAccessProf) {}
  StaticDataProfileInfo() = default;

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
