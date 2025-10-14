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
  /// Accummulate the profile count of a constant that will be lowered to static
  /// data sections.
  DenseMap<const Constant *, uint64_t> ConstantProfileCounts;

  /// Keeps track of the constants that are seen at least once without profile
  /// counts.
  DenseSet<const Constant *> ConstantWithoutCounts;

  /// If \p C has a count, return it. Otherwise, return std::nullopt.
  LLVM_ABI std::optional<uint64_t>
  getConstantProfileCount(const Constant *C) const;

public:
  StaticDataProfileInfo() = default;

  /// If \p Count is not nullopt, add it to the profile count of the constant \p
  /// C in a saturating way, and clamp the count to \p getInstrMaxCountValue if
  /// the result exceeds it. Otherwise, mark the constant as having no profile
  /// count.
  LLVM_ABI void addConstantProfileCount(const Constant *C,
                                        std::optional<uint64_t> Count);

  /// Return a section prefix for the constant \p C based on its profile count.
  /// - If a constant doesn't have a counter, return an empty string.
  /// - Otherwise,
  ///   - If it has a hot count, return "hot".
  ///   - If it is seen by unprofiled function, return an empty string.
  ///   - If it has a cold count, return "unlikely".
  ///   - Otherwise (e.g. it's used by lukewarm functions), return an empty
  ///     string.
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
