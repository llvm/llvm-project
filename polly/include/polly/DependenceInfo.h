//===--- polly/DependenceInfo.h - Polyhedral dependency analysis *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Calculate the data dependency relations for a Scop using ISL.
//
// The integer set library (ISL) from Sven has an integrated dependency analysis
// to calculate data dependences. This pass takes advantage of this and
// calculates those dependences of a Scop.
//
// The dependences in this pass are exact in terms that for a specific read
// statement instance only the last write statement instance is returned. In
// case of may-writes, a set of possible write instances is returned. This
// analysis will never produce redundant dependences.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_DEPENDENCE_INFO_H
#define POLLY_DEPENDENCE_INFO_H

#include "polly/ScopPass.h"
#include "isl/ctx.h"

struct isl_pw_aff;
struct isl_union_map;
struct isl_union_set;
struct isl_map;
struct isl_set;
struct clast_for;

using namespace llvm;

namespace polly {

class Scop;
class ScopStmt;
class MemoryAccess;

/// @brief The accumulated dependence information for a SCoP.
///
/// The Dependences struct holds all dependence information we collect and
/// compute for one SCoP. It also offers an interface that allows users to
/// query only specific parts.
struct Dependences {

  /// @brief Map type for reduction dependences.
  using ReductionDependencesMapTy = DenseMap<MemoryAccess *, isl_map *>;

  /// @brief Map type to associate statements with schedules.
  using StatementToIslMapTy = DenseMap<ScopStmt *, isl_map *>;

  /// @brief The type of the dependences.
  ///
  /// Reduction dependences are separated from RAW/WAW/WAR dependences because
  /// we can ignore them during the scheduling. That's because the order
  /// in which the reduction statements are executed does not matter. However,
  /// if they are executed in parallel we need to take additional measures
  /// (e.g, privatization) to ensure a correct result. The (reverse) transitive
  /// closure of the reduction dependences are used to check for parallel
  /// executed reduction statements during code generation. These dependences
  /// connect all instances of a reduction with each other, they are therefore
  /// cyclic and possibly "reversed".
  enum Type {
    // Write after read
    TYPE_WAR = 1 << 0,

    // Read after write
    TYPE_RAW = 1 << 1,

    // Write after write
    TYPE_WAW = 1 << 2,

    // Reduction dependences
    TYPE_RED = 1 << 3,

    // Transitive closure of the reduction dependences (& the reverse)
    TYPE_TC_RED = 1 << 4,
  };

  /// @brief Get the dependences of type @p Kinds.
  ///
  /// @param Kinds This integer defines the different kinds of dependences
  ///              that will be returned. To return more than one kind, the
  ///              different kinds are 'ored' together.
  __isl_give isl_union_map *getDependences(int Kinds) const;

  /// @brief Report if valid dependences are available.
  bool hasValidDependences() const;

  /// @brief Return the reduction dependences caused by @p MA.
  ///
  /// @return The reduction dependences caused by @p MA or nullptr if none.
  __isl_give isl_map *getReductionDependences(MemoryAccess *MA) const;

  /// @brief Return all reduction dependences.
  const ReductionDependencesMapTy &getReductionDependences() const {
    return ReductionDependences;
  }

  /// @brief Check if a partial schedule is parallel wrt to @p Deps.
  ///
  /// @param Schedule       The subset of the schedule space that we want to
  ///                       check.
  /// @param Deps           The dependences @p Schedule needs to respect.
  /// @param MinDistancePtr If not nullptr, the minimal dependence distance will
  ///                       be returned at the address of that pointer
  ///
  /// @return Returns true, if executing parallel the outermost dimension of
  ///         @p Schedule is valid according to the dependences @p Deps.
  bool isParallel(__isl_keep isl_union_map *Schedule,
                  __isl_take isl_union_map *Deps,
                  __isl_give isl_pw_aff **MinDistancePtr = nullptr) const;

  /// @brief Check if a new schedule is valid.
  ///
  /// @param S             The current SCoP.
  /// @param NewSchedules  The new schedules
  ///
  /// @return True if the new schedule is valid, false it it reverses
  ///         dependences.
  bool isValidSchedule(Scop &S, StatementToIslMapTy *NewSchedules) const;

  /// @brief Print the stored dependence information.
  void print(llvm::raw_ostream &OS) const;

  /// @brief Dump the dependence information stored to the dbgs stream.
  void dump() const;

  /// @brief Allow the DependenceInfo access to private members and methods.
  ///
  /// To restrict access to the internal state, only the DependenceInfo class
  /// is able to call or modify a Dependences struct.
  friend class DependenceInfo;

  /// @brief Destructor that will free internal objects.
  ~Dependences() { releaseMemory(); }

private:
  /// @brief Create an empty Dependences struct.
  explicit Dependences(const std::shared_ptr<isl_ctx> &IslCtx)
      : RAW(nullptr), WAR(nullptr), WAW(nullptr), RED(nullptr), TC_RED(nullptr),
        IslCtx(IslCtx) {}

  /// @brief Calculate and add at the privatization dependences.
  void addPrivatizationDependences();

  /// @brief Calculate the dependences for a certain SCoP @p S.
  void calculateDependences(Scop &S);

  /// @brief Set the reduction dependences for @p MA to @p Deps.
  void setReductionDependences(MemoryAccess *MA, __isl_take isl_map *Deps);

  /// @brief Free the objects associated with this Dependences struct.
  ///
  /// The Dependences struct will again be "empty" afterwards.
  void releaseMemory();

  /// @brief The different basic kinds of dependences we calculate.
  isl_union_map *RAW;
  isl_union_map *WAR;
  isl_union_map *WAW;

  /// @brief The special reduction dependences.
  isl_union_map *RED;

  /// @brief The (reverse) transitive closure of reduction dependences.
  isl_union_map *TC_RED;

  /// @brief Mapping from memory accesses to their reduction dependences.
  ReductionDependencesMapTy ReductionDependences;

  /// @brief Isl context from the SCoP.
  std::shared_ptr<isl_ctx> IslCtx;
};

class DependenceInfo : public ScopPass {
public:
  static char ID;

  /// @brief Construct a new DependenceInfo pass.
  DependenceInfo() : ScopPass(ID) {}

  /// @brief Return the dependence information for the current SCoP.
  const Dependences &getDependences() { return *D; }

  /// @brief Recompute dependences from schedule and memory accesses.
  void recomputeDependences();

  /// @brief Compute the dependence information for the SCoP @p S.
  bool runOnScop(Scop &S) override;

  /// @brief Print the dependences for the given SCoP to @p OS.
  void printScop(raw_ostream &OS, Scop &) const override { D->print(OS); }

  /// @brief Release the internal memory.
  void releaseMemory() override { D.reset(); }

  /// @brief Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  Scop *S;

  /// @brief Dependences struct for the current SCoP.
  std::unique_ptr<Dependences> D;
};

} // End polly namespace.

namespace llvm {
class PassRegistry;
void initializeDependenceInfoPass(llvm::PassRegistry &);
}

#endif
