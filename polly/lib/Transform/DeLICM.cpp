//===------ DeLICM.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Undo the effect of Loop Invariant Code Motion (LICM) and
// GVN Partial Redundancy Elimination (PRE) on SCoP-level.
//
// Namely, remove register/scalar dependencies by mapping them back to array
// elements.
//
// The algorithms here work on the scatter space - the image space of the
// schedule returned by Scop::getSchedule(). We call an element in that space a
// "timepoint". Timepoints are lexicographically ordered such that we can
// defined ranges in the scatter space. We use two flavors of such ranges:
// Timepoint sets and zones. A timepoint set is simply a subset of the scatter
// space and is directly stored as isl_set.
//
// Zones are used to describe the space between timepoints as open sets, i.e.
// they do not contain the extrema. Using isl rational sets to express these
// would be overkill. We also cannot store them as the integer timepoints they
// contain; the (nonempty) zone between 1 and 2 would be empty and
// indistinguishable from e.g. the zone between 3 and 4. Also, we cannot store
// the integer set including the extrema; the set ]1,2[ + ]3,4[ could be
// coalesced to ]1,3[, although we defined the range [2,3] to be not in the set.
// Instead, we store the "half-open" integer extrema, including the lower bound,
// but excluding the upper bound. Examples:
//
// * The set { [i] : 1 <= i <= 3 } represents the zone ]0,3[ (which contains the
//   integer points 1 and 2, but not 0 or 3)
//
// * { [1] } represents the zone ]0,1[
//
// * { [i] : i = 1 or i = 3 } represents the zone ]0,1[ + ]2,3[
//
// Therefore, an integer i in the set represents the zone ]i-1,i[, i.e. strictly
// speaking the integer points never belong to the zone. However, depending an
// the interpretation, one might want to include them. Part of the
// interpretation may not be known when the zone is constructed.
//
// Reads are assumed to always take place before writes, hence we can think of
// reads taking place at the beginning of a timepoint and writes at the end.
//
// Let's assume that the zone represents the lifetime of a variable. That is,
// the zone begins with a write that defines the value during its lifetime and
// ends with the last read of that value. In the following we consider whether a
// read/write at the beginning/ending of the lifetime zone should be within the
// zone or outside of it.
//
// * A read at the timepoint that starts the live-range loads the previous
//   value. Hence, exclude the timepoint starting the zone.
//
// * A write at the timepoint that starts the live-range is not defined whether
//   it occurs before or after the write that starts the lifetime. We do not
//   allow this situation to occur. Hence, we include the timepoint starting the
//   zone to determine whether they are conflicting.
//
// * A read at the timepoint that ends the live-range reads the same variable.
//   We include the timepoint at the end of the zone to include that read into
//   the live-range. Doing otherwise would mean that the two reads access
//   different values, which would mean that the value they read are both alive
//   at the same time but occupy the same variable.
//
// * A write at the timepoint that ends the live-range starts a new live-range.
//   It must not be included in the live-range of the previous definition.
//
// All combinations of reads and writes at the endpoints are possible, but most
// of the time only the write->read (for instance, a live-range from definition
// to last use) and read->write (for instance, an unused range from last use to
// overwrite) and combinations are interesting (half-open ranges). write->write
// zones might be useful as well in some context to represent
// output-dependencies.
//
// @see convertZoneToTimepoints
//
//
// The code makes use of maps and sets in many different spaces. To not loose
// track in which space a set or map is expected to be in, variables holding an
// isl reference are usually annotated in the comments. They roughly follow isl
// syntax for spaces, but only the tuples, not the dimensions. The tuples have a
// meaning as follows:
//
// * Space[] - An unspecified tuple. Used for function parameters such that the
//             function caller can use it for anything they like.
//
// * Domain[] - A statement instance as returned by ScopStmt::getDomain()
//     isl_id_get_name: Stmt_<NameOfBasicBlock>
//     isl_id_get_user: Pointer to ScopStmt
//
// * Element[] - An array element as in the range part of
//               MemoryAccess::getAccessRelation()
//     isl_id_get_name: MemRef_<NameOfArrayVariable>
//     isl_id_get_user: Pointer to ScopArrayInfo
//
// * Scatter[] - Scatter space or space of timepoints
//     Has no tuple id
//
// * Zone[] - Range between timepoints as described above
//     Has no tuple id
//
// An annotation "{ Domain[] -> Scatter[] }" therefore means: A map from a
// statement instance to a timepoint, aka a schedule. There is only one scatter
// space, but most of the time multiple statements are processed in one set.
// This is why most of the time isl_union_map has to be used.
//
//===----------------------------------------------------------------------===//

#include "polly/DeLICM.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/ISLTools.h"
#define DEBUG_TYPE "polly-delicm"

using namespace polly;
using namespace llvm;

namespace {

/// Represent the knowledge of the contents of any array elements in any zone or
/// the knowledge we would add when mapping a scalar to an array element.
///
/// Every array element at every zone unit has one of two states:
///
/// - Unused: Not occupied by any value so a transformation can change it to
///   other values.
///
/// - Occupied: The element contains a value that is still needed.
///
/// The union of Unused and Unknown zones forms the universe, the set of all
/// elements at every timepoint. The universe can easily be derived from the
/// array elements that are accessed someway. Arrays that are never accessed
/// also never play a role in any computation and can hence be ignored. With a
/// given universe, only one of the sets needs to stored implicitly. Computing
/// the complement is also an expensive operation, hence this class has been
/// designed that only one of sets is needed while the other is assumed to be
/// implicit. It can still be given, but is mostly ignored.
///
/// There are two use cases for the Knowledge class:
///
/// 1) To represent the knowledge of the current state of ScopInfo. The unused
///    state means that an element is currently unused: there is no read of it
///    before the next overwrite. Also called 'Existing'.
///
/// 2) To represent the requirements for mapping a scalar to array elements. The
///    unused state means that there is no change/requirement. Also called
///    'Proposed'.
///
/// In addition to these states at unit zones, Knowledge needs to know when
/// values are written. This is because written values may have no lifetime (one
/// reason is that the value is never read). Such writes would therefore never
/// conflict, but overwrite values that might still be required. Another source
/// of problems are multiple writes to the same element at the same timepoint,
/// because their order is undefined.
class Knowledge {
private:
  /// { [Element[] -> Zone[]] }
  /// Set of array elements and when they are alive.
  /// Can contain a nullptr; in this case the set is implicitly defined as the
  /// complement of #Unused.
  ///
  /// The set of alive array elements is represented as zone, as the set of live
  /// values can differ depending on how the elements are interpreted.
  /// Assuming a value X is written at timestep [0] and read at timestep [1]
  /// without being used at any later point, then the value is alive in the
  /// interval ]0,1[. This interval cannot be represented by an integer set, as
  /// it does not contain any integer point. Zones allow us to represent this
  /// interval and can be converted to sets of timepoints when needed (e.g., in
  /// isConflicting when comparing to the write sets).
  /// @see convertZoneToTimepoints and this file's comment for more details.
  IslPtr<isl_union_set> Occupied;

  /// { [Element[] -> Zone[]] }
  /// Set of array elements when they are not alive, i.e. their memory can be
  /// used for other purposed. Can contain a nullptr; in this case the set is
  /// implicitly defined as the complement of #Occupied.
  IslPtr<isl_union_set> Unused;

  /// { [Element[] -> Scatter[]] }
  /// The write actions currently in the scop or that would be added when
  /// mapping a scalar.
  IslPtr<isl_union_set> Written;

  /// Check whether this Knowledge object is well-formed.
  void checkConsistency() const {
#ifndef NDEBUG
    // Default-initialized object
    if (!Occupied && !Unused && !Written)
      return;

    assert(Occupied || Unused);
    assert(Written);

    // If not all fields are defined, we cannot derived the universe.
    if (!Occupied || !Unused)
      return;

    assert(isl_union_set_is_disjoint(Occupied.keep(), Unused.keep()) ==
           isl_bool_true);
    auto Universe = give(isl_union_set_union(Occupied.copy(), Unused.copy()));
    assert(isl_union_set_is_subset(Written.keep(), Universe.keep()) ==
           isl_bool_true);
#endif
  }

public:
  /// Initialize a nullptr-Knowledge. This is only provided for convenience; do
  /// not use such an object.
  Knowledge() {}

  /// Create a new object with the given members.
  Knowledge(IslPtr<isl_union_set> Occupied, IslPtr<isl_union_set> Unused,
            IslPtr<isl_union_set> Written)
      : Occupied(std::move(Occupied)), Unused(std::move(Unused)),
        Written(std::move(Written)) {
    checkConsistency();
  }

  /// Alternative constructor taking isl_sets instead isl_union_sets.
  Knowledge(IslPtr<isl_set> Occupied, IslPtr<isl_set> Unused,
            IslPtr<isl_set> Written)
      : Knowledge(give(isl_union_set_from_set(Occupied.take())),
                  give(isl_union_set_from_set(Unused.take())),
                  give(isl_union_set_from_set(Written.take()))) {}

  /// Return whether this object was not default-constructed.
  bool isUsable() const { return (Occupied || Unused) && Written; }

  /// Print the content of this object to @p OS.
  void print(llvm::raw_ostream &OS, unsigned Indent = 0) const {
    if (isUsable()) {
      if (Occupied)
        OS.indent(Indent) << "Occupied: " << Occupied << "\n";
      else
        OS.indent(Indent) << "Occupied: <Everything else not in Unused>\n";
      if (Unused)
        OS.indent(Indent) << "Unused:   " << Unused << "\n";
      else
        OS.indent(Indent) << "Unused:   <Everything else not in Occupied>\n";
      OS.indent(Indent) << "Written : " << Written << '\n';
    } else {
      OS.indent(Indent) << "Invalid knowledge\n";
    }
  }

  /// Combine two knowledges, this and @p That.
  void learnFrom(Knowledge That) {
    assert(!isConflicting(*this, That));
    assert(Unused && That.Occupied);
    assert(
        !That.Unused &&
        "This function is only prepared to learn occupied elements from That");
    assert(!Occupied && "This function does not implement "
                        "`this->Occupied = "
                        "give(isl_union_set_union(this->Occupied.take(), "
                        "That.Occupied.copy()));`");

    Unused = give(isl_union_set_subtract(Unused.take(), That.Occupied.copy()));
    Written = give(isl_union_set_union(Written.take(), That.Written.take()));

    checkConsistency();
  }

  /// Determine whether two Knowledges conflict with each other.
  ///
  /// In theory @p Existing and @p Proposed are symmetric, but the
  /// implementation is constrained by the implicit interpretation. That is, @p
  /// Existing must have #Unused defined (use case 1) and @p Proposed must have
  /// #Occupied defined (use case 1).
  ///
  /// A conflict is defined as non-preserved semantics when they are merged. For
  /// instance, when for the same array and zone they assume different
  /// llvm::Values.
  ///
  /// @param Existing One of the knowledges with #Unused defined.
  /// @param Proposed One of the knowledges with #Occupied defined.
  /// @param OS       Dump the conflict reason to this output stream; use
  ///                 nullptr to not output anything.
  /// @param Indent   Indention for the conflict reason.
  ///
  /// @return True, iff the two knowledges are conflicting.
  static bool isConflicting(const Knowledge &Existing,
                            const Knowledge &Proposed,
                            llvm::raw_ostream *OS = nullptr,
                            unsigned Indent = 0) {
    assert(Existing.Unused);
    assert(Proposed.Occupied);

#ifndef NDEBUG
    if (Existing.Occupied && Proposed.Unused) {
      auto ExistingUniverse = give(isl_union_set_union(Existing.Occupied.copy(),
                                                       Existing.Unused.copy()));
      auto ProposedUniverse = give(isl_union_set_union(Proposed.Occupied.copy(),
                                                       Proposed.Unused.copy()));
      assert(isl_union_set_is_equal(ExistingUniverse.keep(),
                                    ProposedUniverse.keep()) == isl_bool_true &&
             "Both inputs' Knowledges must be over the same universe");
    }
#endif

    // Are the new lifetimes required for Proposed unused in Existing?
    if (isl_union_set_is_subset(Proposed.Occupied.keep(),
                                Existing.Unused.keep()) != isl_bool_true) {
      if (OS) {
        auto ConflictingLifetimes = give(isl_union_set_subtract(
            Proposed.Occupied.copy(), Existing.Unused.copy()));
        OS->indent(Indent) << "Proposed lifetimes are not unused in existing\n";
        OS->indent(Indent) << "Conflicting lifetimes: " << ConflictingLifetimes
                           << "\n";
      }
      return true;
    }

    // Do the writes in Existing only overwrite unused values in Proposed?
    // We convert here the set of lifetimes to actual timepoints. A lifetime is
    // in conflict with a set of write timepoints, if either a live timepoint is
    // clearly within the lifetime or if a write happens at the beginning of the
    // lifetime (where it would conflict with the value that actually writes the
    // value alive). There is no conflict at the end of a lifetime, as the alive
    // value will always be read, before it is overwritten again. The last
    // property holds in Polly for all scalar values and we expect all users of
    // Knowledge to check this property also for accesses to MemoryKind::Array.
    auto ProposedFixedDefs =
        convertZoneToTimepoints(Proposed.Occupied, true, false);
    if (isl_union_set_is_disjoint(Existing.Written.keep(),
                                  ProposedFixedDefs.keep()) != isl_bool_true) {
      if (OS) {
        auto ConflictingWrites = give(isl_union_set_intersect(
            Existing.Written.copy(), ProposedFixedDefs.copy()));
        OS->indent(Indent) << "Proposed writes into range used by existing\n";
        OS->indent(Indent) << "Conflicting writes: " << ConflictingWrites
                           << "\n";
      }
      return true;
    }

    // Do the new writes in Proposed only overwrite unused values in Existing?
    auto ExistingAvailableDefs =
        convertZoneToTimepoints(Existing.Unused, true, false);
    if (isl_union_set_is_subset(Proposed.Written.keep(),
                                ExistingAvailableDefs.keep()) !=
        isl_bool_true) {
      if (OS) {
        auto ConflictingWrites = give(isl_union_set_subtract(
            Proposed.Written.copy(), ExistingAvailableDefs.copy()));
        OS->indent(Indent)
            << "Proposed a lifetime where there is an Existing write into it\n";
        OS->indent(Indent) << "Conflicting writes: " << ConflictingWrites
                           << "\n";
      }
      return true;
    }

    // Does Proposed write at the same time as Existing already does (order of
    // writes is undefined)?
    if (isl_union_set_is_disjoint(Existing.Written.keep(),
                                  Proposed.Written.keep()) != isl_bool_true) {
      if (OS) {
        auto ConflictingWrites = give(isl_union_set_intersect(
            Existing.Written.copy(), Proposed.Written.copy()));
        OS->indent(Indent) << "Proposed writes at the same time as an already "
                              "Existing write\n";
        OS->indent(Indent) << "Conflicting writes: " << ConflictingWrites
                           << "\n";
      }
      return true;
    }

    return false;
  }
};

class DeLICM : public ScopPass {
private:
  DeLICM(const DeLICM &) = delete;
  const DeLICM &operator=(const DeLICM &) = delete;

public:
  static char ID;
  explicit DeLICM() : ScopPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitive<ScopInfoRegionPass>();
    AU.setPreservesAll();
  }

  virtual bool runOnScop(Scop &S) override {
    // Free resources for previous scop's computation, if not yet done.
    releaseMemory();

    // TODO: Run DeLICM algorithm

    return false;
  }

  virtual void printScop(raw_ostream &OS, Scop &S) const override {
    OS << "DeLICM result:\n";
    // TODO: Print analysis results and performed transformation details
  }

  virtual void releaseMemory() override {
    // TODO: Release resources (eg. shared_ptr to isl_ctx)
  }
};

char DeLICM::ID;
} // anonymous namespace

Pass *polly::createDeLICMPass() { return new DeLICM(); }

INITIALIZE_PASS_BEGIN(DeLICM, "polly-delicm", "Polly - DeLICM/DePRE", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass)
INITIALIZE_PASS_END(DeLICM, "polly-delicm", "Polly - DeLICM/DePRE", false,
                    false)

bool polly::isConflicting(IslPtr<isl_union_set> ExistingOccupied,
                          IslPtr<isl_union_set> ExistingUnused,
                          IslPtr<isl_union_set> ExistingWrites,
                          IslPtr<isl_union_set> ProposedOccupied,
                          IslPtr<isl_union_set> ProposedUnused,
                          IslPtr<isl_union_set> ProposedWrites,
                          llvm::raw_ostream *OS, unsigned Indent) {
  Knowledge Existing(std::move(ExistingOccupied), std::move(ExistingUnused),
                     std::move(ExistingWrites));
  Knowledge Proposed(std::move(ProposedOccupied), std::move(ProposedUnused),
                     std::move(ProposedWrites));

  return Knowledge::isConflicting(Existing, Proposed, OS, Indent);
}
