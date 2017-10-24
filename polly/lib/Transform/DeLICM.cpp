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
//===----------------------------------------------------------------------===//

#include "polly/DeLICM.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/ISLOStream.h"
#include "polly/Support/ISLTools.h"
#include "polly/ZoneAlgo.h"
#include "llvm/ADT/Statistic.h"
#define DEBUG_TYPE "polly-delicm"

using namespace polly;
using namespace llvm;

namespace {

cl::opt<int>
    DelicmMaxOps("polly-delicm-max-ops",
                 cl::desc("Maximum number of isl operations to invest for "
                          "lifetime analysis; 0=no limit"),
                 cl::init(1000000), cl::cat(PollyCategory));

cl::opt<bool> DelicmOverapproximateWrites(
    "polly-delicm-overapproximate-writes",
    cl::desc(
        "Do more PHI writes than necessary in order to avoid partial accesses"),
    cl::init(false), cl::Hidden, cl::cat(PollyCategory));

cl::opt<bool> DelicmPartialWrites("polly-delicm-partial-writes",
                                  cl::desc("Allow partial writes"),
                                  cl::init(true), cl::Hidden,
                                  cl::cat(PollyCategory));

cl::opt<bool>
    DelicmComputeKnown("polly-delicm-compute-known",
                       cl::desc("Compute known content of array elements"),
                       cl::init(true), cl::Hidden, cl::cat(PollyCategory));

STATISTIC(DeLICMAnalyzed, "Number of successfully analyzed SCoPs");
STATISTIC(DeLICMOutOfQuota,
          "Analyses aborted because max_operations was reached");
STATISTIC(MappedValueScalars, "Number of mapped Value scalars");
STATISTIC(MappedPHIScalars, "Number of mapped PHI scalars");
STATISTIC(TargetsMapped, "Number of stores used for at least one mapping");
STATISTIC(DeLICMScopsModified, "Number of SCoPs optimized");

STATISTIC(NumValueWrites, "Number of scalar value writes after DeLICM");
STATISTIC(NumValueWritesInLoops,
          "Number of scalar value writes nested in affine loops after DeLICM");
STATISTIC(NumPHIWrites, "Number of scalar phi writes after DeLICM");
STATISTIC(NumPHIWritesInLoops,
          "Number of scalar phi writes nested in affine loops after DeLICM");
STATISTIC(NumSingletonWrites, "Number of singleton writes after DeLICM");
STATISTIC(NumSingletonWritesInLoops,
          "Number of singleton writes nested in affine loops after DeLICM");

isl::union_map computeReachingOverwrite(isl::union_map Schedule,
                                        isl::union_map Writes,
                                        bool InclPrevWrite,
                                        bool InclOverwrite) {
  return computeReachingWrite(Schedule, Writes, true, InclPrevWrite,
                              InclOverwrite);
}

/// Compute the next overwrite for a scalar.
///
/// @param Schedule      { DomainWrite[] -> Scatter[] }
///                      Schedule of (at least) all writes. Instances not in @p
///                      Writes are ignored.
/// @param Writes        { DomainWrite[] }
///                      The element instances that write to the scalar.
/// @param InclPrevWrite Whether to extend the timepoints to include
///                      the timepoint where the previous write happens.
/// @param InclOverwrite Whether the reaching overwrite includes the timepoint
///                      of the overwrite itself.
///
/// @return { Scatter[] -> DomainDef[] }
isl::union_map computeScalarReachingOverwrite(isl::union_map Schedule,
                                              isl::union_set Writes,
                                              bool InclPrevWrite,
                                              bool InclOverwrite) {

  // { DomainWrite[] }
  auto WritesMap = give(isl_union_map_from_domain(Writes.take()));

  // { [Element[] -> Scatter[]] -> DomainWrite[] }
  auto Result = computeReachingOverwrite(
      std::move(Schedule), std::move(WritesMap), InclPrevWrite, InclOverwrite);

  return give(isl_union_map_domain_factor_range(Result.take()));
}

/// Overload of computeScalarReachingOverwrite, with only one writing statement.
/// Consequently, the result consists of only one map space.
///
/// @param Schedule      { DomainWrite[] -> Scatter[] }
/// @param Writes        { DomainWrite[] }
/// @param InclPrevWrite Include the previous write to result.
/// @param InclOverwrite Include the overwrite to the result.
///
/// @return { Scatter[] -> DomainWrite[] }
isl::map computeScalarReachingOverwrite(isl::union_map Schedule,
                                        isl::set Writes, bool InclPrevWrite,
                                        bool InclOverwrite) {
  isl::space ScatterSpace = getScatterSpace(Schedule);
  isl::space DomSpace = Writes.get_space();

  isl::union_map ReachOverwrite = computeScalarReachingOverwrite(
      Schedule, isl::union_set(Writes), InclPrevWrite, InclOverwrite);

  isl::space ResultSpace = ScatterSpace.map_from_domain_and_range(DomSpace);
  return singleton(std::move(ReachOverwrite), ResultSpace);
}

/// Try to find a 'natural' extension of a mapped to elements outside its
/// domain.
///
/// @param Relevant The map with mapping that may not be modified.
/// @param Universe The domain to which @p Relevant needs to be extended.
///
/// @return A map with that associates the domain elements of @p Relevant to the
///         same elements and in addition the elements of @p Universe to some
///         undefined elements. The function prefers to return simple maps.
isl::union_map expandMapping(isl::union_map Relevant, isl::union_set Universe) {
  Relevant = Relevant.coalesce();
  isl::union_set RelevantDomain = Relevant.domain();
  isl::union_map Simplified = Relevant.gist_domain(RelevantDomain);
  Simplified = Simplified.coalesce();
  return Simplified.intersect_domain(Universe);
}

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
  isl::union_set Occupied;

  /// { [Element[] -> Zone[]] }
  /// Set of array elements when they are not alive, i.e. their memory can be
  /// used for other purposed. Can contain a nullptr; in this case the set is
  /// implicitly defined as the complement of #Occupied.
  isl::union_set Unused;

  /// { [Element[] -> Zone[]] -> ValInst[] }
  /// Maps to the known content for each array element at any interval.
  ///
  /// Any element/interval can map to multiple known elements. This is due to
  /// multiple llvm::Value referring to the same content. Examples are
  ///
  /// - A value stored and loaded again. The LoadInst represents the same value
  /// as the StoreInst's value operand.
  ///
  /// - A PHINode is equal to any one of the incoming values. In case of
  /// LCSSA-form, it is always equal to its single incoming value.
  ///
  /// Two Knowledges are considered not conflicting if at least one of the known
  /// values match. Not known values are not stored as an unnamed tuple (as
  /// #Written does), but maps to nothing.
  ///
  ///  Known values are usually just defined for #Occupied elements. Knowing
  ///  #Unused contents has no advantage as it can be overwritten.
  isl::union_map Known;

  /// { [Element[] -> Scatter[]] -> ValInst[] }
  /// The write actions currently in the scop or that would be added when
  /// mapping a scalar. Maps to the value that is written.
  ///
  /// Written values that cannot be identified are represented by an unknown
  /// ValInst[] (an unnamed tuple of 0 dimension). It conflicts with itself.
  isl::union_map Written;

  /// Check whether this Knowledge object is well-formed.
  void checkConsistency() const {
#ifndef NDEBUG
    // Default-initialized object
    if (!Occupied && !Unused && !Known && !Written)
      return;

    assert(Occupied || Unused);
    assert(Known);
    assert(Written);

    // If not all fields are defined, we cannot derived the universe.
    if (!Occupied || !Unused)
      return;

    assert(isl_union_set_is_disjoint(Occupied.keep(), Unused.keep()) ==
           isl_bool_true);
    auto Universe = give(isl_union_set_union(Occupied.copy(), Unused.copy()));

    assert(!Known.domain().is_subset(Universe).is_false());
    assert(!Written.domain().is_subset(Universe).is_false());
#endif
  }

public:
  /// Initialize a nullptr-Knowledge. This is only provided for convenience; do
  /// not use such an object.
  Knowledge() {}

  /// Create a new object with the given members.
  Knowledge(isl::union_set Occupied, isl::union_set Unused,
            isl::union_map Known, isl::union_map Written)
      : Occupied(std::move(Occupied)), Unused(std::move(Unused)),
        Known(std::move(Known)), Written(std::move(Written)) {
    checkConsistency();
  }

  /// Return whether this object was not default-constructed.
  bool isUsable() const { return (Occupied || Unused) && Known && Written; }

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
      OS.indent(Indent) << "Known:    " << Known << "\n";
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
    Known = give(isl_union_map_union(Known.take(), That.Known.copy()));
    Written = give(isl_union_map_union(Written.take(), That.Written.take()));

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

    // Do the Existing and Proposed lifetimes conflict?
    //
    // Lifetimes are described as the cross-product of array elements and zone
    // intervals in which they are alive (the space { [Element[] -> Zone[]] }).
    // In the following we call this "element/lifetime interval".
    //
    // In order to not conflict, one of the following conditions must apply for
    // each element/lifetime interval:
    //
    // 1. If occupied in one of the knowledges, it is unused in the other.
    //
    //   - or -
    //
    // 2. Both contain the same value.
    //
    // Instead of partitioning the element/lifetime intervals into a part that
    // both Knowledges occupy (which requires an expensive subtraction) and for
    // these to check whether they are known to be the same value, we check only
    // the second condition and ensure that it also applies when then first
    // condition is true. This is done by adding a wildcard value to
    // Proposed.Known and Existing.Unused such that they match as a common known
    // value. We use the "unknown ValInst" for this purpose. Every
    // Existing.Unused may match with an unknown Proposed.Occupied because these
    // never are in conflict with each other.
    auto ProposedOccupiedAnyVal = makeUnknownForDomain(Proposed.Occupied);
    auto ProposedValues = Proposed.Known.unite(ProposedOccupiedAnyVal);

    auto ExistingUnusedAnyVal = makeUnknownForDomain(Existing.Unused);
    auto ExistingValues = Existing.Known.unite(ExistingUnusedAnyVal);

    auto MatchingVals = ExistingValues.intersect(ProposedValues);
    auto Matches = MatchingVals.domain();

    // Any Proposed.Occupied must either have a match between the known values
    // of Existing and Occupied, or be in Existing.Unused. In the latter case,
    // the previously added "AnyVal" will match each other.
    if (!Proposed.Occupied.is_subset(Matches)) {
      if (OS) {
        auto Conflicting = Proposed.Occupied.subtract(Matches);
        auto ExistingConflictingKnown =
            Existing.Known.intersect_domain(Conflicting);
        auto ProposedConflictingKnown =
            Proposed.Known.intersect_domain(Conflicting);

        OS->indent(Indent) << "Proposed lifetime conflicting with Existing's\n";
        OS->indent(Indent) << "Conflicting occupied: " << Conflicting << "\n";
        if (!ExistingConflictingKnown.is_empty())
          OS->indent(Indent)
              << "Existing Known:       " << ExistingConflictingKnown << "\n";
        if (!ProposedConflictingKnown.is_empty())
          OS->indent(Indent)
              << "Proposed Known:       " << ProposedConflictingKnown << "\n";
      }
      return true;
    }

    // Do the writes in Existing conflict with occupied values in Proposed?
    //
    // In order to not conflict, it must either write to unused lifetime or
    // write the same value. To check, we remove the writes that write into
    // Proposed.Unused (they never conflict) and then see whether the written
    // value is already in Proposed.Known. If there are multiple known values
    // and a written value is known under different names, it is enough when one
    // of the written values (assuming that they are the same value under
    // different names, e.g. a PHINode and one of the incoming values) matches
    // one of the known names.
    //
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
    auto ProposedFixedKnown =
        convertZoneToTimepoints(Proposed.Known, isl::dim::in, true, false);

    auto ExistingConflictingWrites =
        Existing.Written.intersect_domain(ProposedFixedDefs);
    auto ExistingConflictingWritesDomain = ExistingConflictingWrites.domain();

    auto CommonWrittenVal =
        ProposedFixedKnown.intersect(ExistingConflictingWrites);
    auto CommonWrittenValDomain = CommonWrittenVal.domain();

    if (!ExistingConflictingWritesDomain.is_subset(CommonWrittenValDomain)) {
      if (OS) {
        auto ExistingConflictingWritten =
            ExistingConflictingWrites.subtract_domain(CommonWrittenValDomain);
        auto ProposedConflictingKnown = ProposedFixedKnown.subtract_domain(
            ExistingConflictingWritten.domain());

        OS->indent(Indent)
            << "Proposed a lifetime where there is an Existing write into it\n";
        OS->indent(Indent) << "Existing conflicting writes: "
                           << ExistingConflictingWritten << "\n";
        if (!ProposedConflictingKnown.is_empty())
          OS->indent(Indent)
              << "Proposed conflicting known:  " << ProposedConflictingKnown
              << "\n";
      }
      return true;
    }

    // Do the writes in Proposed conflict with occupied values in Existing?
    auto ExistingAvailableDefs =
        convertZoneToTimepoints(Existing.Unused, true, false);
    auto ExistingKnownDefs =
        convertZoneToTimepoints(Existing.Known, isl::dim::in, true, false);

    auto ProposedWrittenDomain = Proposed.Written.domain();
    auto KnownIdentical = ExistingKnownDefs.intersect(Proposed.Written);
    auto IdenticalOrUnused =
        ExistingAvailableDefs.unite(KnownIdentical.domain());
    if (!ProposedWrittenDomain.is_subset(IdenticalOrUnused)) {
      if (OS) {
        auto Conflicting = ProposedWrittenDomain.subtract(IdenticalOrUnused);
        auto ExistingConflictingKnown =
            ExistingKnownDefs.intersect_domain(Conflicting);
        auto ProposedConflictingWritten =
            Proposed.Written.intersect_domain(Conflicting);

        OS->indent(Indent) << "Proposed writes into range used by Existing\n";
        OS->indent(Indent) << "Proposed conflicting writes: "
                           << ProposedConflictingWritten << "\n";
        if (!ExistingConflictingKnown.is_empty())
          OS->indent(Indent)
              << "Existing conflicting known: " << ExistingConflictingKnown
              << "\n";
      }
      return true;
    }

    // Does Proposed write at the same time as Existing already does (order of
    // writes is undefined)? Writing the same value is permitted.
    auto ExistingWrittenDomain =
        isl::manage(isl_union_map_domain(Existing.Written.copy()));
    auto BothWritten =
        Existing.Written.domain().intersect(Proposed.Written.domain());
    auto ExistingKnownWritten = filterKnownValInst(Existing.Written);
    auto ProposedKnownWritten = filterKnownValInst(Proposed.Written);
    auto CommonWritten =
        ExistingKnownWritten.intersect(ProposedKnownWritten).domain();

    if (!BothWritten.is_subset(CommonWritten)) {
      if (OS) {
        auto Conflicting = BothWritten.subtract(CommonWritten);
        auto ExistingConflictingWritten =
            Existing.Written.intersect_domain(Conflicting);
        auto ProposedConflictingWritten =
            Proposed.Written.intersect_domain(Conflicting);

        OS->indent(Indent) << "Proposed writes at the same time as an already "
                              "Existing write\n";
        OS->indent(Indent) << "Conflicting writes: " << Conflicting << "\n";
        if (!ExistingConflictingWritten.is_empty())
          OS->indent(Indent)
              << "Exiting write:      " << ExistingConflictingWritten << "\n";
        if (!ProposedConflictingWritten.is_empty())
          OS->indent(Indent)
              << "Proposed write:     " << ProposedConflictingWritten << "\n";
      }
      return true;
    }

    return false;
  }
};

/// Implementation of the DeLICM/DePRE transformation.
class DeLICMImpl : public ZoneAlgorithm {
private:
  /// Knowledge before any transformation took place.
  Knowledge OriginalZone;

  /// Current knowledge of the SCoP including all already applied
  /// transformations.
  Knowledge Zone;

  /// Number of StoreInsts something can be mapped to.
  int NumberOfCompatibleTargets = 0;

  /// The number of StoreInsts to which at least one value or PHI has been
  /// mapped to.
  int NumberOfTargetsMapped = 0;

  /// The number of llvm::Value mapped to some array element.
  int NumberOfMappedValueScalars = 0;

  /// The number of PHIs mapped to some array element.
  int NumberOfMappedPHIScalars = 0;

  /// Determine whether two knowledges are conflicting with each other.
  ///
  /// @see Knowledge::isConflicting
  bool isConflicting(const Knowledge &Proposed) {
    raw_ostream *OS = nullptr;
    DEBUG(OS = &llvm::dbgs());
    return Knowledge::isConflicting(Zone, Proposed, OS, 4);
  }

  /// Determine whether @p SAI is a scalar that can be mapped to an array
  /// element.
  bool isMappable(const ScopArrayInfo *SAI) {
    assert(SAI);

    if (SAI->isValueKind()) {
      auto *MA = S->getValueDef(SAI);
      if (!MA) {
        DEBUG(dbgs()
              << "    Reject because value is read-only within the scop\n");
        return false;
      }

      // Mapping if value is used after scop is not supported. The code
      // generator would need to reload the scalar after the scop, but it
      // does not have the information to where it is mapped to. Only the
      // MemoryAccesses have that information, not the ScopArrayInfo.
      auto Inst = MA->getAccessInstruction();
      for (auto User : Inst->users()) {
        if (!isa<Instruction>(User))
          return false;
        auto UserInst = cast<Instruction>(User);

        if (!S->contains(UserInst)) {
          DEBUG(dbgs() << "    Reject because value is escaping\n");
          return false;
        }
      }

      return true;
    }

    if (SAI->isPHIKind()) {
      auto *MA = S->getPHIRead(SAI);
      assert(MA);

      // Mapping of an incoming block from before the SCoP is not supported by
      // the code generator.
      auto PHI = cast<PHINode>(MA->getAccessInstruction());
      for (auto Incoming : PHI->blocks()) {
        if (!S->contains(Incoming)) {
          DEBUG(dbgs() << "    Reject because at least one incoming block is "
                          "not in the scop region\n");
          return false;
        }
      }

      return true;
    }

    DEBUG(dbgs() << "    Reject ExitPHI or other non-value\n");
    return false;
  }

  /// Compute the uses of a MemoryKind::Value and its lifetime (from its
  /// definition to the last use).
  ///
  /// @param SAI The ScopArrayInfo representing the value's storage.
  ///
  /// @return { DomainDef[] -> DomainUse[] }, { DomainDef[] -> Zone[] }
  ///         First element is the set of uses for each definition.
  ///         The second is the lifetime of each definition.
  std::tuple<isl::union_map, isl::map>
  computeValueUses(const ScopArrayInfo *SAI) {
    assert(SAI->isValueKind());

    // { DomainRead[] }
    auto Reads = makeEmptyUnionSet();

    // Find all uses.
    for (auto *MA : S->getValueUses(SAI))
      Reads =
          give(isl_union_set_add_set(Reads.take(), getDomainFor(MA).take()));

    // { DomainRead[] -> Scatter[] }
    auto ReadSchedule = getScatterFor(Reads);

    auto *DefMA = S->getValueDef(SAI);
    assert(DefMA);

    // { DomainDef[] }
    auto Writes = getDomainFor(DefMA);

    // { DomainDef[] -> Scatter[] }
    auto WriteScatter = getScatterFor(Writes);

    // { Scatter[] -> DomainDef[] }
    auto ReachDef = getScalarReachingDefinition(DefMA->getStatement());

    // { [DomainDef[] -> Scatter[]] -> DomainUse[] }
    auto Uses = give(
        isl_union_map_apply_range(isl_union_map_from_map(isl_map_range_map(
                                      isl_map_reverse(ReachDef.take()))),
                                  isl_union_map_reverse(ReadSchedule.take())));

    // { DomainDef[] -> Scatter[] }
    auto UseScatter =
        singleton(give(isl_union_set_unwrap(isl_union_map_domain(Uses.copy()))),
                  give(isl_space_map_from_domain_and_range(
                      isl_set_get_space(Writes.keep()), ScatterSpace.copy())));

    // { DomainDef[] -> Zone[] }
    auto Lifetime = betweenScatter(WriteScatter, UseScatter, false, true);

    // { DomainDef[] -> DomainRead[] }
    auto DefUses = give(isl_union_map_domain_factor_domain(Uses.take()));

    return std::make_pair(DefUses, Lifetime);
  }

  /// For each 'execution' of a PHINode, get the incoming block that was
  /// executed before.
  ///
  /// For each PHI instance we can directly determine which was the incoming
  /// block, and hence derive which value the PHI has.
  ///
  /// @param SAI The ScopArrayInfo representing the PHI's storage.
  ///
  /// @return { DomainPHIRead[] -> DomainPHIWrite[] }
  isl::union_map computePerPHI(const ScopArrayInfo *SAI) {
    assert(SAI->isPHIKind());

    // { DomainPHIWrite[] -> Scatter[] }
    auto PHIWriteScatter = makeEmptyUnionMap();

    // Collect all incoming block timepoint.
    for (auto *MA : S->getPHIIncomings(SAI)) {
      auto Scatter = getScatterFor(MA);
      PHIWriteScatter =
          give(isl_union_map_add_map(PHIWriteScatter.take(), Scatter.take()));
    }

    // { DomainPHIRead[] -> Scatter[] }
    auto PHIReadScatter = getScatterFor(S->getPHIRead(SAI));

    // { DomainPHIRead[] -> Scatter[] }
    auto BeforeRead = beforeScatter(PHIReadScatter, true);

    // { Scatter[] }
    auto WriteTimes = singleton(
        give(isl_union_map_range(PHIWriteScatter.copy())), ScatterSpace);

    // { DomainPHIRead[] -> Scatter[] }
    auto PHIWriteTimes =
        give(isl_map_intersect_range(BeforeRead.take(), WriteTimes.take()));
    auto LastPerPHIWrites = give(isl_map_lexmax(PHIWriteTimes.take()));

    // { DomainPHIRead[] -> DomainPHIWrite[] }
    auto Result = give(isl_union_map_apply_range(
        isl_union_map_from_map(LastPerPHIWrites.take()),
        isl_union_map_reverse(PHIWriteScatter.take())));
    assert(isl_union_map_is_single_valued(Result.keep()) == isl_bool_true);
    assert(isl_union_map_is_injective(Result.keep()) == isl_bool_true);
    return Result;
  }

  /// Try to map a MemoryKind::Value to a given array element.
  ///
  /// @param SAI       Representation of the scalar's memory to map.
  /// @param TargetElt { Scatter[] -> Element[] }
  ///                  Suggestion where to map a scalar to when at a timepoint.
  ///
  /// @return true if the scalar was successfully mapped.
  bool tryMapValue(const ScopArrayInfo *SAI, isl::map TargetElt) {
    assert(SAI->isValueKind());

    auto *DefMA = S->getValueDef(SAI);
    assert(DefMA->isValueKind());
    assert(DefMA->isMustWrite());
    auto *V = DefMA->getAccessValue();
    auto *DefInst = DefMA->getAccessInstruction();

    // Stop if the scalar has already been mapped.
    if (!DefMA->getLatestScopArrayInfo()->isValueKind())
      return false;

    // { DomainDef[] -> Scatter[] }
    auto DefSched = getScatterFor(DefMA);

    // Where each write is mapped to, according to the suggestion.
    // { DomainDef[] -> Element[] }
    auto DefTarget = give(isl_map_apply_domain(
        TargetElt.copy(), isl_map_reverse(DefSched.copy())));
    simplify(DefTarget);
    DEBUG(dbgs() << "    Def Mapping: " << DefTarget << '\n');

    auto OrigDomain = getDomainFor(DefMA);
    auto MappedDomain = give(isl_map_domain(DefTarget.copy()));
    if (!isl_set_is_subset(OrigDomain.keep(), MappedDomain.keep())) {
      DEBUG(dbgs()
            << "    Reject because mapping does not encompass all instances\n");
      return false;
    }

    // { DomainDef[] -> Zone[] }
    isl::map Lifetime;

    // { DomainDef[] -> DomainUse[] }
    isl::union_map DefUses;

    std::tie(DefUses, Lifetime) = computeValueUses(SAI);
    DEBUG(dbgs() << "    Lifetime: " << Lifetime << '\n');

    /// { [Element[] -> Zone[]] }
    auto EltZone = give(
        isl_map_wrap(isl_map_apply_domain(Lifetime.copy(), DefTarget.copy())));
    simplify(EltZone);

    // When known knowledge is disabled, just return the unknown value. It will
    // either get filtered out or conflict with itself.
    // { DomainDef[] -> ValInst[] }
    isl::map ValInst;
    if (DelicmComputeKnown)
      ValInst = makeValInst(V, DefMA->getStatement(),
                            LI->getLoopFor(DefInst->getParent()));
    else
      ValInst = makeUnknownForDomain(DefMA->getStatement());

    // { DomainDef[] -> [Element[] -> Zone[]] }
    auto EltKnownTranslator =
        give(isl_map_range_product(DefTarget.copy(), Lifetime.copy()));

    // { [Element[] -> Zone[]] -> ValInst[] }
    auto EltKnown =
        give(isl_map_apply_domain(ValInst.copy(), EltKnownTranslator.take()));
    simplify(EltKnown);

    // { DomainDef[] -> [Element[] -> Scatter[]] }
    auto WrittenTranslator =
        give(isl_map_range_product(DefTarget.copy(), DefSched.take()));

    // { [Element[] -> Scatter[]] -> ValInst[] }
    auto DefEltSched =
        give(isl_map_apply_domain(ValInst.copy(), WrittenTranslator.take()));
    simplify(DefEltSched);

    Knowledge Proposed(EltZone, nullptr, filterKnownValInst(EltKnown),
                       DefEltSched);
    if (isConflicting(Proposed))
      return false;

    // { DomainUse[] -> Element[] }
    auto UseTarget = give(
        isl_union_map_apply_range(isl_union_map_reverse(DefUses.take()),
                                  isl_union_map_from_map(DefTarget.copy())));

    mapValue(SAI, std::move(DefTarget), std::move(UseTarget),
             std::move(Lifetime), std::move(Proposed));
    return true;
  }

  /// After a scalar has been mapped, update the global knowledge.
  void applyLifetime(Knowledge Proposed) {
    Zone.learnFrom(std::move(Proposed));
  }

  /// Map a MemoryKind::Value scalar to an array element.
  ///
  /// Callers must have ensured that the mapping is valid and not conflicting.
  ///
  /// @param SAI       The ScopArrayInfo representing the scalar's memory to
  ///                  map.
  /// @param DefTarget { DomainDef[] -> Element[] }
  ///                  The array element to map the scalar to.
  /// @param UseTarget { DomainUse[] -> Element[] }
  ///                  The array elements the uses are mapped to.
  /// @param Lifetime  { DomainDef[] -> Zone[] }
  ///                  The lifetime of each llvm::Value definition for
  ///                  reporting.
  /// @param Proposed  Mapping constraints for reporting.
  void mapValue(const ScopArrayInfo *SAI, isl::map DefTarget,
                isl::union_map UseTarget, isl::map Lifetime,
                Knowledge Proposed) {
    // Redirect the read accesses.
    for (auto *MA : S->getValueUses(SAI)) {
      // { DomainUse[] }
      auto Domain = getDomainFor(MA);

      // { DomainUse[] -> Element[] }
      auto NewAccRel = give(isl_union_map_intersect_domain(
          UseTarget.copy(), isl_union_set_from_set(Domain.take())));
      simplify(NewAccRel);

      assert(isl_union_map_n_map(NewAccRel.keep()) == 1);
      MA->setNewAccessRelation(isl::map::from_union_map(NewAccRel));
    }

    auto *WA = S->getValueDef(SAI);
    WA->setNewAccessRelation(DefTarget);
    applyLifetime(Proposed);

    MappedValueScalars++;
    NumberOfMappedValueScalars += 1;
  }

  isl::map makeValInst(Value *Val, ScopStmt *UserStmt, Loop *Scope,
                       bool IsCertain = true) {
    // When known knowledge is disabled, just return the unknown value. It will
    // either get filtered out or conflict with itself.
    if (!DelicmComputeKnown)
      return makeUnknownForDomain(UserStmt);
    return ZoneAlgorithm::makeValInst(Val, UserStmt, Scope, IsCertain);
  }

  /// Express the incoming values of a PHI for each incoming statement in an
  /// isl::union_map.
  ///
  /// @param SAI The PHI scalar represented by a ScopArrayInfo.
  ///
  /// @return { PHIWriteDomain[] -> ValInst[] }
  isl::union_map determinePHIWrittenValues(const ScopArrayInfo *SAI) {
    auto Result = makeEmptyUnionMap();

    // Collect the incoming values.
    for (auto *MA : S->getPHIIncomings(SAI)) {
      // { DomainWrite[] -> ValInst[] }
      isl::union_map ValInst;
      auto *WriteStmt = MA->getStatement();

      auto Incoming = MA->getIncoming();
      assert(!Incoming.empty());
      if (Incoming.size() == 1) {
        ValInst = makeValInst(Incoming[0].second, WriteStmt,
                              LI->getLoopFor(Incoming[0].first));
      } else {
        // If the PHI is in a subregion's exit node it can have multiple
        // incoming values (+ maybe another incoming edge from an unrelated
        // block). We cannot directly represent it as a single llvm::Value.
        // We currently model it as unknown value, but modeling as the PHIInst
        // itself could be OK, too.
        ValInst = makeUnknownForDomain(WriteStmt);
      }

      Result = give(isl_union_map_union(Result.take(), ValInst.take()));
    }

    assert(isl_union_map_is_single_valued(Result.keep()) == isl_bool_true &&
           "Cannot have multiple incoming values for same incoming statement");
    return Result;
  }

  /// Try to map a MemoryKind::PHI scalar to a given array element.
  ///
  /// @param SAI       Representation of the scalar's memory to map.
  /// @param TargetElt { Scatter[] -> Element[] }
  ///                  Suggestion where to map the scalar to when at a
  ///                  timepoint.
  ///
  /// @return true if the PHI scalar has been mapped.
  bool tryMapPHI(const ScopArrayInfo *SAI, isl::map TargetElt) {
    auto *PHIRead = S->getPHIRead(SAI);
    assert(PHIRead->isPHIKind());
    assert(PHIRead->isRead());

    // Skip if already been mapped.
    if (!PHIRead->getLatestScopArrayInfo()->isPHIKind())
      return false;

    // { DomainRead[] -> Scatter[] }
    auto PHISched = getScatterFor(PHIRead);

    // { DomainRead[] -> Element[] }
    auto PHITarget =
        give(isl_map_apply_range(PHISched.copy(), TargetElt.copy()));
    simplify(PHITarget);
    DEBUG(dbgs() << "    Mapping: " << PHITarget << '\n');

    auto OrigDomain = getDomainFor(PHIRead);
    auto MappedDomain = give(isl_map_domain(PHITarget.copy()));
    if (!isl_set_is_subset(OrigDomain.keep(), MappedDomain.keep())) {
      DEBUG(dbgs()
            << "    Reject because mapping does not encompass all instances\n");
      return false;
    }

    // { DomainRead[] -> DomainWrite[] }
    auto PerPHIWrites = computePerPHI(SAI);

    // { DomainWrite[] -> Element[] }
    auto WritesTarget = give(isl_union_map_reverse(isl_union_map_apply_domain(
        PerPHIWrites.copy(), isl_union_map_from_map(PHITarget.copy()))));
    simplify(WritesTarget);

    // { DomainWrite[] }
    auto UniverseWritesDom = give(isl_union_set_empty(ParamSpace.copy()));

    for (auto *MA : S->getPHIIncomings(SAI))
      UniverseWritesDom = give(isl_union_set_add_set(UniverseWritesDom.take(),
                                                     getDomainFor(MA).take()));

    auto RelevantWritesTarget = WritesTarget;
    if (DelicmOverapproximateWrites)
      WritesTarget = expandMapping(WritesTarget, UniverseWritesDom);

    auto ExpandedWritesDom = give(isl_union_map_domain(WritesTarget.copy()));
    if (!DelicmPartialWrites &&
        !isl_union_set_is_subset(UniverseWritesDom.keep(),
                                 ExpandedWritesDom.keep())) {
      DEBUG(dbgs() << "    Reject because did not find PHI write mapping for "
                      "all instances\n");
      if (DelicmOverapproximateWrites)
        DEBUG(dbgs() << "      Relevant Mapping:    " << RelevantWritesTarget
                     << '\n');
      DEBUG(dbgs() << "      Deduced Mapping:     " << WritesTarget << '\n');
      DEBUG(dbgs() << "      Missing instances:    "
                   << give(isl_union_set_subtract(UniverseWritesDom.copy(),
                                                  ExpandedWritesDom.copy()))
                   << '\n');
      return false;
    }

    //  { DomainRead[] -> Scatter[] }
    auto PerPHIWriteScatter = give(isl_map_from_union_map(
        isl_union_map_apply_range(PerPHIWrites.copy(), Schedule.copy())));

    // { DomainRead[] -> Zone[] }
    auto Lifetime = betweenScatter(PerPHIWriteScatter, PHISched, false, true);
    simplify(Lifetime);
    DEBUG(dbgs() << "    Lifetime: " << Lifetime << "\n");

    // { DomainWrite[] -> Zone[] }
    auto WriteLifetime = give(isl_union_map_apply_domain(
        isl_union_map_from_map(Lifetime.copy()), PerPHIWrites.copy()));

    // { DomainWrite[] -> ValInst[] }
    auto WrittenValue = determinePHIWrittenValues(SAI);

    // { DomainWrite[] -> [Element[] -> Scatter[]] }
    auto WrittenTranslator =
        give(isl_union_map_range_product(WritesTarget.copy(), Schedule.copy()));

    // { [Element[] -> Scatter[]] -> ValInst[] }
    auto Written = give(isl_union_map_apply_domain(WrittenValue.copy(),
                                                   WrittenTranslator.copy()));
    simplify(Written);

    // { DomainWrite[] -> [Element[] -> Zone[]] }
    auto LifetimeTranslator = give(
        isl_union_map_range_product(WritesTarget.copy(), WriteLifetime.copy()));

    // { DomainWrite[] -> ValInst[] }
    auto WrittenKnownValue = filterKnownValInst(WrittenValue);

    // { [Element[] -> Zone[]] -> ValInst[] }
    auto EltLifetimeInst = give(isl_union_map_apply_domain(
        WrittenKnownValue.copy(), LifetimeTranslator.copy()));
    simplify(EltLifetimeInst);

    // { [Element[] -> Zone[] }
    auto Occupied = give(isl_union_map_range(LifetimeTranslator.copy()));
    simplify(Occupied);

    Knowledge Proposed(Occupied, nullptr, EltLifetimeInst, Written);
    if (isConflicting(Proposed))
      return false;

    mapPHI(SAI, std::move(PHITarget), std::move(WritesTarget),
           std::move(Lifetime), std::move(Proposed));
    return true;
  }

  /// Map a MemoryKind::PHI scalar to an array element.
  ///
  /// Callers must have ensured that the mapping is valid and not conflicting
  /// with the common knowledge.
  ///
  /// @param SAI         The ScopArrayInfo representing the scalar's memory to
  ///                    map.
  /// @param ReadTarget  { DomainRead[] -> Element[] }
  ///                    The array element to map the scalar to.
  /// @param WriteTarget { DomainWrite[] -> Element[] }
  ///                    New access target for each PHI incoming write.
  /// @param Lifetime    { DomainRead[] -> Zone[] }
  ///                    The lifetime of each PHI for reporting.
  /// @param Proposed    Mapping constraints for reporting.
  void mapPHI(const ScopArrayInfo *SAI, isl::map ReadTarget,
              isl::union_map WriteTarget, isl::map Lifetime,
              Knowledge Proposed) {
    // { Element[] }
    isl::space ElementSpace = ReadTarget.get_space().range();

    // Redirect the PHI incoming writes.
    for (auto *MA : S->getPHIIncomings(SAI)) {
      // { DomainWrite[] }
      auto Domain = getDomainFor(MA);

      // { DomainWrite[] -> Element[] }
      auto NewAccRel = give(isl_union_map_intersect_domain(
          WriteTarget.copy(), isl_union_set_from_set(Domain.copy())));
      simplify(NewAccRel);

      isl::space NewAccRelSpace =
          Domain.get_space().map_from_domain_and_range(ElementSpace);
      isl::map NewAccRelMap = singleton(NewAccRel, NewAccRelSpace);
      MA->setNewAccessRelation(NewAccRelMap);
    }

    // Redirect the PHI read.
    auto *PHIRead = S->getPHIRead(SAI);
    PHIRead->setNewAccessRelation(ReadTarget);
    applyLifetime(Proposed);

    MappedPHIScalars++;
    NumberOfMappedPHIScalars++;
  }

  /// Search and map scalars to memory overwritten by @p TargetStoreMA.
  ///
  /// Start trying to map scalars that are used in the same statement as the
  /// store. For every successful mapping, try to also map scalars of the
  /// statements where those are written. Repeat, until no more mapping
  /// opportunity is found.
  ///
  /// There is currently no preference in which order scalars are tried.
  /// Ideally, we would direct it towards a load instruction of the same array
  /// element.
  bool collapseScalarsToStore(MemoryAccess *TargetStoreMA) {
    assert(TargetStoreMA->isLatestArrayKind());
    assert(TargetStoreMA->isMustWrite());

    auto TargetStmt = TargetStoreMA->getStatement();

    // { DomTarget[] }
    auto TargetDom = getDomainFor(TargetStmt);

    // { DomTarget[] -> Element[] }
    auto TargetAccRel = getAccessRelationFor(TargetStoreMA);

    // { Zone[] -> DomTarget[] }
    // For each point in time, find the next target store instance.
    auto Target =
        computeScalarReachingOverwrite(Schedule, TargetDom, false, true);

    // { Zone[] -> Element[] }
    // Use the target store's write location as a suggestion to map scalars to.
    auto EltTarget =
        give(isl_map_apply_range(Target.take(), TargetAccRel.take()));
    simplify(EltTarget);
    DEBUG(dbgs() << "    Target mapping is " << EltTarget << '\n');

    // Stack of elements not yet processed.
    SmallVector<MemoryAccess *, 16> Worklist;

    // Set of scalars already tested.
    SmallPtrSet<const ScopArrayInfo *, 16> Closed;

    // Lambda to add all scalar reads to the work list.
    auto ProcessAllIncoming = [&](ScopStmt *Stmt) {
      for (auto *MA : *Stmt) {
        if (!MA->isLatestScalarKind())
          continue;
        if (!MA->isRead())
          continue;

        Worklist.push_back(MA);
      }
    };

    auto *WrittenVal = TargetStoreMA->getAccessInstruction()->getOperand(0);
    if (auto *WrittenValInputMA = TargetStmt->lookupInputAccessOf(WrittenVal))
      Worklist.push_back(WrittenValInputMA);
    else
      ProcessAllIncoming(TargetStmt);

    auto AnyMapped = false;
    auto &DL = S->getRegion().getEntry()->getModule()->getDataLayout();
    auto StoreSize =
        DL.getTypeAllocSize(TargetStoreMA->getAccessValue()->getType());

    while (!Worklist.empty()) {
      auto *MA = Worklist.pop_back_val();

      auto *SAI = MA->getScopArrayInfo();
      if (Closed.count(SAI))
        continue;
      Closed.insert(SAI);
      DEBUG(dbgs() << "\n    Trying to map " << MA << " (SAI: " << SAI
                   << ")\n");

      // Skip non-mappable scalars.
      if (!isMappable(SAI))
        continue;

      auto MASize = DL.getTypeAllocSize(MA->getAccessValue()->getType());
      if (MASize > StoreSize) {
        DEBUG(dbgs() << "    Reject because storage size is insufficient\n");
        continue;
      }

      // Try to map MemoryKind::Value scalars.
      if (SAI->isValueKind()) {
        if (!tryMapValue(SAI, EltTarget))
          continue;

        auto *DefAcc = S->getValueDef(SAI);
        ProcessAllIncoming(DefAcc->getStatement());

        AnyMapped = true;
        continue;
      }

      // Try to map MemoryKind::PHI scalars.
      if (SAI->isPHIKind()) {
        if (!tryMapPHI(SAI, EltTarget))
          continue;
        // Add inputs of all incoming statements to the worklist. Prefer the
        // input accesses of the incoming blocks.
        for (auto *PHIWrite : S->getPHIIncomings(SAI)) {
          auto *PHIWriteStmt = PHIWrite->getStatement();
          bool FoundAny = false;
          for (auto Incoming : PHIWrite->getIncoming()) {
            auto *IncomingInputMA =
                PHIWriteStmt->lookupInputAccessOf(Incoming.second);
            if (!IncomingInputMA)
              continue;

            Worklist.push_back(IncomingInputMA);
            FoundAny = true;
          }

          if (!FoundAny)
            ProcessAllIncoming(PHIWrite->getStatement());
        }

        AnyMapped = true;
        continue;
      }
    }

    if (AnyMapped) {
      TargetsMapped++;
      NumberOfTargetsMapped++;
    }
    return AnyMapped;
  }

  /// Compute when an array element is unused.
  ///
  /// @return { [Element[] -> Zone[]] }
  isl::union_set computeLifetime() const {
    // { Element[] -> Zone[] }
    auto ArrayUnused = computeArrayUnused(Schedule, AllMustWrites, AllReads,
                                          false, false, true);

    auto Result = give(isl_union_map_wrap(ArrayUnused.copy()));

    simplify(Result);
    return Result;
  }

  /// Determine when an array element is written to, and which value instance is
  /// written.
  ///
  /// @return { [Element[] -> Scatter[]] -> ValInst[] }
  isl::union_map computeWritten() const {
    // { [Element[] -> Scatter[]] -> ValInst[] }
    auto EltWritten = applyDomainRange(AllWriteValInst, Schedule);

    simplify(EltWritten);
    return EltWritten;
  }

  /// Determine whether an access touches at most one element.
  ///
  /// The accessed element could be a scalar or accessing an array with constant
  /// subscript, such that all instances access only that element.
  ///
  /// @param MA The access to test.
  ///
  /// @return True, if zero or one elements are accessed; False if at least two
  ///         different elements are accessed.
  bool isScalarAccess(MemoryAccess *MA) {
    auto Map = getAccessRelationFor(MA);
    auto Set = give(isl_map_range(Map.take()));
    return isl_set_is_singleton(Set.keep()) == isl_bool_true;
  }

  /// Print mapping statistics to @p OS.
  void printStatistics(llvm::raw_ostream &OS, int Indent = 0) const {
    OS.indent(Indent) << "Statistics {\n";
    OS.indent(Indent + 4) << "Compatible overwrites: "
                          << NumberOfCompatibleTargets << "\n";
    OS.indent(Indent + 4) << "Overwrites mapped to:  " << NumberOfTargetsMapped
                          << '\n';
    OS.indent(Indent + 4) << "Value scalars mapped:  "
                          << NumberOfMappedValueScalars << '\n';
    OS.indent(Indent + 4) << "PHI scalars mapped:    "
                          << NumberOfMappedPHIScalars << '\n';
    OS.indent(Indent) << "}\n";
  }

  /// Return whether at least one transformation been applied.
  bool isModified() const { return NumberOfTargetsMapped > 0; }

public:
  DeLICMImpl(Scop *S, LoopInfo *LI) : ZoneAlgorithm("polly-delicm", S, LI) {}

  /// Calculate the lifetime (definition to last use) of every array element.
  ///
  /// @return True if the computed lifetimes (#Zone) is usable.
  bool computeZone() {
    // Check that nothing strange occurs.
    collectCompatibleElts();

    isl::union_set EltUnused;
    isl::union_map EltKnown, EltWritten;

    {
      IslMaxOperationsGuard MaxOpGuard(IslCtx.get(), DelicmMaxOps);

      computeCommon();

      EltUnused = computeLifetime();
      EltKnown = computeKnown(true, false);
      EltWritten = computeWritten();
    }
    DeLICMAnalyzed++;

    if (!EltUnused || !EltKnown || !EltWritten) {
      assert(isl_ctx_last_error(IslCtx.get()) == isl_error_quota &&
             "The only reason that these things have not been computed should "
             "be if the max-operations limit hit");
      DeLICMOutOfQuota++;
      DEBUG(dbgs() << "DeLICM analysis exceeded max_operations\n");
      DebugLoc Begin, End;
      getDebugLocations(getBBPairForRegion(&S->getRegion()), Begin, End);
      OptimizationRemarkAnalysis R(DEBUG_TYPE, "OutOfQuota", Begin,
                                   S->getEntry());
      R << "maximal number of operations exceeded during zone analysis";
      S->getFunction().getContext().diagnose(R);
      return false;
    }

    Zone = OriginalZone = Knowledge(nullptr, EltUnused, EltKnown, EltWritten);
    DEBUG(dbgs() << "Computed Zone:\n"; OriginalZone.print(dbgs(), 4));

    assert(Zone.isUsable() && OriginalZone.isUsable());
    return true;
  }

  /// Try to map as many scalars to unused array elements as possible.
  ///
  /// Multiple scalars might be mappable to intersecting unused array element
  /// zones, but we can only chose one. This is a greedy algorithm, therefore
  /// the first processed element claims it.
  void greedyCollapse() {
    bool Modified = false;

    for (auto &Stmt : *S) {
      for (auto *MA : Stmt) {
        if (!MA->isLatestArrayKind())
          continue;
        if (!MA->isWrite())
          continue;

        if (MA->isMayWrite()) {
          DEBUG(dbgs() << "Access " << MA
                       << " pruned because it is a MAY_WRITE\n");
          OptimizationRemarkMissed R(DEBUG_TYPE, "TargetMayWrite",
                                     MA->getAccessInstruction());
          R << "Skipped possible mapping target because it is not an "
               "unconditional overwrite";
          S->getFunction().getContext().diagnose(R);
          continue;
        }

        if (Stmt.getNumIterators() == 0) {
          DEBUG(dbgs() << "Access " << MA
                       << " pruned because it is not in a loop\n");
          OptimizationRemarkMissed R(DEBUG_TYPE, "WriteNotInLoop",
                                     MA->getAccessInstruction());
          R << "skipped possible mapping target because it is not in a loop";
          S->getFunction().getContext().diagnose(R);
          continue;
        }

        if (isScalarAccess(MA)) {
          DEBUG(dbgs() << "Access " << MA
                       << " pruned because it writes only a single element\n");
          OptimizationRemarkMissed R(DEBUG_TYPE, "ScalarWrite",
                                     MA->getAccessInstruction());
          R << "skipped possible mapping target because the memory location "
               "written to does not depend on its outer loop";
          S->getFunction().getContext().diagnose(R);
          continue;
        }

        if (!isa<StoreInst>(MA->getAccessInstruction())) {
          DEBUG(dbgs() << "Access " << MA
                       << " pruned because it is not a StoreInst\n");
          OptimizationRemarkMissed R(DEBUG_TYPE, "NotAStore",
                                     MA->getAccessInstruction());
          R << "skipped possible mapping target because non-store instructions "
               "are not supported";
          S->getFunction().getContext().diagnose(R);
          continue;
        }

        // Check for more than one element acces per statement instance.
        // Currently we expect write accesses to be functional, eg. disallow
        //
        //   { Stmt[0] -> [i] : 0 <= i < 2 }
        //
        // This may occur when some accesses to the element write/read only
        // parts of the element, eg. a single byte. Polly then divides each
        // element into subelements of the smallest access length, normal access
        // then touch multiple of such subelements. It is very common when the
        // array is accesses with memset, memcpy or memmove which take i8*
        // arguments.
        isl::union_map AccRel = MA->getLatestAccessRelation();
        if (!AccRel.is_single_valued().is_true()) {
          DEBUG(dbgs() << "Access " << MA
                       << " is incompatible because it writes multiple "
                          "elements per instance\n");
          OptimizationRemarkMissed R(DEBUG_TYPE, "NonFunctionalAccRel",
                                     MA->getAccessInstruction());
          R << "skipped possible mapping target because it writes more than "
               "one element";
          S->getFunction().getContext().diagnose(R);
          continue;
        }

        isl::union_set TouchedElts = AccRel.range();
        if (!TouchedElts.is_subset(CompatibleElts)) {
          DEBUG(
              dbgs()
              << "Access " << MA
              << " is incompatible because it touches incompatible elements\n");
          OptimizationRemarkMissed R(DEBUG_TYPE, "IncompatibleElts",
                                     MA->getAccessInstruction());
          R << "skipped possible mapping target because a target location "
               "cannot be reliably analyzed";
          S->getFunction().getContext().diagnose(R);
          continue;
        }

        assert(isCompatibleAccess(MA));
        NumberOfCompatibleTargets++;
        DEBUG(dbgs() << "Analyzing target access " << MA << "\n");
        if (collapseScalarsToStore(MA))
          Modified = true;
      }
    }

    if (Modified)
      DeLICMScopsModified++;
  }

  /// Dump the internal information about a performed DeLICM to @p OS.
  void print(llvm::raw_ostream &OS, int Indent = 0) {
    if (!Zone.isUsable()) {
      OS.indent(Indent) << "Zone not computed\n";
      return;
    }

    printStatistics(OS, Indent);
    if (!isModified()) {
      OS.indent(Indent) << "No modification has been made\n";
      return;
    }
    printAccesses(OS, Indent);
  }
};

class DeLICM : public ScopPass {
private:
  DeLICM(const DeLICM &) = delete;
  const DeLICM &operator=(const DeLICM &) = delete;

  /// The pass implementation, also holding per-scop data.
  std::unique_ptr<DeLICMImpl> Impl;

  void collapseToUnused(Scop &S) {
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    Impl = make_unique<DeLICMImpl>(&S, &LI);

    if (!Impl->computeZone()) {
      DEBUG(dbgs() << "Abort because cannot reliably compute lifetimes\n");
      return;
    }

    DEBUG(dbgs() << "Collapsing scalars to unused array elements...\n");
    Impl->greedyCollapse();

    DEBUG(dbgs() << "\nFinal Scop:\n");
    DEBUG(dbgs() << S);
  }

public:
  static char ID;
  explicit DeLICM() : ScopPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitive<ScopInfoRegionPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesAll();
  }

  virtual bool runOnScop(Scop &S) override {
    // Free resources for previous scop's computation, if not yet done.
    releaseMemory();

    collapseToUnused(S);

    auto ScopStats = S.getStatistics();
    NumValueWrites += ScopStats.NumValueWrites;
    NumValueWritesInLoops += ScopStats.NumValueWritesInLoops;
    NumPHIWrites += ScopStats.NumPHIWrites;
    NumPHIWritesInLoops += ScopStats.NumPHIWritesInLoops;
    NumSingletonWrites += ScopStats.NumSingletonWrites;
    NumSingletonWritesInLoops += ScopStats.NumSingletonWritesInLoops;

    return false;
  }

  virtual void printScop(raw_ostream &OS, Scop &S) const override {
    if (!Impl)
      return;
    assert(Impl->getScop() == &S);

    OS << "DeLICM result:\n";
    Impl->print(OS);
  }

  virtual void releaseMemory() override { Impl.reset(); }
};

char DeLICM::ID;
} // anonymous namespace

Pass *polly::createDeLICMPass() { return new DeLICM(); }

INITIALIZE_PASS_BEGIN(DeLICM, "polly-delicm", "Polly - DeLICM/DePRE", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(DeLICM, "polly-delicm", "Polly - DeLICM/DePRE", false,
                    false)

bool polly::isConflicting(
    isl::union_set ExistingOccupied, isl::union_set ExistingUnused,
    isl::union_map ExistingKnown, isl::union_map ExistingWrites,
    isl::union_set ProposedOccupied, isl::union_set ProposedUnused,
    isl::union_map ProposedKnown, isl::union_map ProposedWrites,
    llvm::raw_ostream *OS, unsigned Indent) {
  Knowledge Existing(std::move(ExistingOccupied), std::move(ExistingUnused),
                     std::move(ExistingKnown), std::move(ExistingWrites));
  Knowledge Proposed(std::move(ProposedOccupied), std::move(ProposedUnused),
                     std::move(ProposedKnown), std::move(ProposedWrites));

  return Knowledge::isConflicting(Existing, Proposed, OS, Indent);
}
