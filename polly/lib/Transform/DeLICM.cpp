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
// * ValInst[] - An llvm::Value as defined at a specific timepoint.
//
//     A ValInst[] itself can be structured as one of:
//
//     * [] - An unknown value.
//         Always zero dimensions
//         Has no tuple id
//
//     * Value[] - An llvm::Value that is read-only in the SCoP, i.e. its
//                 runtime content does not depend on the timepoint.
//         Always zero dimensions
//         isl_id_get_name: Val_<NameOfValue>
//         isl_id_get_user: A pointer to an llvm::Value
//
//     * SCEV[...] - A synthesizable llvm::SCEV Expression.
//         In contrast to a Value[] is has at least one dimension per
//         SCEVAddRecExpr in the SCEV.
//
//     * [Domain[] -> Value[]] - An llvm::Value that may change during the
//                               Scop's execution.
//         The tuple itself has no id, but it wraps a map space holding a
//         statement instance which defines the llvm::Value as the map's domain
//         and llvm::Value itself as range.
//
// @see makeValInst()
//
// An annotation "{ Domain[] -> Scatter[] }" therefore means: A map from a
// statement instance to a timepoint, aka a schedule. There is only one scatter
// space, but most of the time multiple statements are processed in one set.
// This is why most of the time isl_union_map has to be used.
//
// The basic algorithm works as follows:
// At first we verify that the SCoP is compatible with this technique. For
// instance, two writes cannot write to the same location at the same statement
// instance because we cannot determine within the polyhedral model which one
// comes first. Once this was verified, we compute zones at which an array
// element is unused. This computation can fail if it takes too long. Then the
// main algorithm is executed. Because every store potentially trails an unused
// zone, we start at stores. We search for a scalar (MemoryKind::Value or
// MemoryKind::PHI) that we can map to the array element overwritten by the
// store, preferably one that is used by the store or at least the ScopStmt.
// When it does not conflict with the lifetime of the values in the array
// element, the map is applied and the unused zone updated as it is now used. We
// continue to try to map scalars to the array element until there are no more
// candidates to map. The algorithm is greedy in the sense that the first scalar
// not conflicting will be mapped. Other scalars processed later that could have
// fit the same unused zone will be rejected. As such the result depends on the
// processing order.
//
//===----------------------------------------------------------------------===//

#include "polly/DeLICM.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/VirtualInstruction.h"
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

cl::opt<bool>
    DelicmComputeKnown("polly-delicm-compute-known",
                       cl::desc("Compute known content of array elements"),
                       cl::init(true), cl::Hidden, cl::cat(PollyCategory));

STATISTIC(DeLICMAnalyzed, "Number of successfully analyzed SCoPs");
STATISTIC(DeLICMOutOfQuota,
          "Analyses aborted because max_operations was reached");
STATISTIC(DeLICMIncompatible, "Number of SCoPs incompatible for analysis");
STATISTIC(MappedValueScalars, "Number of mapped Value scalars");
STATISTIC(MappedPHIScalars, "Number of mapped PHI scalars");
STATISTIC(TargetsMapped, "Number of stores used for at least one mapping");
STATISTIC(DeLICMScopsModified, "Number of SCoPs optimized");

/// Class for keeping track of scalar def-use chains in the polyhedral
/// representation.
///
/// MemoryKind::Value:
/// There is one definition per llvm::Value or zero (read-only values defined
/// before the SCoP) and an arbitrary number of reads.
///
/// MemoryKind::PHI, MemoryKind::ExitPHI:
/// There is at least one write (the incoming blocks/stmts) and one
/// (MemoryKind::PHI) or zero (MemoryKind::ExitPHI) reads per llvm::PHINode.
class ScalarDefUseChains {
private:
  /// The definitions (i.e. write MemoryAccess) of a MemoryKind::Value scalar.
  DenseMap<const ScopArrayInfo *, MemoryAccess *> ValueDefAccs;

  /// List of all uses (i.e. read MemoryAccesses) for a MemoryKind::Value
  /// scalar.
  DenseMap<const ScopArrayInfo *, SmallVector<MemoryAccess *, 4>> ValueUseAccs;

  /// The receiving part (i.e. read MemoryAccess) of a MemoryKind::PHI scalar.
  DenseMap<const ScopArrayInfo *, MemoryAccess *> PHIReadAccs;

  /// List of all incoming values (write MemoryAccess) of a MemoryKind::PHI or
  /// MemoryKind::ExitPHI scalar.
  DenseMap<const ScopArrayInfo *, SmallVector<MemoryAccess *, 4>>
      PHIIncomingAccs;

public:
  /// Find the MemoryAccesses that access the ScopArrayInfo-represented memory.
  ///
  /// @param S The SCoP to analyze.
  void compute(Scop *S) {
    // Purge any previous result.
    reset();

    for (auto &Stmt : *S) {
      for (auto *MA : Stmt) {
        if (MA->isOriginalValueKind() && MA->isWrite()) {
          auto *SAI = MA->getScopArrayInfo();
          assert(!ValueDefAccs.count(SAI) &&
                 "There can be at most one "
                 "definition per MemoryKind::Value scalar");
          ValueDefAccs[SAI] = MA;
        }

        if (MA->isOriginalValueKind() && MA->isRead())
          ValueUseAccs[MA->getScopArrayInfo()].push_back(MA);

        if (MA->isOriginalAnyPHIKind() && MA->isRead()) {
          auto *SAI = MA->getScopArrayInfo();
          assert(!PHIReadAccs.count(SAI) &&
                 "There must be exactly one read "
                 "per PHI (that's where the PHINode is)");
          PHIReadAccs[SAI] = MA;
        }

        if (MA->isOriginalAnyPHIKind() && MA->isWrite())
          PHIIncomingAccs[MA->getScopArrayInfo()].push_back(MA);
      }
    }
  }

  /// Free all memory used by the analysis.
  void reset() {
    ValueDefAccs.clear();
    ValueUseAccs.clear();
    PHIReadAccs.clear();
    PHIIncomingAccs.clear();
  }

  MemoryAccess *getValueDef(const ScopArrayInfo *SAI) const {
    return ValueDefAccs.lookup(SAI);
  }

  ArrayRef<MemoryAccess *> getValueUses(const ScopArrayInfo *SAI) const {
    auto It = ValueUseAccs.find(SAI);
    if (It == ValueUseAccs.end())
      return {};
    return It->second;
  }

  MemoryAccess *getPHIRead(const ScopArrayInfo *SAI) const {
    return PHIReadAccs.lookup(SAI);
  }

  ArrayRef<MemoryAccess *> getPHIIncomings(const ScopArrayInfo *SAI) const {
    auto It = PHIIncomingAccs.find(SAI);
    if (It == PHIIncomingAccs.end())
      return {};
    return It->second;
  }
};

isl::union_map computeReachingDefinition(isl::union_map Schedule,
                                         isl::union_map Writes, bool InclDef,
                                         bool InclRedef) {
  return computeReachingWrite(Schedule, Writes, false, InclDef, InclRedef);
}

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
  auto ScatterSpace = getScatterSpace(Schedule);
  auto DomSpace = give(isl_set_get_space(Writes.keep()));

  auto ReachOverwrite = computeScalarReachingOverwrite(
      Schedule, give(isl_union_set_from_set(Writes.take())), InclPrevWrite,
      InclOverwrite);

  auto ResultSpace = give(isl_space_map_from_domain_and_range(
      ScatterSpace.take(), DomSpace.take()));
  return singleton(std::move(ReachOverwrite), ResultSpace);
}

/// Compute the reaching definition of a scalar.
///
/// Compared to computeReachingDefinition, there is just one element which is
/// accessed and therefore only a set if instances that accesses that element is
/// required.
///
/// @param Schedule  { DomainWrite[] -> Scatter[] }
/// @param Writes    { DomainWrite[] }
/// @param InclDef   Include the timepoint of the definition to the result.
/// @param InclRedef Include the timepoint of the overwrite into the result.
///
/// @return { Scatter[] -> DomainWrite[] }
isl::union_map computeScalarReachingDefinition(isl::union_map Schedule,
                                               isl::union_set Writes,
                                               bool InclDef, bool InclRedef) {

  // { DomainWrite[] -> Element[] }
  auto Defs = give(isl_union_map_from_domain(Writes.take()));

  // { [Element[] -> Scatter[]] -> DomainWrite[] }
  auto ReachDefs =
      computeReachingDefinition(Schedule, Defs, InclDef, InclRedef);

  // { Scatter[] -> DomainWrite[] }
  return give(isl_union_set_unwrap(
      isl_union_map_range(isl_union_map_curry(ReachDefs.take()))));
}

/// Compute the reaching definition of a scalar.
///
/// This overload accepts only a single writing statement as an isl_map,
/// consequently the result also is only a single isl_map.
///
/// @param Schedule  { DomainWrite[] -> Scatter[] }
/// @param Writes    { DomainWrite[] }
/// @param InclDef   Include the timepoint of the definition to the result.
/// @param InclRedef Include the timepoint of the overwrite into the result.
///
/// @return { Scatter[] -> DomainWrite[] }
isl::map computeScalarReachingDefinition( // { Domain[] -> Zone[] }
    isl::union_map Schedule, isl::set Writes, bool InclDef, bool InclRedef) {
  auto DomainSpace = give(isl_set_get_space(Writes.keep()));
  auto ScatterSpace = getScatterSpace(Schedule);

  //  { Scatter[] -> DomainWrite[] }
  auto UMap = computeScalarReachingDefinition(
      Schedule, give(isl_union_set_from_set(Writes.take())), InclDef,
      InclRedef);

  auto ResultSpace = give(isl_space_map_from_domain_and_range(
      ScatterSpace.take(), DomainSpace.take()));
  return singleton(UMap, ResultSpace);
}

/// Create a domain-to-unknown value mapping.
///
/// Value instances that do not represent a specific value are represented by an
/// unnamed tuple of 0 dimensions. Its meaning depends on the context. It can
/// either mean a specific but unknown value which cannot be represented by
/// other means. It conflicts with itself because those two unknown ValInsts may
/// have different concrete values at runtime.
///
/// The other meaning is an arbitrary or wildcard value that can be chosen
/// freely, like LLVM's undef. If matched with an unknown ValInst, there is no
/// conflict.
///
/// @param Domain { Domain[] }
///
/// @return { Domain[] -> ValInst[] }
isl::union_map makeUnknownForDomain(isl::union_set Domain) {
  return give(isl_union_map_from_domain(Domain.take()));
}

/// Create a domain-to-unknown value mapping.
///
/// @see makeUnknownForDomain(isl::union_set)
///
/// @param Domain { Domain[] }
///
/// @return { Domain[] -> ValInst[] }
isl::map makeUnknownForDomain(isl::set Domain) {
  return give(isl_map_from_domain(Domain.take()));
}

/// Return whether @p Map maps to an unknown value.
///
/// @param { [] -> ValInst[] }
bool isMapToUnknown(const isl::map &Map) {
  auto Space = give(isl_space_range(isl_map_get_space(Map.keep())));
  return !isl_map_has_tuple_id(Map.keep(), isl_dim_set) &&
         !isl_space_is_wrapping(Space.keep()) &&
         isl_map_dim(Map.keep(), isl_dim_out) == 0;
}

/// Return only the mappings that map to known values.
///
/// @param UMap { [] -> ValInst[] }
///
/// @return { [] -> ValInst[] }
isl::union_map filterKnownValInst(const isl::union_map &UMap) {
  auto Result = give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  UMap.foreach_map([=, &Result](isl::map Map) -> isl::stat {
    if (!isMapToUnknown(Map))
      Result = give(isl_union_map_add_map(Result.take(), Map.take()));
    return isl::stat::ok;
  });
  return Result;
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
  Relevant = give(isl_union_map_coalesce(Relevant.take()));
  auto RelevantDomain = give(isl_union_map_domain(Relevant.copy()));
  auto Simplified =
      give(isl_union_map_gist_domain(Relevant.take(), RelevantDomain.take()));
  Simplified = give(isl_union_map_coalesce(Simplified.take()));
  return give(
      isl_union_map_intersect_domain(Simplified.take(), Universe.take()));
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

std::string printIntruction(Instruction *Instr, bool IsForDebug = false) {
  std::string Result;
  raw_string_ostream OS(Result);
  Instr->print(OS, IsForDebug);
  OS.flush();
  size_t i = 0;
  while (i < Result.size() && Result[i] == ' ')
    i += 1;
  return Result.substr(i);
}

/// Base class for algorithms based on zones, like DeLICM.
class ZoneAlgorithm {
protected:
  /// Hold a reference to the isl_ctx to avoid it being freed before we released
  /// all of the isl objects.
  ///
  /// This must be declared before any other member that holds an isl object.
  /// This guarantees that the shared_ptr and its isl_ctx is destructed last,
  /// after all other members free'd the isl objects they were holding.
  std::shared_ptr<isl_ctx> IslCtx;

  /// Cached reaching definitions for each ScopStmt.
  ///
  /// Use getScalarReachingDefinition() to get its contents.
  DenseMap<ScopStmt *, isl::map> ScalarReachDefZone;

  /// The analyzed Scop.
  Scop *S;

  /// LoopInfo analysis used to determine whether values are synthesizable.
  LoopInfo *LI;

  /// Parameter space that does not need realignment.
  isl::space ParamSpace;

  /// Space the schedule maps to.
  isl::space ScatterSpace;

  /// Cached version of the schedule and domains.
  isl::union_map Schedule;

  /// Combined access relations of all MemoryKind::Array READ accesses.
  /// { DomainRead[] -> Element[] }
  isl::union_map AllReads;

  /// Combined access relations of all MemoryKind::Array, MAY_WRITE accesses.
  /// { DomainMayWrite[] -> Element[] }
  isl::union_map AllMayWrites;

  /// Combined access relations of all MemoryKind::Array, MUST_WRITE accesses.
  /// { DomainMustWrite[] -> Element[] }
  isl::union_map AllMustWrites;

  /// The value instances written to array elements of all write accesses.
  /// { [Element[] -> DomainWrite[]] -> ValInst[] }
  isl::union_map AllWriteValInst;

  /// All reaching definitions for  MemoryKind::Array writes.
  /// { [Element[] -> Zone[]] -> DomainWrite[] }
  isl::union_map WriteReachDefZone;

  /// Map llvm::Values to an isl identifier.
  /// Used with -polly-use-llvm-names=false as an alternative method to get
  /// unique ids that do not depend on pointer values.
  DenseMap<Value *, isl::id> ValueIds;

  /// Prepare the object before computing the zones of @p S.
  ZoneAlgorithm(Scop *S, LoopInfo *LI)
      : IslCtx(S->getSharedIslCtx()), S(S), LI(LI),
        Schedule(give(S->getSchedule())) {

    auto Domains = give(S->getDomains());

    Schedule =
        give(isl_union_map_intersect_domain(Schedule.take(), Domains.take()));
    ParamSpace = give(isl_union_map_get_space(Schedule.keep()));
    ScatterSpace = getScatterSpace(Schedule);
  }

private:
  /// Check whether @p Stmt can be accurately analyzed by zones.
  ///
  /// What violates our assumptions:
  /// - A load after a write of the same location; we assume that all reads
  ///   occur before the writes.
  /// - Two writes to the same location; we cannot model the order in which
  ///   these occur.
  ///
  /// Scalar reads implicitly always occur before other accesses therefore never
  /// violate the first condition. There is also at most one write to a scalar,
  /// satisfying the second condition.
  bool isCompatibleStmt(ScopStmt *Stmt) {
    auto Stores = makeEmptyUnionMap();
    auto Loads = makeEmptyUnionMap();

    // This assumes that the MemoryKind::Array MemoryAccesses are iterated in
    // order.
    for (auto *MA : *Stmt) {
      if (!MA->isLatestArrayKind())
        continue;

      auto AccRel =
          give(isl_union_map_from_map(getAccessRelationFor(MA).take()));

      if (MA->isRead()) {
        // Reject load after store to same location.
        if (!isl_union_map_is_disjoint(Stores.keep(), AccRel.keep())) {
          OptimizationRemarkMissed R(DEBUG_TYPE, "LoadAfterStore",
                                     MA->getAccessInstruction());
          R << "load after store of same element in same statement";
          R << " (previous stores: " << Stores;
          R << ", loading: " << AccRel << ")";
          S->getFunction().getContext().diagnose(R);
          return false;
        }

        Loads = give(isl_union_map_union(Loads.take(), AccRel.take()));

        continue;
      }

      if (!isa<StoreInst>(MA->getAccessInstruction())) {
        DEBUG(dbgs() << "WRITE that is not a StoreInst not supported\n");
        OptimizationRemarkMissed R(DEBUG_TYPE, "UnusualStore",
                                   MA->getAccessInstruction());
        R << "encountered write that is not a StoreInst: "
          << printIntruction(MA->getAccessInstruction());
        S->getFunction().getContext().diagnose(R);
        return false;
      }

      // In region statements the order is less clear, eg. the load and store
      // might be in a boxed loop.
      if (Stmt->isRegionStmt() &&
          !isl_union_map_is_disjoint(Loads.keep(), AccRel.keep())) {
        OptimizationRemarkMissed R(DEBUG_TYPE, "StoreInSubregion",
                                   MA->getAccessInstruction());
        R << "store is in a non-affine subregion";
        S->getFunction().getContext().diagnose(R);
        return false;
      }

      // Do not allow more than one store to the same location.
      if (!isl_union_map_is_disjoint(Stores.keep(), AccRel.keep())) {
        OptimizationRemarkMissed R(DEBUG_TYPE, "StoreAfterStore",
                                   MA->getAccessInstruction());
        R << "store after store of same element in same statement";
        R << " (previous stores: " << Stores;
        R << ", storing: " << AccRel << ")";
        S->getFunction().getContext().diagnose(R);
        return false;
      }

      Stores = give(isl_union_map_union(Stores.take(), AccRel.take()));
    }

    return true;
  }

  void addArrayReadAccess(MemoryAccess *MA) {
    assert(MA->isLatestArrayKind());
    assert(MA->isRead());

    // { DomainRead[] -> Element[] }
    auto AccRel = getAccessRelationFor(MA);
    AllReads = give(isl_union_map_add_map(AllReads.take(), AccRel.copy()));
  }

  void addArrayWriteAccess(MemoryAccess *MA) {
    assert(MA->isLatestArrayKind());
    assert(MA->isWrite());
    auto *Stmt = MA->getStatement();

    // { Domain[] -> Element[] }
    auto AccRel = getAccessRelationFor(MA);

    if (MA->isMustWrite())
      AllMustWrites =
          give(isl_union_map_add_map(AllMustWrites.take(), AccRel.copy()));

    if (MA->isMayWrite())
      AllMayWrites =
          give(isl_union_map_add_map(AllMayWrites.take(), AccRel.copy()));

    // { Domain[] -> ValInst[] }
    auto WriteValInstance =
        makeValInst(MA->getAccessValue(), Stmt,
                    LI->getLoopFor(MA->getAccessInstruction()->getParent()),
                    MA->isMustWrite());

    // { Domain[] -> [Element[] -> Domain[]] }
    auto IncludeElement =
        give(isl_map_curry(isl_map_domain_map(AccRel.copy())));

    // { [Element[] -> DomainWrite[]] -> ValInst[] }
    auto EltWriteValInst = give(
        isl_map_apply_domain(WriteValInstance.take(), IncludeElement.take()));

    AllWriteValInst = give(
        isl_union_map_add_map(AllWriteValInst.take(), EltWriteValInst.take()));
  }

protected:
  isl::union_set makeEmptyUnionSet() const {
    return give(isl_union_set_empty(ParamSpace.copy()));
  }

  isl::union_map makeEmptyUnionMap() const {
    return give(isl_union_map_empty(ParamSpace.copy()));
  }

  /// Check whether @p S can be accurately analyzed by zones.
  bool isCompatibleScop() {
    for (auto &Stmt : *S) {
      if (!isCompatibleStmt(&Stmt))
        return false;
    }
    return true;
  }

  /// Get the schedule for @p Stmt.
  ///
  /// The domain of the result is as narrow as possible.
  isl::map getScatterFor(ScopStmt *Stmt) const {
    auto ResultSpace = give(isl_space_map_from_domain_and_range(
        Stmt->getDomainSpace(), ScatterSpace.copy()));
    return give(isl_union_map_extract_map(Schedule.keep(), ResultSpace.take()));
  }

  /// Get the schedule of @p MA's parent statement.
  isl::map getScatterFor(MemoryAccess *MA) const {
    return getScatterFor(MA->getStatement());
  }

  /// Get the schedule for the statement instances of @p Domain.
  isl::union_map getScatterFor(isl::union_set Domain) const {
    return give(isl_union_map_intersect_domain(Schedule.copy(), Domain.take()));
  }

  /// Get the schedule for the statement instances of @p Domain.
  isl::map getScatterFor(isl::set Domain) const {
    auto ResultSpace = give(isl_space_map_from_domain_and_range(
        isl_set_get_space(Domain.keep()), ScatterSpace.copy()));
    auto UDomain = give(isl_union_set_from_set(Domain.copy()));
    auto UResult = getScatterFor(std::move(UDomain));
    auto Result = singleton(std::move(UResult), std::move(ResultSpace));
    assert(!Result ||
           isl_set_is_equal(give(isl_map_domain(Result.copy())).keep(),
                            Domain.keep()) == isl_bool_true);
    return Result;
  }

  /// Get the domain of @p Stmt.
  isl::set getDomainFor(ScopStmt *Stmt) const {
    return give(isl_set_remove_redundancies(Stmt->getDomain()));
  }

  /// Get the domain @p MA's parent statement.
  isl::set getDomainFor(MemoryAccess *MA) const {
    return getDomainFor(MA->getStatement());
  }

  /// Get the access relation of @p MA.
  ///
  /// The domain of the result is as narrow as possible.
  isl::map getAccessRelationFor(MemoryAccess *MA) const {
    auto Domain = getDomainFor(MA);
    auto AccRel = give(MA->getLatestAccessRelation());
    return give(isl_map_intersect_domain(AccRel.take(), Domain.take()));
  }

  /// Get the reaching definition of a scalar defined in @p Stmt.
  ///
  /// Note that this does not depend on the llvm::Instruction, only on the
  /// statement it is defined in. Therefore the same computation can be reused.
  ///
  /// @param Stmt The statement in which a scalar is defined.
  ///
  /// @return { Scatter[] -> DomainDef[] }
  isl::map getScalarReachingDefinition(ScopStmt *Stmt) {
    auto &Result = ScalarReachDefZone[Stmt];
    if (Result)
      return Result;

    auto Domain = getDomainFor(Stmt);
    Result = computeScalarReachingDefinition(Schedule, Domain, false, true);
    simplify(Result);

    return Result;
  }

  /// Get the reaching definition of a scalar defined in @p DefDomain.
  ///
  /// @param DomainDef { DomainDef[] }
  ///              The write statements to get the reaching definition for.
  ///
  /// @return { Scatter[] -> DomainDef[] }
  isl::map getScalarReachingDefinition(isl::set DomainDef) {
    auto DomId = give(isl_set_get_tuple_id(DomainDef.keep()));
    auto *Stmt = static_cast<ScopStmt *>(isl_id_get_user(DomId.keep()));

    auto StmtResult = getScalarReachingDefinition(Stmt);

    return give(isl_map_intersect_range(StmtResult.take(), DomainDef.take()));
  }

  /// Create a statement-to-unknown value mapping.
  ///
  /// @param Stmt The statement whose instances are mapped to unknown.
  ///
  /// @return { Domain[] -> ValInst[] }
  isl::map makeUnknownForDomain(ScopStmt *Stmt) const {
    return ::makeUnknownForDomain(getDomainFor(Stmt));
  }

  /// Create an isl_id that represents @p V.
  isl::id makeValueId(Value *V) {
    if (!V)
      return nullptr;

    auto &Id = ValueIds[V];
    if (Id.is_null()) {
      auto Name = getIslCompatibleName("Val_", V, ValueIds.size() - 1,
                                       std::string(), UseInstructionNames);
      Id = give(isl_id_alloc(IslCtx.get(), Name.c_str(), V));
    }
    return Id;
  }

  /// Create the space for an llvm::Value that is available everywhere.
  isl::space makeValueSpace(Value *V) {
    auto Result = give(isl_space_set_from_params(ParamSpace.copy()));
    return give(isl_space_set_tuple_id(Result.take(), isl_dim_set,
                                       makeValueId(V).take()));
  }

  /// Create a set with the llvm::Value @p V which is available everywhere.
  isl::set makeValueSet(Value *V) {
    auto Space = makeValueSpace(V);
    return give(isl_set_universe(Space.take()));
  }

  /// Create a mapping from a statement instance to the instance of an
  /// llvm::Value that can be used in there.
  ///
  /// Although LLVM IR uses single static assignment, llvm::Values can have
  /// different contents in loops, when they get redefined in the last
  /// iteration. This function tries to get the statement instance of the
  /// previous definition, relative to a user.
  ///
  /// Example:
  /// for (int i = 0; i < N; i += 1) {
  /// DEF:
  ///    int v = A[i];
  /// USE:
  ///    use(v);
  ///  }
  ///
  /// The value instance used by statement instance USE[i] is DEF[i]. Hence,
  /// makeValInst returns:
  ///
  /// { USE[i] -> [DEF[i] -> v[]] : 0 <= i < N }
  ///
  /// @param Val       The value to get the instance of.
  /// @param UserStmt  The statement that uses @p Val. Can be nullptr.
  /// @param Scope     Loop the using instruction resides in.
  /// @param IsCertain Pass true if the definition of @p Val is a
  ///                  MUST_WRITE or false if the write is conditional.
  ///
  /// @return { DomainUse[] -> ValInst[] }
  isl::map makeValInst(Value *Val, ScopStmt *UserStmt, Loop *Scope,
                       bool IsCertain = true) {
    // When known knowledge is disabled, just return the unknown value. It will
    // either get filtered out or conflict with itself.
    if (!DelicmComputeKnown)
      return makeUnknownForDomain(UserStmt);

    // If the definition/write is conditional, the value at the location could
    // be either the written value or the old value. Since we cannot know which
    // one, consider the value to be unknown.
    if (!IsCertain)
      return makeUnknownForDomain(UserStmt);

    auto DomainUse = getDomainFor(UserStmt);
    auto VUse = VirtualUse::create(S, UserStmt, Scope, Val, true);
    switch (VUse.getKind()) {
    case VirtualUse::Constant:
    case VirtualUse::Block:
    case VirtualUse::Hoisted:
    case VirtualUse::ReadOnly: {
      // The definition does not depend on the statement which uses it.
      auto ValSet = makeValueSet(Val);
      return give(
          isl_map_from_domain_and_range(DomainUse.take(), ValSet.take()));
    }

    case VirtualUse::Synthesizable: {
      auto *ScevExpr = VUse.getScevExpr();
      auto UseDomainSpace = give(isl_set_get_space(DomainUse.keep()));

      // Construct the SCEV space.
      // TODO: Add only the induction variables referenced in SCEVAddRecExpr
      // expressions, not just all of them.
      auto ScevId = give(isl_id_alloc(UseDomainSpace.get_ctx().get(), nullptr,
                                      const_cast<SCEV *>(ScevExpr)));
      auto ScevSpace =
          give(isl_space_drop_dims(UseDomainSpace.copy(), isl_dim_set, 0, 0));
      ScevSpace = give(
          isl_space_set_tuple_id(ScevSpace.take(), isl_dim_set, ScevId.copy()));

      // { DomainUse[] -> ScevExpr[] }
      auto ValInst = give(isl_map_identity(isl_space_map_from_domain_and_range(
          UseDomainSpace.copy(), ScevSpace.copy())));
      return ValInst;
    }

    case VirtualUse::Intra: {
      // Definition and use is in the same statement. We do not need to compute
      // a reaching definition.

      // { llvm::Value }
      auto ValSet = makeValueSet(Val);

      // {  UserDomain[] -> llvm::Value }
      auto ValInstSet =
          give(isl_map_from_domain_and_range(DomainUse.take(), ValSet.take()));

      // { UserDomain[] -> [UserDomain[] - >llvm::Value] }
      auto Result =
          give(isl_map_reverse(isl_map_domain_map(ValInstSet.take())));
      simplify(Result);
      return Result;
    }

    case VirtualUse::Inter: {
      // The value is defined in a different statement.

      auto *Inst = cast<Instruction>(Val);
      auto *ValStmt = S->getStmtFor(Inst);

      // If the llvm::Value is defined in a removed Stmt, we cannot derive its
      // domain. We could use an arbitrary statement, but this could result in
      // different ValInst[] for the same llvm::Value.
      if (!ValStmt)
        return ::makeUnknownForDomain(DomainUse);

      // { DomainDef[] }
      auto DomainDef = getDomainFor(ValStmt);

      // { Scatter[] -> DomainDef[] }
      auto ReachDef = getScalarReachingDefinition(DomainDef);

      // { DomainUse[] -> Scatter[] }
      auto UserSched = getScatterFor(DomainUse);

      // { DomainUse[] -> DomainDef[] }
      auto UsedInstance =
          give(isl_map_apply_range(UserSched.take(), ReachDef.take()));

      // { llvm::Value }
      auto ValSet = makeValueSet(Val);

      // { DomainUse[] -> llvm::Value[] }
      auto ValInstSet =
          give(isl_map_from_domain_and_range(DomainUse.take(), ValSet.take()));

      // { DomainUse[] -> [DomainDef[] -> llvm::Value]  }
      auto Result =
          give(isl_map_range_product(UsedInstance.take(), ValInstSet.take()));

      simplify(Result);
      return Result;
    }
    }
    llvm_unreachable("Unhandled use type");
  }

  /// Compute the different zones.
  void computeCommon() {
    AllReads = makeEmptyUnionMap();
    AllMayWrites = makeEmptyUnionMap();
    AllMustWrites = makeEmptyUnionMap();
    AllWriteValInst = makeEmptyUnionMap();

    for (auto &Stmt : *S) {
      for (auto *MA : Stmt) {
        if (!MA->isLatestArrayKind())
          continue;

        if (MA->isRead())
          addArrayReadAccess(MA);

        if (MA->isWrite())
          addArrayWriteAccess(MA);
      }
    }

    // { DomainWrite[] -> Element[] }
    auto AllWrites =
        give(isl_union_map_union(AllMustWrites.copy(), AllMayWrites.copy()));

    // { [Element[] -> Zone[]] -> DomainWrite[] }
    WriteReachDefZone =
        computeReachingDefinition(Schedule, AllWrites, false, true);
    simplify(WriteReachDefZone);
  }

  /// Print the current state of all MemoryAccesses to @p.
  void printAccesses(llvm::raw_ostream &OS, int Indent = 0) const {
    OS.indent(Indent) << "After accesses {\n";
    for (auto &Stmt : *S) {
      OS.indent(Indent + 4) << Stmt.getBaseName() << "\n";
      for (auto *MA : Stmt)
        MA->print(OS);
    }
    OS.indent(Indent) << "}\n";
  }

public:
  /// Return the SCoP this object is analyzing.
  Scop *getScop() const { return S; }
};

/// Implementation of the DeLICM/DePRE transformation.
class DeLICMImpl : public ZoneAlgorithm {
private:
  /// Knowledge before any transformation took place.
  Knowledge OriginalZone;

  /// Current knowledge of the SCoP including all already applied
  /// transformations.
  Knowledge Zone;

  /// For getting the MemoryAccesses that write or read a given scalar.
  ScalarDefUseChains DefUse;

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
      auto *MA = DefUse.getValueDef(SAI);
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
      auto *MA = DefUse.getPHIRead(SAI);
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
    for (auto *MA : DefUse.getValueUses(SAI))
      Reads =
          give(isl_union_set_add_set(Reads.take(), getDomainFor(MA).take()));

    // { DomainRead[] -> Scatter[] }
    auto ReadSchedule = getScatterFor(Reads);

    auto *DefMA = DefUse.getValueDef(SAI);
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
    for (auto *MA : DefUse.getPHIIncomings(SAI)) {
      auto Scatter = getScatterFor(MA);
      PHIWriteScatter =
          give(isl_union_map_add_map(PHIWriteScatter.take(), Scatter.take()));
    }

    // { DomainPHIRead[] -> Scatter[] }
    auto PHIReadScatter = getScatterFor(DefUse.getPHIRead(SAI));

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

    auto *DefMA = DefUse.getValueDef(SAI);
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

    // { DomainDef[] -> ValInst[] }
    auto ValInst = makeValInst(V, DefMA->getStatement(),
                               LI->getLoopFor(DefInst->getParent()));

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
    for (auto *MA : DefUse.getValueUses(SAI)) {
      // { DomainUse[] }
      auto Domain = getDomainFor(MA);

      // { DomainUse[] -> Element[] }
      auto NewAccRel = give(isl_union_map_intersect_domain(
          UseTarget.copy(), isl_union_set_from_set(Domain.take())));
      simplify(NewAccRel);

      assert(isl_union_map_n_map(NewAccRel.keep()) == 1);
      MA->setNewAccessRelation(isl_map_from_union_map(NewAccRel.take()));
    }

    auto *WA = DefUse.getValueDef(SAI);
    WA->setNewAccessRelation(DefTarget.copy());
    applyLifetime(Proposed);

    MappedValueScalars++;
    NumberOfMappedValueScalars += 1;
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
    for (auto *MA : DefUse.getPHIIncomings(SAI)) {
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
    auto *PHIRead = DefUse.getPHIRead(SAI);
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

    for (auto *MA : DefUse.getPHIIncomings(SAI))
      UniverseWritesDom = give(isl_union_set_add_set(UniverseWritesDom.take(),
                                                     getDomainFor(MA).take()));

    auto RelevantWritesTarget = WritesTarget;
    if (DelicmOverapproximateWrites)
      WritesTarget = expandMapping(WritesTarget, UniverseWritesDom);

    auto ExpandedWritesDom = give(isl_union_map_domain(WritesTarget.copy()));
    if (!isl_union_set_is_subset(UniverseWritesDom.keep(),
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
    // Redirect the PHI incoming writes.
    for (auto *MA : DefUse.getPHIIncomings(SAI)) {
      // { DomainWrite[] }
      auto Domain = getDomainFor(MA);

      // { DomainWrite[] -> Element[] }
      auto NewAccRel = give(isl_union_map_intersect_domain(
          WriteTarget.copy(), isl_union_set_from_set(Domain.take())));
      simplify(NewAccRel);

      assert(isl_union_map_n_map(NewAccRel.keep()) == 1);
      MA->setNewAccessRelation(isl_map_from_union_map(NewAccRel.take()));
    }

    // Redirect the PHI read.
    auto *PHIRead = DefUse.getPHIRead(SAI);
    PHIRead->setNewAccessRelation(ReadTarget.copy());
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

        auto *DefAcc = DefUse.getValueDef(SAI);
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
        for (auto *PHIWrite : DefUse.getPHIIncomings(SAI)) {
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

  /// Compute which value an array element stores at every instant.
  ///
  /// @return { [Element[] -> Zone[]] -> ValInst[] }
  isl::union_map computeKnown() const {
    // { [Element[] -> Zone[]] -> [Element[] -> DomainWrite[]] }
    auto EltReachdDef =
        distributeDomain(give(isl_union_map_curry(WriteReachDefZone.copy())));

    // { [Element[] -> DomainWrite[]] -> ValInst[] }
    auto AllKnownWriteValInst = filterKnownValInst(AllWriteValInst);

    // { [Element[] -> Zone[]] -> ValInst[] }
    return EltReachdDef.apply_range(AllKnownWriteValInst);
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
  DeLICMImpl(Scop *S, LoopInfo *LI) : ZoneAlgorithm(S, LI) {}

  /// Calculate the lifetime (definition to last use) of every array element.
  ///
  /// @return True if the computed lifetimes (#Zone) is usable.
  bool computeZone() {
    // Check that nothing strange occurs.
    if (!isCompatibleScop()) {
      DeLICMIncompatible++;
      return false;
    }

    DefUse.compute(S);
    isl::union_set EltUnused;
    isl::union_map EltKnown, EltWritten;

    {
      IslMaxOperationsGuard MaxOpGuard(IslCtx.get(), DelicmMaxOps);

      computeCommon();

      EltUnused = computeLifetime();
      EltKnown = computeKnown();
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
    DEBUG(S.print(dbgs()));
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
