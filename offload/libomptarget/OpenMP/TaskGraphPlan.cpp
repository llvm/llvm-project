//===-- OpenMP/TaskGraphPlan.cpp - A taskgraph's mapped data --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpenMP/TaskGraphPlan.h"

#include "OpenMP/Mapping.h"
#include "PluginManager.h"
#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Shared/EnvironmentVar.h"
#include "device.h"
#include "omptarget.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

using namespace llvm;
using namespace llvm::omp::target;

namespace mt = omptarget::maptype;

/// Report every replay that finds the host captures relocated or the mapping
/// state changed since the previous one.  Off by default; tests use it to check
/// that a change they set up was noticed.
static bool reportDrift() {
  static BoolEnvar ReportDrift("LIBOMPTARGET_TASKGRAPH_REPORT_DRIFT", false);
  return ReportDrift;
}

namespace llvm {
namespace omp {
namespace target {

/// Builds a graph's data plan and keeps it current across replays: everything
/// that reads or writes the plan's own state, which a backend cannot see.
class TaskGraphPlanner {
public:
  static bool buildTaskGraphPlan(DeviceTy &Device, TaskGraphTy &Graph);
  static bool
  acquireTaskGraphMappings(DeviceTy &Device, TaskGraphTy &Graph,
                           SmallVectorImpl<TaskGraphArgUpdateTy> &Updates,
                           bool &Resync);
  static void commitTaskGraphUpdates(TaskGraphTy &Graph);
  static void releaseTaskGraphMappings(DeviceTy &Device, TaskGraphTy &Graph,
                                       bool Replayed = true);
  static void beginTaskGraphReplay(TaskGraphTy &Graph);
  static void releaseTaskGraphPlan(DeviceTy &Device, TaskGraphTy &Graph);

private:
  /// What a map does to its region's reference count, which is what decides
  /// whether the map performs its implicit copies.
  enum class ClaimKindTy : uint8_t {
    /// A `target` region's map: takes a reference for the duration of the
    /// region and gives it back, so one map both creates the mapping and
    /// removes it.
    Balanced,
    /// `target enter data`: takes a reference and keeps it.
    Enter,
    /// `target exit data`: gives one back.
    Exit,
    /// `target update`, or a map naming a region the graph does not map itself.
    /// It copies explicitly or not at all, and either way the count is not its
    /// business.
    Lookup,
  };

  /// One map naming one region, in encounter order, so that the reference count
  /// over each range can be walked once the maps have been gathered.
  struct ClaimTy {
    ClaimKindTy Kind = ClaimKindTy::Lookup;
    /// The argument the map came from.
    TaskGraphArgSlotTy *Arg = nullptr;
    char *Begin = nullptr;
    uint64_t Size = 0;
    int64_t ArgType = 0;
    /// For a struct member's map, the map of its enclosing struct.  The member
    /// shares the struct's mapping and leaves the reference count alone, so it
    /// copies on whichever change the struct's map owns.
    const TaskGraphArgSlotTy *Parent = nullptr;
  };

  /// The ranges being gathered while a plan is built, keyed by host begin.
  using RangeMapTy = std::map<char *, TaskGraphRangeTy>;

  static MutableArrayRef<TaskGraphArgSlotTy> argsOf(TaskGraphNode *Node) {
    return {Node->ArgPlan, Node->NumArgPlans};
  }

  static bool hostResidentIfUnmapped(const DeviceTy &Device,
                                     const TaskGraphRangeTy &Range);
  static void reportUnresolvable(DeviceTy &Device,
                                 const TaskGraphRangeTy &Range);
  static TaskGraphRangeTy *
  containingRange(MutableArrayRef<TaskGraphRangeTy> Ranges, char *Begin,
                  uint64_t Size);
  static void gatherClaim(RangeMapTy &Ranges, const ClaimTy &C);
  static bool rekeyRanges(TaskGraphTy &Graph, SmallVectorImpl<bool> &HostMoved);
  static void *allocateRangeStorage(DeviceTy &Device,
                                    const TaskGraphRangeTy &Range);
  static bool publishRange(DeviceTy &Device,
                           MappingInfoTy::HDTTMapAccessorTy &HDTTMap,
                           const TaskGraphRangeTy &Range, void *Storage);
  static bool acquireRange(DeviceTy &Device, TaskGraphRangeTy &Range,
                           void *&Address, bool &Inserted, bool &HostResident);
  static void releaseReference(DeviceTy &Device, TaskGraphRangeTy &Range,
                               bool UseHoldRefCount, bool ForceDelete,
                               bool MayRemove);
  static void releaseRange(DeviceTy &Device, TaskGraphRangeTy &Range);
  static bool pointeeDeviceBase(const TaskGraphRangeTy &Pointee,
                                int64_t PointeeOffset, char *HstPteeBase,
                                char *HstPteeBegin, char *&TgtPteeBase);
  static bool stageTaskGraphAttachments(DeviceTy &Device, TaskGraphTy &Graph,
                                        MutableArrayRef<uint8_t> Changes);
  static void restoreTaskGraphAttachments(TaskGraphTy &Graph);
  static bool copiesThisReplay(TaskGraphNode *Node, uint32_t Range);
  static bool buildOpsAndParams(TaskGraphTy &Graph);
};

} // namespace target
} // namespace omp
} // namespace llvm

/// Why a map cannot be lowered to a device buffer plus transfers, or null if it
/// can.  Struct members and pointer-and-pointee entries need nothing special:
/// once relocated they are a host address and an extent like any other, and
/// gathering puts a member in the range of its enclosing struct.  Attachments
/// and privatised arguments are classified before this is asked.
static const char *whyUnlowerable(int64_t ArgType, void *Mapper) {
  if (Mapper)
    return "uses a user-defined mapper";
  if (mt::isNonContig(ArgType))
    return "is a non-contiguous transfer";
  return nullptr;
}

/// Whether an argument is the fix-up for a lambda's by-reference capture, which
/// writes the captured variable's device address into the device copy of the
/// lambda.  It looks like a literal, but the plan has no operation that writes
/// into an argument's device copy, so a graph with one cannot be lowered.
static bool isLambdaCaptureFixup(int64_t ArgType) {
  constexpr int64_t Fixup = OMP_TGT_MAPTYPE_PTR_AND_OBJ |
                            OMP_TGT_MAPTYPE_LITERAL | OMP_TGT_MAPTYPE_IMPLICIT;
  return !mt::isTargetParam(ArgType) && (ArgType & Fixup) == Fixup;
}

/// Whether an argument is a pointer attachment: a device pointer to be pointed
/// at a device pointee once both are mapped.  `PRIVATE | ATTACH` is not one; it
/// initialises a privatised pointer, so it is classified as a private.
static bool isAttachment(int64_t ArgType) {
  return mt::isAttach(ArgType) && !mt::isPrivate(ArgType);
}

/// Whether an argument is privatised: storage for the construct alone, which
/// the graph provides rather than maps.
static bool isPrivatised(int64_t ArgType) { return mt::isPrivate(ArgType); }

/// The index of the argument mapping the struct that argument \p ArgType is a
/// member of, or -1 if it is not a member whose copies follow that struct's
/// reference count.  A pointer-and-pointee member counts on its pointee, as in
/// targetDataBegin.
static int64_t structParentIndex(int64_t ArgType) {
  if (!mt::isMemberOf(ArgType) || mt::isPtrAndObj(ArgType))
    return -1;
  return ((ArgType & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

/// Whether child \p To of an irreducible region can be reached from child
/// \p From along the region's edges.
static bool reaches(const TaskGraphIrreducibleRegion &Region, uint32_t From,
                    uint32_t To) {
  SmallVector<bool, 16> Seen(Region.NumChildren, false);
  SmallVector<uint32_t, 16> Work = {From};
  Seen[From] = true;
  while (!Work.empty()) {
    const uint32_t Child = Work.pop_back_val();
    for (const TaskGraphEdge &E : Region.edges()) {
      if (E.Src != static_cast<int32_t>(Child) || E.Dst < 0 ||
          static_cast<uint32_t>(E.Dst) >= Region.NumChildren)
        continue;
      if (static_cast<uint32_t>(E.Dst) == To)
        return true;
      if (!Seen[E.Dst]) {
        Seen[E.Dst] = true;
        Work.push_back(E.Dst);
      }
    }
  }
  return false;
}

/// Whether the region tree orders leaves \p A and \p B one way or the other, so
/// that they never run at the same time.  It does if their lowest common region
/// is sequential, or is irreducible and has a path between the children leading
/// to them.  It also does for an exclusive region: its children are lowered in
/// child order, the order the plan's reference-count walk assumes, and the HIP
/// backend declines such graphs anyway.  A parallel region's children are
/// unordered.
static bool areOrdered(const TaskGraphElement &A, const TaskGraphElement &B) {
  auto depth = [](const TaskGraphElement *E) {
    uint32_t Depth = 0;
    for (; E->Parent; E = E->Parent)
      ++Depth;
    return Depth;
  };
  // Lift the deeper one to the other's depth, then both until their parents
  // meet: the parent is then the lowest common region, and each is its child
  // leading to the leaf.
  const TaskGraphElement *X = &A, *Y = &B;
  uint32_t DepthX = depth(X), DepthY = depth(Y);
  for (; DepthX > DepthY; --DepthX)
    X = X->Parent;
  for (; DepthY > DepthX; --DepthY)
    Y = Y->Parent;
  while (X->Parent != Y->Parent) {
    X = X->Parent;
    Y = Y->Parent;
  }
  assert(X != Y && X->Parent && "two distinct leaves of one graph");

  switch (X->Parent->getKind()) {
  case TGKind::SeqRegion:
  case TGKind::ExclRegion:
    return true;
  case TGKind::IrreducibleRegion: {
    const auto &Region = *cast<TaskGraphIrreducibleRegion>(X->Parent);
    return reaches(Region, X->ChildIndex, Y->ChildIndex) ||
           reaches(Region, Y->ChildIndex, X->ChildIndex);
  }
  default:
    return false;
  }
}

/// Alignment for one entry's slot in the private block: the alignment the host
/// object happens to have, as PrivateArgumentManager infers it, capped at 16
/// and floored at a pointer so that a pointer initialisation is one aligned
/// word.
static uint64_t privateAlignment(const void *HstPtr) {
  constexpr uint64_t MaxAlign = 16;
  const uint64_t Addr = reinterpret_cast<uintptr_t>(HstPtr);
  const uint64_t Natural = Addr ? (Addr & ~(Addr - 1)) : MaxAlign;
  return std::clamp<uint64_t>(Natural, sizeof(void *), MaxAlign);
}

/// Whether \p Range uses the host memory directly when nothing has mapped it:
/// under `unified_shared_memory` unless a map asked for `close`, and always
/// under automatic zero-copy.  The same rule getTargetPointer applies.
bool TaskGraphPlanner::hostResidentIfUnmapped(const DeviceTy &Device,
                                              const TaskGraphRangeTy &Range) {
  return ((Device.PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY) &&
          !Range.Close) ||
         (Device.PM->getRequirements() & OMPX_REQ_AUTO_ZERO_COPY);
}

/// A range's end as a void *, so that a report stream prints it as an address
/// rather than as a string.
static void *endOf(const TaskGraphRangeTy &Range) { return Range.end(); }

/// Say why a range could not be resolved.  Looks the range up again, which
/// only the failing path pays for.
void TaskGraphPlanner::reportUnresolvable(DeviceTy &Device,
                                          const TaskGraphRangeTy &Range) {
  MappingInfoTy &MappingInfo = Device.getMappingInfo();
  auto HDTTMap = MappingInfo.HostDataToTargetMap.getExclusiveAccessor();
  LookupResult LR = MappingInfo.lookupMapping(HDTTMap, Range.HostBegin,
                                              static_cast<int64_t>(Range.Size));
  const bool Contained = LR.Flags.IsContained;
  const bool Partial = LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter;
  LR.TPR.reset();
  HDTTMap.destroy();

  if (Partial) {
    // Outside a graph an explicit map of this shape is refused too.  An
    // implicit one may borrow the existing entry, but a range gathered from
    // several maps has no single map's implicitness to go on.
    REPORT() << "taskgraph maps [" << Range.HostBegin << ", " << endOf(Range)
             << ") which partially overlaps a region mapped outside the graph, "
                "leaving it with neither a device address to share nor one it "
                "can map without aliasing";
    return;
  }
  if (Range.RequiresPresent && !Contained) {
    REPORT() << "taskgraph requires [" << Range.HostBegin << ", "
             << endOf(Range) << ") to be present, and it is not";
    return;
  }
  REPORT() << "taskgraph could not map " << Range.Size << " bytes for ["
           << Range.HostBegin << ", " << endOf(Range) << ")";
}

/// The direction a data leaf transfers an argument in.  A `target data` region
/// records its enter and exit actions with one shared map array, so a `tofrom`
/// map asks for both directions on both; the enter action only copies in and
/// the exit action only copies out.  A map naming neither direction, such as
/// `alloc` or `release`, transfers nothing.
static mt::TransferDir directionFor(const TaskGraphTargetDataNode *DataNode,
                                    int64_t ArgType) {
  mt::TransferDir Dir = mt::transferDir(ArgType);
  if (isa<TaskGraphTargetEnterDataNode>(DataNode))
    return mt::isTo(ArgType) ? mt::TransferDir::H2D : mt::TransferDir::None;
  if (isa<TaskGraphTargetExitDataNode>(DataNode))
    return mt::isFrom(ArgType) ? mt::TransferDir::D2H : mt::TransferDir::None;
  return Dir;
}

/// Where a leaf's argument \p Arg currently begins, after relocation.
static char *argBegin(TaskGraphNode *Node, uint32_t Arg) {
  if (auto *TargetNode = dyn_cast<TaskGraphTargetNode>(Node))
    return static_cast<char *>(
        static_cast<KernelArgsTy *>(TargetNode->KernelArgs)->ArgPtrs[Arg]);
  return static_cast<char *>(cast<TaskGraphTargetDataNode>(Node)->Args[Arg]);
}

/// Where a leaf's argument \p Arg has its base, which for an attach entry is
/// the pointer object.
static char *argBase(TaskGraphNode *Node, uint32_t Arg) {
  if (auto *TargetNode = dyn_cast<TaskGraphTargetNode>(Node))
    return static_cast<char *>(
        static_cast<KernelArgsTy *>(TargetNode->KernelArgs)->ArgBasePtrs[Arg]);
  return static_cast<char *>(
      cast<TaskGraphTargetDataNode>(Node)->ArgsBase[Arg]);
}

/// How many bytes a leaf's argument \p Arg currently covers.
static uint64_t argSize(TaskGraphNode *Node, uint32_t Arg) {
  if (auto *TargetNode = dyn_cast<TaskGraphTargetNode>(Node))
    return static_cast<KernelArgsTy *>(TargetNode->KernelArgs)->ArgSizes[Arg];
  return cast<TaskGraphTargetDataNode>(Node)->ArgSizes[Arg];
}

/// Whether two host extents have to share a range.  Overlap is strict, so
/// extents that merely abut are kept apart.  A zero extent `[p, p)` shares with
/// an extent that contains `p` or begins at it, so that a zero-length array
/// section falls into the range mapping its base.
static bool shareRange(const char *A, uint64_t ASize, const char *B,
                       uint64_t BSize) {
  if (!ASize && !BSize)
    return A == B;
  if (!ASize)
    return B <= A && A < B + BSize;
  if (!BSize)
    return A <= B && B < A + ASize;
  return A < B + BSize && B < A + ASize;
}

/// The range holding all of \p Size bytes at \p Begin, or null if none does.
/// A zero extent is held by a range containing it or beginning at it.  The
/// ranges have to be in host-address order, as they are when the plan is
/// built.
TaskGraphRangeTy *
TaskGraphPlanner::containingRange(MutableArrayRef<TaskGraphRangeTy> Ranges,
                                  char *Begin, uint64_t Size) {
  auto *It =
      llvm::upper_bound(Ranges, Begin, [](char *B, const TaskGraphRangeTy &R) {
        return B < R.begin();
      });
  if (It == Ranges.begin())
    return nullptr;
  TaskGraphRangeTy &Range = *std::prev(It);
  if (!Size)
    return Begin < Range.end() || Begin == Range.begin() ? &Range : nullptr;
  return Begin + Size <= Range.end() ? &Range : nullptr;
}

/// Gather one map into the ranges, merging it with every range it shares host
/// memory with.  This mirrors the mapping table, where a map contained in an
/// existing entry reuses that entry's device memory, so a `target` region
/// mapping part of an enclosing `target data` region joins its range.  Ranges
/// that merely abut stay apart, as separate mappings would.
void TaskGraphPlanner::gatherClaim(RangeMapTy &Ranges, const ClaimTy &C) {
  TaskGraphRangeTy Merged;
  Merged.RefDelta = C.Parent                       ? 0
                    : C.Kind == ClaimKindTy::Enter ? 1
                    : C.Kind == ClaimKindTy::Exit  ? -1
                                                   : 0;
  Merged.RequiresPresent = mt::isPresent(C.ArgType);
  // A `present` map never creates the mapping.  Another map in the graph may,
  // in which case the range is present by the time the `present` map runs.
  Merged.MayCreate =
      (C.Kind == ClaimKindTy::Balanced || C.Kind == ClaimKindTy::Enter) &&
      !Merged.RequiresPresent;
  Merged.Deleted = mt::isDelete(C.ArgType);
  Merged.Holds = mt::isOmpxHold(C.ArgType);
  Merged.Close = mt::isClose(C.ArgType);

  char *Lo = C.Begin;
  char *Hi = C.Begin + C.Size;
  // The ranges are disjoint and ordered, so only the one before the insertion
  // point can reach back over the claim's begin, and the ones the claim shares
  // with follow it.  The merged extent may bridge ranges that were apart.
  auto It = Ranges.upper_bound(Lo);
  if (It != Ranges.begin() &&
      shareRange(std::prev(It)->first, std::prev(It)->second.Size, Lo, Hi - Lo))
    --It;
  while (It != Ranges.end() &&
         shareRange(It->first, It->second.Size, Lo, Hi - Lo)) {
    const TaskGraphRangeTy &R = It->second;
    Lo = std::min(Lo, R.begin());
    Hi = std::max(Hi, R.end());
    Merged.RefDelta += R.RefDelta;
    Merged.MayCreate |= R.MayCreate;
    Merged.RequiresPresent |= R.RequiresPresent;
    Merged.Deleted |= R.Deleted;
    Merged.Holds |= R.Holds;
    Merged.Close |= R.Close;
    It = Ranges.erase(It);
  }
  Merged.HostBegin = Lo;
  Merged.Size = Hi - Lo;
  Ranges.emplace(Lo, Merged);
}

/// Move the ranges to wherever relocation has just put the captures they were
/// resolved from, and set \p HostMoved, indexed by range Id, for the ones that
/// moved.
///
/// The grouping is fixed, so a range may only move as a whole: every argument
/// in it must have moved by the same distance, and the moved ranges must not
/// overlap one another.  A relocation that breaks this, for instance by moving
/// only one of two leaves mapping the same object, is refused and leaves the
/// plan unchanged.
bool TaskGraphPlanner::rekeyRanges(TaskGraphTy &Graph,
                                   SmallVectorImpl<bool> &HostMoved) {
  TaskGraphDataPlanTy &Plan = Graph.Plan;
  const uint64_t Stamp = ++Plan.Relocations;

  // Extents are fixed as well, since a lowered transfer cannot grow.  An
  // argument whose size changed, such as a reallocated Fortran allocatable, is
  // refused.
  auto resized = [&](const TaskGraphRangeTy &R, uint64_t Was, uint64_t Now) {
    if (Was == Now)
      return false;
    REPORT() << "taskgraph replay found the region [" << R.HostBegin << ", "
             << endOf(R) << ") covering " << Now
             << " bytes where the recording had " << Was
             << "; a graph's mapped extents are fixed when it is recorded";
    return true;
  };

  // The first argument to reach a range in this relocation records the
  // distance and later ones are checked against it.  A zero-length entry whose
  // address did not move is exempt: its address can come from a slot the
  // relocation helper leaves alone, such as a static pointer into a stack
  // array, and it simply keeps its offset in the range.
  auto note = [&](TaskGraphRangeTy &R, char *Now, uint64_t Offset,
                  uint64_t Size) {
    const int64_t D = Now - (R.begin() + Offset);
    if (!Size && !D)
      return true;
    if (R.MovedInRelocation != Stamp) {
      R.MovedInRelocation = Stamp;
      R.MovedBy = D;
      return true;
    }
    if (R.MovedBy == D)
      return true;
    REPORT() << "taskgraph relocation moved part of the region [" << R.HostBegin
             << ", " << endOf(R) << ") without moving the rest of it";
    return false;
  };

  // Check every entry naming a range as its own before moving anything.  A
  // pointee is only looked up, and is re-read every replay, so it is not
  // checked.
  for (TaskGraphNode *Node : Graph.leaves()) {
    for (auto [I, S] : llvm::enumerate(argsOf(Node))) {
      const uint32_t Arg = static_cast<uint32_t>(I);
      const TaskGraphArgBaseTy *A = &S.get();
      char *Now;
      TaskGraphRangeTy *Range;
      uint64_t Offset, Size;

      switch (A->getKind()) {
      case TaskGraphArgBaseTy::KindTy::Mapped: {
        Now = argBegin(Node, Arg);
        const auto *M = cast<TaskGraphMappedArgTy>(A);
        Range = M->Range;
        Size = M->Size;
        Offset = M->Offset;
        break;
      }
      case TaskGraphArgBaseTy::KindTy::Attach: {
        Now = argBase(Node, Arg);
        const auto *T = cast<TaskGraphAttachArgTy>(A);
        Range = T->PtrRange;
        Size = T->Size;
        Offset = T->Offset;
        break;
      }
      default:
        continue;
      }

      if (resized(*Range, Size, argSize(Node, Arg)) ||
          !note(*Range, Now, Offset, Size))
        return false;
    }
  }

  auto moves = [&](const TaskGraphRangeTy &R) {
    return R.MovedInRelocation == Stamp && R.MovedBy;
  };
  auto newBegin = [&](const TaskGraphRangeTy &R) {
    return moves(R) ? R.begin() + R.MovedBy : R.begin();
  };

  // The moved ranges must still be disjoint from one another and from the rest.
  // Check before moving anything, so a refusal leaves the plan as it was.
  SmallVector<const TaskGraphRangeTy *, 16> Sorted;
  bool AnyMoves = false;
  for (const TaskGraphRangeTy &Range : Plan.Ranges) {
    Sorted.push_back(&Range);
    AnyMoves |= moves(Range);
  }
  if (!AnyMoves)
    return true;
  llvm::sort(Sorted, [&](const TaskGraphRangeTy *A, const TaskGraphRangeTy *B) {
    return newBegin(*A) < newBegin(*B);
  });
  for (size_t I = 1; I < Sorted.size(); ++I) {
    const TaskGraphRangeTy &Prev = *Sorted[I - 1];
    const TaskGraphRangeTy &Next = *Sorted[I];
    // A range of no extent shares its begin with nothing.
    if (newBegin(Next) != newBegin(Prev) &&
        newBegin(Next) >= newBegin(Prev) + Prev.Size)
      continue;
    const TaskGraphRangeTy &Clash = moves(Next) ? Next : Prev;
    const TaskGraphRangeTy &Over = moves(Next) ? Prev : Next;
    REPORT() << "taskgraph relocation put [" << Clash.HostBegin << ", "
             << endOf(Clash) << ") over [" << Over.HostBegin << ", "
             << endOf(Over)
             << "), which were separate regions when the graph was recorded";
    return false;
  }

  for (TaskGraphRangeTy &Range : Plan.Ranges)
    if (moves(Range)) {
      Range.HostBegin = Range.begin() + Range.MovedBy;
      HostMoved[Range.Id] = true;
    }
  return true;
}

/// Device storage for a range the graph maps and finds absent.
void *TaskGraphPlanner::allocateRangeStorage(DeviceTy &Device,
                                             const TaskGraphRangeTy &Range) {
  return Device.allocData(static_cast<int64_t>(Range.Size), Range.HostBegin);
}

/// Publish \p Storage as the mapping of \p Range, found absent under
/// \p HDTTMap, holding the graph's hold reference.  A range the graph leaves
/// mapped also gets the reference an unbalanced `target enter data` leaves: on
/// the hold count under `ompx_hold`, and on the dynamic count otherwise.
bool TaskGraphPlanner::publishRange(DeviceTy &Device,
                                    MappingInfoTy::HDTTMapAccessorTy &HDTTMap,
                                    const TaskGraphRangeTy &Range,
                                    void *Storage) {
  const uintptr_t Tgt = reinterpret_cast<uintptr_t>(Storage);
  HostDataToTargetTy *Entry = Device.getMappingInfo().insertMapping(
      HDTTMap, /*HstPtrBase=*/Range.HostBegin, Range.HostBegin,
      static_cast<int64_t>(Range.Size), /*TgtAllocBegin=*/Tgt,
      /*TgtPtrBegin=*/Tgt, /*UseHoldRefCount=*/true, /*HstPtrName=*/nullptr);
  if (!Entry)
    return false;
  if (Range.persists())
    Entry->incRefCount(/*UseHoldRefCount=*/Range.Holds);
  return true;
}

/// Resolve one range against the mapping table for a replay about to start,
/// mapping it if the graph brings it into the device data environment, and
/// take a hold reference so the mapping cannot be removed while the replay
/// runs.  A mapping the graph creates starts with its hold count at one, which
/// is that reference.
///
/// Sets \p Address to the range's device address for this replay, or null if
/// it is absent and the graph does not map it; \p Inserted if the graph created
/// the mapping or the storage; and \p HostResident if the device uses the host
/// memory directly.
bool TaskGraphPlanner::acquireRange(DeviceTy &Device, TaskGraphRangeTy &Range,
                                    void *&Address, bool &Inserted,
                                    bool &HostResident) {
  MappingInfoTy &MappingInfo = Device.getMappingInfo();
  Address = nullptr;
  Inserted = false;
  HostResident = false;
  Range.DropsLast = false;

  const int64_t Size = static_cast<int64_t>(Range.Size);

  // The exclusive accessor is held until the range is borrowed or published, so
  // no other thread can map the range in between.  A range of no extent only
  // ever borrows; lookupMapping finds an entry for it by the entry's extended
  // range, so a section starting before the mapped bytes is still found.
  auto HDTTMap = MappingInfo.HostDataToTargetMap.getExclusiveAccessor();
  LookupResult LR = MappingInfo.lookupMapping(HDTTMap, Range.HostBegin, Size);

  // A self-contained range gets storage of its own unless the program has
  // since mapped it.  A mapping containing the range is borrowed below, as for
  // any other range; a partial overlap declines the replay.
  if (Range.SelfContained && !LR.Flags.IsContained) {
    const bool Overlaps = LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter;
    LR.TPR.reset();
    HDTTMap.destroy();
    if (Overlaps) {
      MESSAGE("taskgraph keeps [" DPxMOD ", " DPxMOD
              ") on the device for itself, and it is now partly mapped "
              "outside the graph",
              DPxPTR(Range.HostBegin), DPxPTR(Range.end()));
      return false;
    }
    void *Storage = allocateRangeStorage(Device, Range);
    if (!Storage) {
      REPORT() << "taskgraph could not allocate " << Range.Size
               << " bytes of device memory for [" << Range.HostBegin << ", "
               << endOf(Range) << ")";
      return false;
    }
    Address = Storage;
    Inserted = true;
    Range.StorageHeld = true;
    // Claimed only for as long as the storage exists: the host address need
    // not be the program's once the replay ends.
    MappingInfo.addTaskGraphClaim(Range.HostBegin, Size);
    return true;
  }

  if (LR.Flags.IsContained) {
    HostDataToTargetTy *Entry = LR.TPR.getEntry();
    // The graph drops one reference to a mapping it found, which removes it
    // only if it is the last -- and one with a hold reference is not removed
    // by a dynamic one at all.
    Range.DropsLast = Range.releases() && Entry->getHoldRefCount() == 0 &&
                      Entry->getDynRefCount() == 1;
    Entry->incRefCount(/*UseHoldRefCount=*/true);
    Address = reinterpret_cast<void *>(
        Entry->TgtPtrBegin +
        (reinterpret_cast<uintptr_t>(Range.HostBegin) - Entry->HstPtrBegin));
    Range.Held = true;
    HDTTMap.destroy();
    // Whoever mapped the range may still be copying its contents in, and the
    // graph's first reader must not overtake that.
    if (void *Event = Entry->getEvent()) {
      AsyncInfoTy AsyncInfo(Device);
      if (Device.waitEvent(Event, AsyncInfo) != OFFLOAD_SUCCESS ||
          AsyncInfo.synchronize() != OFFLOAD_SUCCESS)
        return false;
    }
    return true;
  }

  // A partial overlap can be neither borrowed nor mapped.
  if (LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter) {
    LR.TPR.reset();
    HDTTMap.destroy();
    reportUnresolvable(Device, Range);
    return false;
  }
  LR.TPR.reset();

  if (hostResidentIfUnmapped(Device, Range)) {
    HostResident = true;
    Address = Range.HostBegin;
    return true;
  }

  // `present` is only enforced on a range the graph does not map; one it does
  // map is brought in here, before any of the graph runs.
  if (!Range.Size || !Range.MayCreate) {
    if (!Range.RequiresPresent)
      return true; // absent, and not ours to map: its updates are no-ops.
    HDTTMap.destroy();
    reportUnresolvable(Device, Range);
    return false;
  }

  void *Storage = allocateRangeStorage(Device, Range);
  if (!Storage) {
    REPORT() << "taskgraph could not map " << Range.Size << " bytes for ["
             << Range.HostBegin << ", " << endOf(Range) << ")";
    return false;
  }
  if (!publishRange(Device, HDTTMap, Range, Storage)) {
    HDTTMap.destroy();
    REPORT() << "taskgraph could not publish its mapping of ["
             << Range.HostBegin << ", " << endOf(Range) << ")";
    return false;
  }
  Inserted = true;
  Address = Storage;
  Range.Held = true;
  return true;
}

/// Drop one reference to the mapping covering \p Range and, if that was the
/// last and \p MayRemove is set, remove it.  Follows the protocol of
/// postProcessingTargetDataEnd, so that only the last of several concurrent
/// releasers frees the entry.
void TaskGraphPlanner::releaseReference(DeviceTy &Device,
                                        TaskGraphRangeTy &Range,
                                        bool UseHoldRefCount, bool ForceDelete,
                                        bool MayRemove) {
  MappingInfoTy &MappingInfo = Device.getMappingInfo();
  const int64_t Size = static_cast<int64_t>(Range.Size);

  TargetPointerResultTy TPR = MappingInfo.getTgtPtrBegin(
      Range.HostBegin, Size, /*UpdateRefCount=*/true, UseHoldRefCount,
      /*MustContain=*/true, ForceDelete, /*FromDataEnd=*/true);
  if (!TPR.isPresent() || !TPR.getEntry())
    return;

  bool DelEntry = MayRemove && TPR.Flags.IsLast && !TPR.isHostPointer();
  HostDataToTargetTy *Entry = TPR.getEntry();

  Entry->unlock();
  MappingInfoTy::HDTTMapAccessorTy HDTTMap =
      MappingInfo.HostDataToTargetMap.getExclusiveAccessor();
  Entry->lock();
  const bool NotLastUser = Entry->decDataEndThreadCount() != 0;
  if (DelEntry && (Entry->getTotalRefCount() != 0 || NotLastUser))
    DelEntry = false;
  TPR.setEntry(nullptr);
  if (!DelEntry)
    return;

  int Ret = MappingInfo.eraseMapEntry(HDTTMap, Entry, Size);
  HDTTMap.destroy();
  Ret |= MappingInfo.deallocTgtPtrAndEntry(Entry, Size);
  if (Ret != OFFLOAD_SUCCESS)
    REPORT() << "taskgraph could not release its mapping of ["
             << Range.HostBegin << ", " << endOf(Range) << ")";
}

/// Drop the hold reference acquireRange took on one range, and remove the
/// mapping if the graph created it and nothing in the graph asked for it to
/// stay.
void TaskGraphPlanner::releaseRange(DeviceTy &Device, TaskGraphRangeTy &Range) {
  if (Range.StorageHeld) {
    Range.StorageHeld = false;
    Range.PlanInserted = false;
    Device.getMappingInfo().removeTaskGraphClaim(
        Range.HostBegin, static_cast<int64_t>(Range.Size));
    if (Device.deleteData(Range.DevicePtr) != OFFLOAD_SUCCESS)
      REPORT() << "taskgraph could not free its device memory for ["
               << Range.HostBegin << ", " << endOf(Range) << ")";
    return;
  }
  if (!Range.Held)
    return;
  Range.Held = false;

  // A mapping the graph created and whose maps balance goes with the hold
  // reference, which was its only reference.
  releaseReference(Device, Range, /*UseHoldRefCount=*/true,
                   /*ForceDelete=*/false,
                   /*MayRemove=*/Range.PlanInserted && !Range.persists());

  // `delete` acts on the dynamic count, which `ompx_hold` survives, so it is a
  // separate release.
  if (Range.Deleted) {
    releaseReference(Device, Range, /*UseHoldRefCount=*/false,
                     /*ForceDelete=*/true, /*MayRemove=*/true);
    Range.PlanInserted = false;
    return;
  }

  // An unbalanced exit-data map drops a reference the graph did not take, on a
  // mapping the graph found rather than made.
  if (Range.releases() && !Range.PlanInserted)
    releaseReference(Device, Range, /*UseHoldRefCount=*/false,
                     /*ForceDelete=*/false, /*MayRemove=*/true);

  Range.PlanInserted = false;
}

/// Compute in \p TgtPteeBase the device address a pointer must hold, for an
/// entry whose base is the pointer object and whose begin is its pointee, given
/// the pointer's value \p HstPteeBase.  Null if the pointee has no device copy
/// this replay.  Returns false if the pointee moved within its range, which the
/// fixed grouping cannot follow.
bool TaskGraphPlanner::pointeeDeviceBase(const TaskGraphRangeTy &Pointee,
                                         int64_t PointeeOffset,
                                         char *HstPteeBase, char *HstPteeBegin,
                                         char *&TgtPteeBase) {
  TgtPteeBase = nullptr;

  if (HstPteeBegin != Pointee.begin() + PointeeOffset) {
    REPORT() << "taskgraph replay found the pointee at " << HstPteeBegin
             << " where the recording had it "
             << static_cast<void *>(Pointee.begin() + PointeeOffset)
             << ", inside the region [" << Pointee.HostBegin << ", "
             << endOf(Pointee)
             << "); a graph's mapped topology is fixed when it is recorded";
    return false;
  }
  if (!Pointee.DevicePtr || Pointee.HostResident)
    return true;

  // The device pointer holds the device counterpart of the pointee's base, not
  // of its begin: `map(s.p[10:n])` maps from `&s.p[10]`, but `s.p` points at
  // `s.p[0]`.
  TgtPteeBase = reinterpret_cast<char *>(Pointee.DevicePtr) + PointeeOffset -
                (HstPteeBegin - HstPteeBase);
  return true;
}

/// Fill in the bytes every attachment writes, for a replay about to start.
/// Restaged every replay, since the pointee's device address depends on the
/// current mapping state and on relocation.
bool TaskGraphPlanner::stageTaskGraphAttachments(
    DeviceTy &Device, TaskGraphTy &Graph, MutableArrayRef<uint8_t> Changes) {
  TaskGraphDataPlanTy &Plan = Graph.Plan;
  constexpr uint64_t VoidPtrSize = sizeof(void *);
  MappingInfoTy &MappingInfo = Device.getMappingInfo();

  for (TaskGraphNode *Node : Graph.leaves()) {
    for (auto [I, S] : llvm::enumerate(argsOf(Node))) {
      auto *A = dyn_cast<TaskGraphAttachArgTy>(&S.get());
      if (!A)
        continue;
      A->Enabled = false;
      const TaskGraphRangeTy &Pointer = *A->PtrRange;
      // A pointer with no device copy of its own needs no correction.
      if (!Pointer.DevicePtr || Pointer.HostResident)
        continue;

      const uint32_t Arg = static_cast<uint32_t>(I);
      char *HstPtrAddr = argBase(Node, Arg);
      char *HstPteeBegin = argBegin(Node, Arg);

      char *TgtPteeBase = nullptr;
      if (!pointeeDeviceBase(*A->PteeRange, A->PteeOffset,
                             *reinterpret_cast<char **>(HstPtrAddr),
                             HstPteeBegin, TgtPteeBase))
        return false;
      if (!TgtPteeBase)
        continue;

      char *Staging = Plan.AttachStaging.data() + A->StagingOffset;
      std::memcpy(Plan.AttachHostBytes.data() + A->StagingOffset, HstPtrAddr,
                  A->Size);
      // The rest of a Fortran descriptor is the same on the device.
      uint8_t &Change = Changes[Plan.argIndex(S)];
      auto stage = [&](uint64_t Offset, const void *Bytes, uint64_t Size) {
        if (!std::memcmp(Staging + Offset, Bytes, Size))
          return;
        std::memcpy(Staging + Offset, Bytes, Size);
        Change |= TaskGraphArgUpdateTy::DataChanged;
      };
      stage(0, &TgtPteeBase, VoidPtrSize);
      if (A->Size > VoidPtrSize)
        stage(VoidPtrSize, HstPtrAddr + VoidPtrSize, A->Size - VoidPtrSize);

      // Record the shadow pointer on the pointer's mapping, as the runtime
      // does, so that a `from` copy of the enclosing mapping does not hand the
      // host a device address.
      void *TgtPtrAddr =
          reinterpret_cast<char *>(Pointer.DevicePtr) + A->Offset;
      TargetPointerResultTy TPR = MappingInfo.getTgtPtrBegin(
          HstPtrAddr, static_cast<int64_t>(A->Size), /*UpdateRefCount=*/false,
          /*UseHoldRefCount=*/false, /*MustContain=*/true);
      if (TPR.getEntry())
        TPR.getEntry()->addShadowPointer(
            ShadowPtrInfoTy{reinterpret_cast<void **>(HstPtrAddr),
                            reinterpret_cast<void **>(TgtPtrAddr), TgtPteeBase,
                            static_cast<int64_t>(A->Size)});

      A->Enabled = true;
    }
  }
  return true;
}

/// Take the initial value of every privatised argument, when a replay of the
/// taskgraph begins, and mark those whose device copy is out of date.
///
/// A firstprivate is initialised from its variable as it stands when the
/// replay begins (OpenMP 6.0 [14.3]), which can be well before the graph runs,
/// except under `saved`, where it comes from the data environment saved at
/// recording and so is staged once. Only bytes that changed are marked for
/// copying.
void TaskGraphPlanner::beginTaskGraphReplay(TaskGraphTy &Graph) {
  TaskGraphDataPlanTy &Plan = Graph.Plan;

  for (TaskGraphNode *Node : Graph.leaves()) {
    for (auto [I, S] : llvm::enumerate(argsOf(Node))) {
      const uint32_t Arg = static_cast<uint32_t>(I);
      if (auto *F = dyn_cast<TaskGraphFirstPrivateArgTy>(&S.get())) {
        if (F->Saved && F->Staged)
          continue;

        char *Staging = Plan.PrivateStaging.data() + F->StagingOffset;
        const char *Bytes = argBegin(Node, Arg);

        if (!F->Staged || std::memcmp(Staging, Bytes, F->Size)) {
          std::memcpy(Staging, Bytes, F->Size);
          F->InitPending = true;
        }

        F->Staged = true;
      }
    }
  }
}

/// Put back the host value of every attached pointer whose range a copy brought
/// back from the device, for a replay that has finished.  The runtime does the
/// same at the end of a data region.  Pointers in ranges no copy brought back
/// are left as the program has them.
void TaskGraphPlanner::restoreTaskGraphAttachments(TaskGraphTy &Graph) {
  TaskGraphDataPlanTy &Plan = Graph.Plan;
  if (Plan.AttachStaging.empty())
    return;

  SmallVector<bool, 8> CopiedBack(Plan.Ranges.size(), false);
  for (const TaskGraphArgSlotTy &S : Plan.Args) {
    if (auto *A = dyn_cast<TaskGraphMappedArgTy>(&S.get())) {
      if (A->Out.Requested && A->Out.Enabled)
        CopiedBack[A->Range->Id] = true;
    }
  }

  for (TaskGraphNode *Node : Graph.leaves())
    for (auto [I, S] : llvm::enumerate(argsOf(Node)))
      if (auto *A = dyn_cast<TaskGraphAttachArgTy>(&S.get()))
        if (A->Enabled && CopiedBack[A->PtrRange->Id])
          std::memcpy(argBase(Node, static_cast<uint32_t>(I)),
                      Plan.AttachHostBytes.data() + A->StagingOffset, A->Size);
}

/// Whether \p Node moves bytes into or out of the range numbered \p Range in
/// the replay being acquired, whose enablement has been settled.
bool TaskGraphPlanner::copiesThisReplay(TaskGraphNode *Node, uint32_t Range) {
  for (const TaskGraphArgSlotTy &S : argsOf(Node)) {
    if (const auto *A = dyn_cast<TaskGraphMappedArgTy>(&S.get())) {
      if (A->Range->Id != Range)
        continue;
      if (A->In.Requested && A->In.Enabled)
        return true;
      if (A->Out.Requested && A->Out.Enabled)
        return true;
    } else if (const auto *T = dyn_cast<TaskGraphAttachArgTy>(&S.get())) {
      if (T->PtrRange->Id != Range)
        continue;
      if (T->Enabled)
        return true;
    }
  }
  return false;
}

/// How many argument entries a leaf's plan has: one per recorded map argument,
/// for the leaves that carry map arrays.
static uint32_t numPlanArgs(const TaskGraphNode *Node) {
  if (auto *TargetNode = dyn_cast<TaskGraphTargetNode>(Node)) {
    auto *KA = static_cast<KernelArgsTy *>(TargetNode->KernelArgs);
    return KA ? KA->NumArgs : 0;
  }
  if (auto *DataNode = dyn_cast<TaskGraphTargetDataNode>(Node))
    return DataNode->ArgNum;
  return 0;
}

/// Lay out every leaf's operations, in the order the leaf performs them, and
/// every target leaf's kernel parameters.  Returns false for a kernel argument
/// that has no place in the plan, which should not happen.
bool TaskGraphPlanner::buildOpsAndParams(TaskGraphTy &Graph) {
  TaskGraphDataPlanTy &Plan = Graph.Plan;
  using OpKindTy = TaskGraphOpTy::KindTy;
  using ArgKindTy = TaskGraphArgBaseTy::KindTy;
  using ParamKindTy = TaskGraphKernelParamTy::KindTy;

  // Built end to end, and only then pointed at, since the vectors grow.
  SmallVector<std::pair<size_t, size_t>, 16> OpSpans, ParamSpans;
  for (TaskGraphNode *Node : Graph.leaves()) {
    const size_t FirstOp = Plan.Ops.size();
    const size_t FirstParam = Plan.KernelParams.size();
    ArrayRef<TaskGraphArgSlotTy> Args = argsOf(Node);
    auto each = [&](ArgKindTy Kind, auto &&Fn) {
      for (auto [I, S] : llvm::enumerate(Args)) {
        if (const auto *A = dyn_cast<TaskGraphArgBaseTy>(&S.get()))
          if (A->Kind == Kind)
            Fn(static_cast<uint32_t>(I), S);
      }
    };
    auto copies = [&](bool In) {
      each(ArgKindTy::Mapped, [&](uint32_t I, const TaskGraphArgSlotTy &S) {
        const auto *M = cast<TaskGraphMappedArgTy>(&S.get());
        if (In ? M->In.Requested : M->Out.Requested)
          Plan.Ops.push_back({In ? OpKindTy::CopyIn : OpKindTy::CopyOut, I});
      });
    };
    // Attachments follow the leaf's mapping and precede anything following the
    // pointer; so does filling privatised storage, a private pointer
    // initialised to a device address being an attachment in all but name.
    auto attachments = [&]() {
      each(ArgKindTy::Attach, [&](uint32_t I, const TaskGraphArgSlotTy &) {
        Plan.Ops.push_back({OpKindTy::Attach, I});
      });
      each(ArgKindTy::FirstPrivate,
           [&](uint32_t I, const TaskGraphArgSlotTy &) {
             Plan.Ops.push_back({OpKindTy::PrivateInit, I});
           });
    };

    if (isa<TaskGraphTargetDataNode>(Node)) {
      each(ArgKindTy::Mapped, [&](uint32_t I, const TaskGraphArgSlotTy &S) {
        const auto *M = cast<TaskGraphMappedArgTy>(&S.get());
        if (M->In.Requested)
          Plan.Ops.push_back({OpKindTy::CopyIn, I});
        if (M->Out.Requested)
          Plan.Ops.push_back({OpKindTy::CopyOut, I});
      });
      attachments();
    } else {
      auto *Target = cast<TaskGraphTargetNode>(Node);
      copies(/*In=*/true);
      attachments();
      Plan.Ops.push_back({OpKindTy::Launch, 0});
      copies(/*In=*/false);

      // The host appends the launch environment after the last real target
      // parameter, which finalize has checked the recording to be new enough
      // to do.  What remains are the kernel's parameters, in order.
      auto *KA = static_cast<KernelArgsTy *>(Target->KernelArgs);
      int32_t LaunchEnvironment = -1;
      for (uint32_t I = 0; I < KA->NumArgs; ++I)
        if (mt::isTargetParam(KA->ArgTypes[I]))
          LaunchEnvironment = static_cast<int32_t>(I);
      for (uint32_t I = 0; I < KA->NumArgs; ++I) {
        if (!mt::isTargetParam(KA->ArgTypes[I]) ||
            static_cast<int32_t>(I) == LaunchEnvironment)
          continue;
        const TaskGraphArgSlotTy &S = Args[I];
        const TaskGraphArgBaseTy *A = &S.get();
        switch (A->getKind()) {
        case ArgKindTy::Literal:
          Plan.KernelParams.push_back({ParamKindTy::Literal, I, 0});
          break;
        case ArgKindTy::FirstPrivate: {
          const auto *F = cast<TaskGraphFirstPrivateArgTy>(A);
          Plan.KernelParams.push_back({ParamKindTy::Private, I, F->BaseOffset});
          break;
        }
        case ArgKindTy::Mapped: {
          char *Base = static_cast<char *>(KA->ArgBasePtrs[I]);
          const auto *M = cast<TaskGraphMappedArgTy>(A);
          // A pointer-and-pointee entry records the address of the pointer as
          // its base, and the kernel is handed what the pointer holds, as in
          // targetKernel.
          if (mt::isPtrAndObj(KA->ArgTypes[I]))
            Base = *reinterpret_cast<char **>(Base);
          Plan.KernelParams.push_back(
              {ParamKindTy::Mapped, I,
               static_cast<int64_t>(M->Offset) +
                   (Base - static_cast<char *>(KA->ArgPtrs[I]))});
          break;
        }
        case ArgKindTy::None:
        case ArgKindTy::Attach:
        case ArgKindTy::NoTargetAttach:
          assert(false && "kernel parameter with no place in the plan");
          return false;
        }
      }
      if (LaunchEnvironment != -1)
        Plan.KernelParams.push_back({ParamKindTy::LaunchEnvironment, 0, 0});
    }
    OpSpans.push_back({FirstOp, Plan.Ops.size() - FirstOp});
    ParamSpans.push_back({FirstParam, Plan.KernelParams.size() - FirstParam});
  }

  for (auto [Node, Ops, Params] :
       llvm::zip_equal(Graph.leaves(), OpSpans, ParamSpans)) {
    Node->OpPlan = Plan.Ops.data() + Ops.first;
    Node->NumOps = Ops.second;
    if (auto *Target = dyn_cast<TaskGraphTargetNode>(Node)) {
      Target->Params = Plan.KernelParams.data() + Params.first;
      Target->NumParams = Params.second;
    }
  }
  return true;
}

bool TaskGraphPlanner::buildTaskGraphPlan(DeviceTy &Device,
                                          TaskGraphTy &Graph) {
  TaskGraphDataPlanTy &Plan = Graph.Plan;

  // One entry per recorded argument of every leaf, in a single allocation that
  // is never resized, so the leaves can point into it.
  size_t NumArgs = 0;
  for (TaskGraphNode *Node : Graph.leaves())
    NumArgs += numPlanArgs(Node);
  Plan.Args.assign(NumArgs, TaskGraphArgSlotTy{});
  NumArgs = 0;
  for (TaskGraphNode *Node : Graph.leaves()) {
    Node->ArgPlan = Plan.Args.data() + NumArgs;
    Node->NumArgPlans = numPlanArgs(Node);
    NumArgs += Node->NumArgPlans;
  }

  // Classify every argument and collect the maps in encounter order, including
  // maps that move no data: `alloc` and `release` still decide how long a
  // range's mapping lives.  Whether a range is mapped is decided per replay.
  std::vector<ClaimTy> Claims;
  // A graph the plan cannot express is replayed on the host.  Say why, since
  // it is otherwise silent.  There is no taskgraph info category, so this
  // shows under any of them.
  auto decline = [&](uint32_t Leaf, uint32_t Arg, void *Begin,
                     const char *Why) {
    INFO(OMP_INFOTYPE_ALL, Device.DeviceID,
         "taskgraph %" PRIu64 " replays on the host: argument %" PRIu32
         " of leaf %" PRIu32 " (" DPxMOD
         ") %s, which the graph cannot lower onto the device\n",
         static_cast<uint64_t>(Graph.GraphId), Arg, Leaf, DPxPTR(Begin), Why);
    return false;
  };
  // An attachment's pointer and pointee are mapped by other maps, so both are
  // looked up once the ranges are settled.
  auto noteAttach = [&](TaskGraphArgSlotTy &S, uint64_t Size) {
    auto &A = S.emplace<TaskGraphAttachArgTy>();
    A.Kind = TaskGraphArgBaseTy::KindTy::Attach;
    A.Size = Size;
    A.StagingOffset = Plan.AttachStaging.size();
    Plan.AttachStaging.resize(A.StagingOffset + Size);
    Plan.AttachHostBytes.resize(A.StagingOffset + Size);
  };
  // A privatised argument gets a slot in the private block, laid out here and
  // allocated once the whole graph has been walked.  The staging bytes share
  // the block's layout.
  auto notePrivate = [&](TaskGraphArgSlotTy &S, void *HstPtrBegin,
                         void *HstPtrBase, uint64_t Size, bool Initialized,
                         bool IsSaved) {
    uint64_t StagingOffset =
        alignTo(Plan.PrivateStaging.size(), privateAlignment(HstPtrBegin));
    uint64_t BaseOffset =
        static_cast<char *>(HstPtrBase) - static_cast<char *>(HstPtrBegin);
    if (Initialized) {
      auto &F = S.emplace<TaskGraphFirstPrivateArgTy>();
      F.Size = Size;
      F.StagingOffset = StagingOffset;
      F.BaseOffset = BaseOffset;
      F.Saved = IsSaved;
    }
    Plan.PrivateStaging.resize(StagingOffset + Size);
    Plan.PrivateHostBytes.resize(StagingOffset + Size);
    return true;
  };
  auto noteClaim = [&](MutableArrayRef<TaskGraphArgSlotTy> Args, uint32_t I,
                       void *Begin, uint64_t Size, int64_t ArgType,
                       ClaimKindTy Kind) -> TaskGraphMappedArgTy & {
    TaskGraphArgSlotTy &S = Args[I];
    // An entry of no extent, such as `map(p[0:0])`, only looks memory up and
    // takes no part in the reference count.
    if (!Size)
      Kind = ClaimKindTy::Lookup;
    auto &M = S.emplace<TaskGraphMappedArgTy>();
    M.Size = Size;
    // A struct's entry precedes its members', so it has been classified.
    const TaskGraphArgSlotTy *Parent = nullptr;
    const int64_t P = structParentIndex(ArgType);
    if (Kind != ClaimKindTy::Lookup && P >= 0 && P < I &&
        isa<TaskGraphMappedArgTy>(Args[P].get()))
      Parent = &Args[P];
    Claims.push_back(
        ClaimTy{Kind, &S, static_cast<char *>(Begin), Size, ArgType, Parent});
    return M;
  };
  // A map's copies.  `alloc`, `release` and `delete` move no bytes.
  auto noteCopies = [](TaskGraphMappedArgTy &M, mt::TransferDir Dir) {
    if (!M.Size || Dir == mt::TransferDir::None)
      return;
    M.In.Requested =
        Dir == mt::TransferDir::H2D || Dir == mt::TransferDir::Both;
    M.Out.Requested =
        Dir == mt::TransferDir::D2H || Dir == mt::TransferDir::Both;
  };

  for (uint32_t L = 0; L < Graph.leaves().size(); ++L) {
    TaskGraphNode *Node = Graph.leaves()[L];
    MutableArrayRef<TaskGraphArgSlotTy> Args = argsOf(Node);

    if (auto *DataNode = dyn_cast<TaskGraphTargetDataNode>(Node)) {
      // `target update` transfers whoever mapped the region; enter and exit
      // maps transfer only on the reference-count change that owns the copy.
      const bool IsUpdate = isa<TaskGraphTargetUpdateDataNode>(DataNode);
      const bool IsEnter = isa<TaskGraphTargetEnterDataNode>(DataNode);
      const bool IsExit = isa<TaskGraphTargetExitDataNode>(DataNode);
      for (int32_t I = 0; I < DataNode->ArgNum; ++I) {
        TaskGraphArgSlotTy &S = Args[I];
        const int64_t ArgType = DataNode->ArgTypes[I];
        // A data leaf has no kernel to hand privatised storage to, so the
        // runtime's data paths skip these entries too.
        if (mt::isLiteral(ArgType) || isPrivatised(ArgType))
          continue;
        if (isAttachment(ArgType)) {
          noteAttach(S, static_cast<uint64_t>(DataNode->ArgSizes[I]));
          continue;
        }
        if (const char *Why = whyUnlowerable(
                ArgType,
                DataNode->ArgMappers ? DataNode->ArgMappers[I] : nullptr))
          return decline(L, I, DataNode->Args[I], Why);
        auto &M =
            noteClaim(Args, I, DataNode->Args[I],
                      static_cast<uint64_t>(DataNode->ArgSizes[I]), ArgType,
                      IsEnter  ? ClaimKindTy::Enter
                      : IsExit ? ClaimKindTy::Exit
                               : ClaimKindTy::Lookup);
        noteCopies(M, directionFor(DataNode, ArgType));
        M.Always = IsUpdate || mt::isAlways(ArgType);
      }
      continue;
    }

    auto *TargetNode = cast<TaskGraphTargetNode>(Node);
    auto *KA = static_cast<KernelArgsTy *>(TargetNode->KernelArgs);
    if (!KA)
      return false;

    for (uint32_t I = 0; I < KA->NumArgs; ++I) {
      TaskGraphArgSlotTy &S = Args[I];
      const int64_t ArgType = KA->ArgTypes[I];
      if (isLambdaCaptureFixup(ArgType))
        return decline(L, I, KA->ArgPtrs[I],
                       "is a lambda's by-reference capture");
      if (mt::isLiteral(ArgType)) {
        // Pass-by-value: the argument is the value, not an address.
        if (mt::isTargetParam(ArgType)) {
          uint64_t Val = reinterpret_cast<uint64_t>(KA->ArgBasePtrs[I]);
          // Apparently only 'Size' bytes of the literal are valid data, so
          // we must mask the rest off.)
          uint64_t BitSize = KA->ArgSizes[I] * 8;
          assert(BitSize <= 64);
          uint64_t Mask =
              BitSize == 64 ? ~(uint64_t)0 : (((uint64_t)1 << BitSize) - 1);
          S.emplace<TaskGraphLiteralArgTy>(Val & Mask);
        }
        continue;
      }
      if (isPrivatised(ArgType)) {
        // We aren't expecting to see a 'PRIVATE | ATTACH' entry from current
        // Clang.  If we do see one, decline: if we need this later for some
        // reason, we can add a new argument type for it.
        if (mt::isAttach(ArgType))
          return decline(L, I, KA->ArgPtrs[I],
                         "unexpected mapping-bit encoding 'PRIVATE | ATTACH'");
        notePrivate(S, /*HstPtrBegin=*/KA->ArgPtrs[I],
                    /*HstPtrBase=*/KA->ArgBasePtrs[I],
                    static_cast<uint64_t>(KA->ArgSizes[I]),
                    /*Initialized=*/mt::isTo(ArgType), mt::isSaved(ArgType));
        continue;
      }
      if (isAttachment(ArgType)) {
        noteAttach(S, static_cast<uint64_t>(KA->ArgSizes[I]));
        continue;
      }
      if (const char *Why = whyUnlowerable(
              ArgType, KA->ArgMappers ? KA->ArgMappers[I] : nullptr))
        return decline(L, I, KA->ArgPtrs[I], Why);

      // A `target` region's map balances within the region, but may create
      // the mapping.
      auto &M = noteClaim(Args, I, KA->ArgPtrs[I],
                          static_cast<uint64_t>(KA->ArgSizes[I]), ArgType,
                          ClaimKindTy::Balanced);
      noteCopies(M, mt::transferDir(ArgType));
      M.Always = mt::isAlways(ArgType);
    }
  }

  // Gather the maps into ranges, then lay the ranges out in address order,
  // which numbers them.
  RangeMapTy Gathered;
  for (const ClaimTy &C : Claims)
    gatherClaim(Gathered, C);
  Plan.Ranges.reserve(Gathered.size());
  for (auto &[Key, Range] : Gathered) {
    Range.Id = Plan.Ranges.size();
    Plan.Ranges.push_back(Range);
  }

  // Place every argument in the range holding it.  A map always finds one;
  // a lookup finds one only if some map in the graph covers the memory.
  for (TaskGraphNode *Node : Graph.leaves()) {
    for (auto [I, S] : llvm::enumerate(argsOf(Node))) {
      const uint32_t Arg = static_cast<uint32_t>(I);
      if (auto *M = dyn_cast<TaskGraphMappedArgTy>(&S.get())) {
        char *Begin = argBegin(Node, Arg);
        M->Range = containingRange(Plan.Ranges, Begin, M->Size);
        assert(M->Range && "a gathered map outside every range");
        M->Offset = Begin - M->Range->begin();
      } else if (auto *A = dyn_cast<TaskGraphAttachArgTy>(&S.get())) {
        // The pointee is looked up with zero extent, since its own map says
        // how much of it there is.
        char *Pointer = argBase(Node, Arg);
        char *Pointee = argBegin(Node, Arg);
        A->PtrRange = containingRange(Plan.Ranges, Pointer, A->Size);
        A->PteeRange = containingRange(Plan.Ranges, Pointee, 0);
        // An attachment missing either end is dropped, as the runtime drops it
        // when a lookup comes back absent.  The usual case is `map(p[0:n])` on
        // a plain pointer, where `p` is privatised and has no device copy.
        if (!A->PtrRange || !A->PteeRange) {
          S.emplace<TaskGraphNoTargetAttachArgTy>();
        } else {
          A->Offset = Pointer - A->PtrRange->begin();
          A->PteeOffset = Pointee - A->PteeRange->begin();
        }
      }
    }
  }

  if (!buildOpsAndParams(Graph)) {
    INFO(OMP_INFOTYPE_ALL, Device.DeviceID,
         "taskgraph %" PRIu64 " replays on the host: a kernel argument has no "
         "place in its plan\n",
         static_cast<uint64_t>(Graph.GraphId));
    return false;
  }

  // Decide which map over each range performs its implicit copies.  Outside a
  // graph a map copies in on the reference-count increment that creates the
  // mapping and out on the decrement that removes it, so `target enter data
  // map(alloc:)` followed by a `target` region mapping the same data `tofrom`
  // copies in neither direction.  Walk the recorded maps in order with a
  // simulated count starting at zero; a range the graph did not create is
  // borrowed, and its implicit copies are switched off per replay instead --
  // except the copy out of an exit that drops the borrowed mapping's last
  // reference, which a replay can only tell once it finds the mapping.
  SmallVector<int32_t, 8> Count(Plan.Ranges.size(), 0);
  SmallVector<bool, 8> Borrowed(Plan.Ranges.size(), false);
  for (const ClaimTy &C : Claims) {
    auto *Arg = cast<TaskGraphMappedArgTy>(&C.Arg->get());
    // A struct member copies on its struct's count change and makes none of
    // its own.  The runtime processes an exit's members before their struct,
    // but that is the same decrement to zero.
    if (C.Parent) {
      auto *Parent = cast<TaskGraphMappedArgTy>(&C.Parent->get());
      Arg->In.Perimeter = Parent->In.Perimeter;
      Arg->Out.Perimeter = Parent->Out.Perimeter;
      Arg->OwnsBorrowedOut = Parent->OwnsBorrowedOut;
      continue;
    }
    int32_t &N = Count[Arg->Range->Id];
    switch (C.Kind) {
    case ClaimKindTy::Balanced:
      // Entry and exit of one region, with nothing in between.
      if (++N == 1)
        Arg->In.Perimeter = true;
      if (--N == 0)
        Arg->Out.Perimeter = true;
      break;
    case ClaimKindTy::Enter:
      if (++N == 1)
        Arg->In.Perimeter = true;
      break;
    case ClaimKindTy::Exit:
      // The copy-out happens on the decrement to zero.
      if (N == 1)
        Arg->Out.Perimeter = true;
      else if (N == 0 && !Borrowed[Arg->Range->Id])
        Arg->OwnsBorrowedOut = Borrowed[Arg->Range->Id] = true;
      N = std::max(0, N - 1);
      break;
    case ClaimKindTy::Lookup:
      break;
    }
  }

  // Outside a graph, whichever construct maps a range first copies it, and a
  // construct that finds the range mapped waits for that copy to land, so a
  // map is atomic as OpenMP requires.  Here the walk above settled who copies
  // in leaf order, and only the region tree orders one leaf against another:
  // a leaf the tree leaves unordered with a range's copier could use the range
  // mid-copy, or before it.  Record each such pair, so that a replay in which
  // the copy does run is declined to the host rather than ordered.  A replay
  // that finds the range mapped outside the graph copies nothing, and keeps the
  // device graph.
  //
  // Two leaves that each copy the range in for themselves and write nothing
  // back are exempt: each kernel follows its own copy, and the other's copy
  // writes the same bytes, since the host data cannot change meanwhile without
  // a race in the program.  That is a shared input read by concurrent tasks.
  {
    struct UseTy {
      uint32_t Leaf;
      /// Some argument of the leaf moves bytes into or out of the range.
      bool Copies = false;
      /// It copies the range in itself, as the owner of the copy.
      bool OwnsIn = false;
      /// It moves bytes some other way: a copy out, an attachment, or a copy
      /// regardless of ownership.
      bool Writes = false;
    };
    std::vector<SmallVector<UseTy, 2>> Uses(Plan.Ranges.size());
    for (uint32_t L = 0; L < Graph.leaves().size(); ++L)
      for (const TaskGraphArgSlotTy &S : argsOf(Graph.leaves()[L])) {
        if (const auto *A = dyn_cast<TaskGraphAttachArgTy>(&S.get())) {
          SmallVector<UseTy, 2> &RangeUses = Uses[A->PtrRange->Id];
          if (RangeUses.empty() || RangeUses.back().Leaf != L)
            RangeUses.push_back({L});
          UseTy &Use = RangeUses.back();
          // An attachment's write is unconditional.
          Use.Copies = true;
          Use.Writes = true;
        } else if (const auto *M = dyn_cast<TaskGraphMappedArgTy>(&S.get())) {
          SmallVector<UseTy, 2> &RangeUses = Uses[M->Range->Id];
          if (RangeUses.empty() || RangeUses.back().Leaf != L)
            RangeUses.push_back({L});
          UseTy &Use = RangeUses.back();
          const bool Forced =
              (M->In.Requested || M->Out.Requested) && M->Always;
          const bool OwnsIn = M->In.Requested && M->In.Perimeter;
          const bool OwnsOut =
              M->Out.Requested && (M->Out.Perimeter || M->OwnsBorrowedOut);
          Use.Copies |= Forced || OwnsIn || OwnsOut;
          Use.OwnsIn |= OwnsIn;
          Use.Writes |= Forced || OwnsOut;
        }
      }
    auto readsOwnCopy = [](const UseTy &Use) {
      return Use.OwnsIn && !Use.Writes;
    };

    for (const TaskGraphRangeTy &Range : Plan.Ranges) {
      ArrayRef<UseTy> RangeUses = Uses[Range.Id];
      for (const UseTy &Copier : RangeUses) {
        if (!Copier.Copies)
          continue;
        // Every copier gets its own entry, since a replay may run the copies of
        // one and not the other's.
        for (const UseTy &Other : RangeUses) {
          if (Other.Leaf == Copier.Leaf)
            continue;
          if ((readsOwnCopy(Copier) && readsOwnCopy(Other)) ||
              areOrdered(*Graph.leaves()[Copier.Leaf],
                         *Graph.leaves()[Other.Leaf]))
            continue;
          // One unordered user is enough to name.
          Plan.CopyConflicts.push_back({Range.Id, Copier.Leaf, Other.Leaf});
          break;
        }
      }
    }
  }

  // Decide which ranges the graph owns outright.  A range the graph creates,
  // whose maps balance, and that nothing outside the graph has mapped needs no
  // mapping-table entry, so each replay gives it storage of its own.  The
  // decision is fixed here, against the mapping state the recording left.  A
  // range that is only partly mapped declines the graph.  Device addresses stay
  // null until the first replay resolves them.
  MappingInfoTy &MappingInfo = Device.getMappingInfo();
  {
    auto HDTTMap = MappingInfo.HostDataToTargetMap.getExclusiveAccessor();
    for (TaskGraphRangeTy &Range : Plan.Ranges) {
      if (!Range.Size)
        continue;
      LookupResult LR = MappingInfo.lookupMapping(
          HDTTMap, Range.HostBegin, static_cast<int64_t>(Range.Size));
      const bool Mapped = LR.Flags.IsContained;
      const bool Partial = LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter;
      LR.TPR.reset();
      if (Partial) {
        MESSAGE("taskgraph %" PRIu64 " replays on the host: [" DPxMOD
                ", " DPxMOD ") is only partly mapped outside the graph",
                static_cast<uint64_t>(Graph.GraphId), DPxPTR(Range.HostBegin),
                DPxPTR(Range.end()));
        return false;
      }
      Range.SelfContained = !Mapped && Range.MayCreate && !Range.RefDelta &&
                            !Range.Deleted &&
                            !hostResidentIfUnmapped(Device, Range);
    }
  }

  // A device pointer that outlives the replay must not point into storage that
  // does not, so an attachment whose pointer is not self-contained demotes its
  // pointee.  That can demote the pointer of another attachment, hence the
  // fixpoint.
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (TaskGraphArgSlotTy &S : Plan.Args)
      if (auto *A = dyn_cast<TaskGraphAttachArgTy>(&S.get()))
        if (A->PteeRange->SelfContained && !A->PtrRange->SelfContained) {
          A->PteeRange->SelfContained = false;
          Changed = true;
        }
  }

  Plan.Epoch = MappingInfo.getEpoch();

  // Private storage is not in the device data environment, so it is allocated
  // once for the graph, at an address lowered kernel arguments can name.
  if (!Plan.PrivateStaging.empty()) {
    Plan.PrivateBlock = Device.allocData(Plan.PrivateStaging.size());
    if (!Plan.PrivateBlock) {
      REPORT() << "taskgraph could not allocate " << Plan.PrivateStaging.size()
               << " bytes of device memory for its privatised arguments";
      return false;
    }
  }
  return true;
}

bool TaskGraphPlanner::acquireTaskGraphMappings(
    DeviceTy &Device, TaskGraphTy &Graph,
    SmallVectorImpl<TaskGraphArgUpdateTy> &Updates, bool &Resync) {
  Updates.clear();
  TaskGraphDataPlanTy &Plan = Graph.Plan;
  // libomp holds the record's map_lock for the whole of a replay, so a graph is
  // never acquired twice at once.
  assert(!Plan.MappingsHeld && "taskgraph replay entered twice");
  Resync = Plan.Resync;
  Plan.Resync = true;
  Plan.DeclineReported = false;

  // Relocation has already rewritten the recorded maps, so move the ranges
  // before resolving them against the mapping table.
  SmallVector<bool, 8> HostMoved(Plan.Ranges.size(), false);
  if (!rekeyRanges(Graph, HostMoved))
    return false;

  // The epoch baseline is taken when the previous replay released its
  // mappings, so the graph's own mapping and unmapping do not count as drift.
  const bool Moved = llvm::is_contained(HostMoved, true);
  if (reportDrift() &&
      (Moved || Device.getMappingInfo().getEpoch() != Plan.Epoch))
    fprintf(stderr, "taskgraph: %s, re-resolving ranges\n",
            Moved ? "relocated host captures" : "mapping drift");

  // Set before acquiring anything, so that the ranges acquired before a failure
  // are released.
  Plan.MappingsHeld = true;

  SmallVector<bool, 8> DeviceMoved(Plan.Ranges.size(), false);
  SmallVector<uint8_t, 16> Changes(Plan.Args.size(), 0);

  for (TaskGraphRangeTy &Range : Plan.Ranges) {
    void *Address = nullptr;
    bool Inserted = false;
    bool HostResident = false;
    if (!acquireRange(Device, Range, Address, Inserted, HostResident))
      return false;

    DeviceMoved[Range.Id] = Range.DevicePtr != Address;
    Range.DevicePtr = Address;
    Range.PlanInserted = Inserted;
    Range.HostResident = HostResident;
  }

  // Borrowing a self-contained range can recreate the shape the build ruled
  // out: an attachment writing into a mapping that outlives the replay the
  // address of storage that does not.  Decline such a replay.
  for (const TaskGraphArgSlotTy &S : Plan.Args)
    if (auto *A = dyn_cast<TaskGraphAttachArgTy>(&S.get()))
      if (A->PteeRange && A->PtrRange->SelfContained && A->PtrRange->Held &&
          A->PteeRange->StorageHeld) {
        MESSAGE("taskgraph borrows [" DPxMOD ", " DPxMOD
                ") from a mapping outside the graph, which would be left "
                "pointing at device memory only this replay keeps",
                DPxPTR(A->PtrRange->HostBegin), DPxPTR(A->PtrRange->end()));
        return false;
      }

  // An implicit transfer runs only if the graph created this replay's mapping
  // (PlanInserted) and this map is the one whose count change performs it
  // (OwnsIn or OwnsOut), or the graph removes a mapping it found (DropsLast)
  // and this is the exit that drops it (OwnsBorrowedOut).  `always` and
  // `target update` transfer regardless.  A host-resident range has nothing to
  // copy.  A toggle is reported when the resulting enablement changes, not
  // when its inputs do.
  for (TaskGraphArgSlotTy &S : Plan.Args) {
    if (auto *M = dyn_cast<TaskGraphMappedArgTy>(&S.get())) {
      const TaskGraphRangeTy &Range = *M->Range;
      const bool Resolved = !Range.HostResident && Range.DevicePtr;
      auto enable = [&](bool Copies, bool Owns, bool &Enabled) {
        if (!Copies)
          return;
        const bool Was = Enabled;
        Enabled = Resolved && (Owns || M->Always);
        if (Enabled != Was)
          Changes[Plan.argIndex(S)] |= TaskGraphArgUpdateTy::EnabledChanged;
      };
      enable(M->In.Requested, Range.PlanInserted && M->In.Perimeter,
             M->In.Enabled);
      enable(M->Out.Requested,
             (Range.PlanInserted && M->Out.Perimeter) ||
                 (Range.DropsLast && M->OwnsBorrowedOut),
             M->Out.Enabled);
    }
  }

  // An attachment depends on its pointee's range as well as its pointer's, so
  // its enablement is compared after staging.
  SmallVector<bool, 8> WasAttached(Plan.Args.size());
  for (uint32_t I = 0; I < Plan.Args.size(); ++I)
    if (const auto *A = dyn_cast<TaskGraphAttachArgTy>(&Plan.Args[I].get()))
      WasAttached[I] = A->Enabled;

  if (!stageTaskGraphAttachments(Device, Graph, Changes))
    return false;

  // A range a leaf copies while a leaf nothing orders with it uses the range
  // has to go to the host, but only when the copy runs: a range mapped outside
  // the graph has none of its implicit copies this replay.
  for (const TaskGraphDataPlanTy::CopyConflictTy &C : Plan.CopyConflicts) {
    if (!copiesThisReplay(Graph.leaves()[C.Copier], C.Range))
      continue;
    // Reported when it starts happening, not on every replay it goes on for,
    // and only under LIBOMPTARGET_INFO: unlike the other declines here, this
    // one is the plan working as intended.
    Plan.DeclineReported = true;
    if (!Plan.ConflictDeclined) {
      const TaskGraphRangeTy &Range = Plan.Ranges[C.Range];
      INFO(OMP_INFOTYPE_ALL, Device.DeviceID,
           "taskgraph %" PRIu64 " replays on the host: leaf %" PRIu32
           " copies [" DPxMOD ", " DPxMOD ") and leaf %" PRIu32
           " uses it, and nothing orders the two\n",
           static_cast<uint64_t>(Graph.GraphId), C.Copier,
           DPxPTR(Range.HostBegin), DPxPTR(Range.end()), C.User);
    }
    Plan.ConflictDeclined = true;
    return false;
  }
  Plan.ConflictDeclined = false;

  for (TaskGraphNode *Node : Graph.leaves()) {
    for (auto [I, Slot] : llvm::enumerate(argsOf(Node))) {
      auto *Arg = &Slot.get();
      uint8_t &Change = Changes[Plan.argIndex(Slot)];
      if (const auto *M = dyn_cast<TaskGraphMappedArgTy>(Arg)) {
        if (DeviceMoved[M->Range->Id])
          Change |= TaskGraphArgUpdateTy::DeviceAddressChanged;
        if (HostMoved[M->Range->Id])
          Change |= TaskGraphArgUpdateTy::HostAddressChanged;
      } else if (const auto *A = dyn_cast<TaskGraphAttachArgTy>(Arg)) {
        if (DeviceMoved[A->PtrRange->Id])
          Change |= TaskGraphArgUpdateTy::DeviceAddressChanged;
        if (A->Enabled != WasAttached[Plan.argIndex(Slot)])
          Change |= TaskGraphArgUpdateTy::EnabledChanged;
      } else if (const auto *F = dyn_cast<TaskGraphFirstPrivateArgTy>(Arg)) {
        if (F->InitPending)
          Change |= TaskGraphArgUpdateTy::DataChanged;
      } else if (auto *L = dyn_cast<TaskGraphLiteralArgTy>(Arg)) {
        KernelArgsTy *KernArgs = static_cast<KernelArgsTy *>(
            cast<TaskGraphTargetNode>(Node)->KernelArgs);
        // Relocation has re-read the variable into the recorded slot.
        uint64_t Value = reinterpret_cast<uint64_t>(KernArgs->ArgBasePtrs[I]);
        uint64_t BitSize = KernArgs->ArgSizes[I] * 8;
        assert(BitSize <= 64);
        uint64_t Mask =
            BitSize == 64 ? ~(uint64_t)0 : (((uint64_t)1 << BitSize) - 1);
        Value &= Mask;
        if (Value != L->LiteralValue)
          Change |= TaskGraphArgUpdateTy::DataChanged;
        L->LiteralValue = Value;
      }
    }
  }

  // Report only the entries that changed.  A resync makes the backend re-read
  // everything, so it needs no list.
  for (auto [Index, Change] : llvm::enumerate(Changes))
    if (Change)
      Updates.push_back({static_cast<uint32_t>(Index), Change});
  return true;
}

void TaskGraphPlanner::commitTaskGraphUpdates(TaskGraphTy &Graph) {
  Graph.Plan.Resync = false;
}

void TaskGraphPlanner::releaseTaskGraphMappings(DeviceTy &Device,
                                                TaskGraphTy &Graph,
                                                bool Replayed) {
  if (!Graph.Plan.MappingsHeld)
    return;
  Graph.Plan.MappingsHeld = false;
  // A declined replay may not have reached staging, so it has neither host
  // bytes to restore nor private copies the backend has performed.
  if (Replayed) {
    restoreTaskGraphAttachments(Graph);
    for (TaskGraphArgSlotTy &S : Graph.Plan.Args)
      if (auto *A = dyn_cast<TaskGraphFirstPrivateArgTy>(&S.get()))
        A->InitPending = false;
  }
  for (TaskGraphRangeTy &Range : Graph.Plan.Ranges)
    releaseRange(Device, Range);
  // Taken after the graph's own unmapping, so only the program's changes count
  // as drift.
  Graph.Plan.Epoch = Device.getMappingInfo().getEpoch();
}

void TaskGraphPlanner::releaseTaskGraphPlan(DeviceTy &Device,
                                            TaskGraphTy &Graph) {
  // A graph is never destroyed mid-replay, but hold references left behind
  // could never be released, so do not rely on it.
  releaseTaskGraphMappings(Device, Graph);
  for (TaskGraphRangeTy &Range : Graph.Plan.Ranges)
    Range.DevicePtr = nullptr;

  if (Graph.Plan.PrivateBlock) {
    if (Device.deleteData(Graph.Plan.PrivateBlock) != OFFLOAD_SUCCESS)
      REPORT() << "taskgraph could not release the device memory holding its "
                  "privatised arguments";
    Graph.Plan.PrivateBlock = nullptr;
  }
}

bool llvm::omp::target::buildTaskGraphPlan(DeviceTy &Device,
                                           TaskGraphTy &Graph) {
  return TaskGraphPlanner::buildTaskGraphPlan(Device, Graph);
}

bool llvm::omp::target::acquireTaskGraphMappings(
    DeviceTy &Device, TaskGraphTy &Graph,
    SmallVectorImpl<TaskGraphArgUpdateTy> &Updates, bool &Resync) {
  return TaskGraphPlanner::acquireTaskGraphMappings(Device, Graph, Updates,
                                                    Resync);
}

void llvm::omp::target::commitTaskGraphUpdates(TaskGraphTy &Graph) {
  TaskGraphPlanner::commitTaskGraphUpdates(Graph);
}

void llvm::omp::target::releaseTaskGraphMappings(DeviceTy &Device,
                                                 TaskGraphTy &Graph,
                                                 bool Replayed) {
  TaskGraphPlanner::releaseTaskGraphMappings(Device, Graph, Replayed);
}

void llvm::omp::target::beginTaskGraphReplay(TaskGraphTy &Graph) {
  TaskGraphPlanner::beginTaskGraphReplay(Graph);
}

void llvm::omp::target::releaseTaskGraphPlan(DeviceTy &Device,
                                             TaskGraphTy &Graph) {
  TaskGraphPlanner::releaseTaskGraphPlan(Device, Graph);
}
