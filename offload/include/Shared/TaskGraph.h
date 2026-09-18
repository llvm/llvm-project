//===-- Shared/TaskGraph.h - Processed-taskgraph object model ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The representation of a target subgraph of a processed OpenMP taskgraph --
// a subtree of its regions that holds target constructs and nothing else --
// shared between libomptarget and the device plugins.  libomp runs the rest of
// the taskgraph itself, and has libomptarget run each subgraph a plugin claims
// when the replay reaches it.
//
// The representation here is somewhat decoupled from the one used within
// libomp, partly because of the different code styles adopted by each library,
// and partly because different aspects of the representation are important when
// recording/host-replaying the graph vs. executing it on an offload device.
// Particular care has been taken over avoiding the need for strict binary
// compatibility (e.g. structure layout) between libomp and libomptarget -- the
// interface between the two is via function calls with POD argument types.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_TASKGRAPH_H
#define OMPTARGET_SHARED_TASKGRAPH_H

#include "omptarget.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace llvm {
namespace omp {
namespace target {

//===----------------------------------------------------------------------===//
// Object model (LLVM-style RTTI)
//
// libomp streams each subgraph in preorder, with each region announcing
// its child count up front (prefix notation).  We materialize it as a tree of
// the types below, laid out in a single contiguous block: see BuildCursor and
// the two-pass allocation it drives.
//
//===----------------------------------------------------------------------===//

/// Taskgraph node/region type discriminator.  Order is significant here.
enum class TGKind : uint8_t {
  // Interior regions (own an ordered child list).
  ParRegion,
  SeqRegion,
  ExclRegion,
  IrreducibleRegion,

  // Leaves.
  TargetNode,
  // Data leaves form their own sub-range at the tail of the leaf range.
  TargetEnterDataNode,
  TargetExitDataNode,
  TargetUpdateDataNode,
};

struct TaskGraphRegion;
class TaskGraphArgBaseTy;
struct TaskGraphOpTy;
struct TaskGraphKernelParamTy;
class TaskGraphPlanner;

/// Common base for every taskgraph element.
struct TaskGraphElement {
  const TGKind Kind;
  TaskGraphRegion *Parent = nullptr; // null iff this is the graph root
  uint32_t ChildIndex = 0;           // ordinal within Parent's child list

  TGKind getKind() const { return Kind; }

protected:
  explicit TaskGraphElement(TGKind K) : Kind(K) {}
};

//===----------------------------------------------------------------------===//
// Regions (interior nodes)
//===----------------------------------------------------------------------===//

/// An interior region owning an ordered list of children.  The child pointer
/// array is bump-allocated from the same block and filled in child order by
/// the build cursor; its length is known up front from the stream's prefix
/// count, so the region is fully wired before its children are streamed in.
struct TaskGraphRegion : TaskGraphElement {
  uint32_t NumChildren = 0;
  TaskGraphElement **Children = nullptr;

  llvm::MutableArrayRef<TaskGraphElement *> children() {
    return {Children, NumChildren};
  }
  llvm::ArrayRef<const TaskGraphElement *> children() const {
    return llvm::ArrayRef<TaskGraphElement *>(Children, NumChildren);
  }

  static bool classof(const TaskGraphElement *E) {
    return E->getKind() >= TGKind::ParRegion &&
           E->getKind() <= TGKind::IrreducibleRegion;
  }

protected:
  TaskGraphRegion(TGKind K, uint32_t N) : TaskGraphElement(K), NumChildren(N) {}
};

/// Parallel region (children may run concurrently).
struct TaskGraphParRegion : TaskGraphRegion {
  explicit TaskGraphParRegion(uint32_t N)
      : TaskGraphRegion(TGKind::ParRegion, N) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::ParRegion;
  }
};

/// Sequential region (children run in order).
struct TaskGraphSeqRegion : TaskGraphRegion {
  explicit TaskGraphSeqRegion(uint32_t N)
      : TaskGraphRegion(TGKind::SeqRegion, N) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::SeqRegion;
  }
};

/// Children are mutually exclusive (mutexinoutset), unordered.
struct TaskGraphExclRegion : TaskGraphRegion {
  explicit TaskGraphExclRegion(uint32_t N)
      : TaskGraphRegion(TGKind::ExclRegion, N) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::ExclRegion;
  }
};

/// An intra-region dependence edge of an irreducible region, naming children by
/// ordinal.  Both endpoints name a child: the region's sources and sinks are
/// whichever children have no incoming or no outgoing edge, same as for the
/// corresponding edges in libomp.
struct TaskGraphEdge {
  int32_t Src;
  int32_t Dst;
};

/// An irreducible region.  Dependencies between children are represented by
/// an explicit edge array (a pointer into the same allocation block as the
/// rest of the transmitted graph).
struct TaskGraphIrreducibleRegion : TaskGraphRegion {
  uint32_t NumEdges = 0;
  uint32_t EdgeFill = 0;          // transient build cursor into Edges
  TaskGraphEdge *Edges = nullptr; // len == NumEdges, into same block

  llvm::ArrayRef<TaskGraphEdge> edges() const { return {Edges, NumEdges}; }

  explicit TaskGraphIrreducibleRegion(uint32_t N)
      : TaskGraphRegion(TGKind::IrreducibleRegion, N) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::IrreducibleRegion;
  }
};

class TaskGraphRangeTy;

/// What the plan does with one argument of one leaf.
///
/// Every leaf with map arrays has one entry per recorded argument, indexed like
/// the map arrays.  An entry names its range directly, with a range-relative
/// offset, so it sees the range's current addresses without any update.
class TaskGraphArgBaseTy {
public:
  enum class KindTy : uint8_t {
    None,
    Literal,
    Mapped,
    Attach,
    NoTargetAttach,
    FirstPrivate
  };
  KindTy getKind() const { return Kind; }

protected:
  explicit TaskGraphArgBaseTy(KindTy K) : Kind(K) {}

private:
  friend class TaskGraphPlanner;
  friend class TaskGraphDataPlanTy;

  KindTy Kind = KindTy::None;
};

class TaskGraphNoneArgTy : public TaskGraphArgBaseTy {
public:
  TaskGraphNoneArgTy() : TaskGraphArgBaseTy(KindTy::None) {}

  static bool classof(const TaskGraphArgBaseTy *A) {
    return A->getKind() == KindTy::None;
  }
};

class TaskGraphLiteralArgTy : public TaskGraphArgBaseTy {
public:
  TaskGraphLiteralArgTy() : TaskGraphArgBaseTy(KindTy::Literal) {}
  TaskGraphLiteralArgTy(uint64_t Val)
      : TaskGraphArgBaseTy(KindTy::Literal), LiteralValue(Val) {}

  /// Literal kernel argument.  Refreshed by relocation.
  uint64_t LiteralValue = 0;

  static bool classof(const TaskGraphArgBaseTy *A) {
    return A->getKind() == KindTy::Literal;
  }
};

struct TaskGraphCopyTy {
  // TRUE if this copy kind was implied or present in a clause
  bool Requested = false;
  // TRUE if a static walk over the graph at finalize time showed that this is
  // the construct that should perform the copy (moving the ref count to or
  // from zero).  Only takes effect when the mapping is created by the graph,
  // i.e. not for data that is already present on the device when a replay
  // starts executing.
  bool Perimeter = false;
  // TRUE if this copy is enabled for a given replay.
  bool Enabled = true;
};

class TaskGraphMappedArgTy : public TaskGraphArgBaseTy {
public:
  TaskGraphMappedArgTy() : TaskGraphArgBaseTy(KindTy::Mapped) {}

  /// The argument lies in Range at Offset.
  TaskGraphRangeTy *Range = nullptr;
  uint64_t Offset = 0;
  uint64_t Size = 0;

  TaskGraphCopyTy In;
  TaskGraphCopyTy Out;
  bool OwnsBorrowedOut = false;
  bool Always = false;

  static bool classof(const TaskGraphArgBaseTy *A) {
    return A->getKind() == KindTy::Mapped;
  }
};

class TaskGraphAttachArgTy : public TaskGraphArgBaseTy {
public:
  TaskGraphAttachArgTy() : TaskGraphArgBaseTy(KindTy::Attach) {}
  TaskGraphAttachArgTy(TaskGraphRangeTy *PtrRange, uint64_t Offset,
                       uint64_t Size)
      : TaskGraphArgBaseTy(KindTy::Attach), PtrRange(PtrRange), Offset(Offset),
        Size(Size) {}

  /// A pointer attachment: the device copy of the pointer, at Offset in
  /// PtrRange, is made to point at the device copy of the pointee.  The bytes
  /// written come from staging the plan fills each replay.  The write is
  /// unconditional, unlike outside a graph, because the device copy of the
  /// pointer may have been made by a different replay.
  TaskGraphRangeTy *PtrRange = nullptr;
  uint64_t Offset = 0;
  uint64_t Size = 0;

  // Per-replay enabled flag.
  bool Enabled = false;

  TaskGraphRangeTy *PteeRange = nullptr;
  int64_t PteeOffset = 0;

  // Offset of attachment's bytes in AttachStaging and AttachHostBytes.
  uint64_t StagingOffset = 0;

  static bool classof(const TaskGraphArgBaseTy *A) {
    return A->getKind() == KindTy::Attach;
  }
};

// This is a no-op: keeping it as a distinct value (from e.g. "None") for
// self-documentation and debug dumps, etc.

class TaskGraphNoTargetAttachArgTy : public TaskGraphArgBaseTy {
public:
  TaskGraphNoTargetAttachArgTy() : TaskGraphArgBaseTy(KindTy::NoTargetAttach) {}

  static bool classof(const TaskGraphArgBaseTy *A) {
    return A->getKind() == KindTy::NoTargetAttach;
  }
};

class TaskGraphFirstPrivateArgTy : public TaskGraphArgBaseTy {
public:
  TaskGraphFirstPrivateArgTy() : TaskGraphArgBaseTy(KindTy::FirstPrivate) {}

  uint64_t Size = 0;

  uint64_t StagingOffset = 0;
  int64_t BaseOffset = 0;

  // Each of these is per-replay.
  bool Saved = false;
  bool Staged = false;
  bool InitPending = false;

  static bool classof(const TaskGraphArgBaseTy *A) {
    return A->getKind() == KindTy::FirstPrivate;
  }
};

/// Container for any of the argument types above -- we want to be able to store
/// them as values in an array, so we need to know the largest size.  This
/// allows us to do that.

template <typename... ArgTys> class TaskGraphArgSlotBaseTy {
  static_assert((std::is_trivially_copyable_v<ArgTys> && ...),
                "non-trivial argument type");
  llvm::AlignedCharArrayUnion<ArgTys...> Storage;

public:
  TaskGraphArgSlotBaseTy() { emplace<TaskGraphNoneArgTy>(); }
  template <typename T, typename... As> T &emplace(As &&...Args) {
    static_assert((std::is_same_v<T, ArgTys> || ...), "not a slot kind");
    return *new (Storage.buffer) T(std::forward<As>(Args)...);
  }
  TaskGraphArgBaseTy &get() {
    return *std::launder(
        reinterpret_cast<TaskGraphArgBaseTy *>(Storage.buffer));
  }
  const TaskGraphArgBaseTy &get() const {
    return *std::launder(
        reinterpret_cast<const TaskGraphArgBaseTy *>(Storage.buffer));
  }
};

using TaskGraphArgSlotTy =
    TaskGraphArgSlotBaseTy<TaskGraphNoneArgTy, TaskGraphLiteralArgTy,
                           TaskGraphMappedArgTy, TaskGraphAttachArgTy,
                           TaskGraphNoTargetAttachArgTy,
                           TaskGraphFirstPrivateArgTy>;

//===----------------------------------------------------------------------===//
// Leaves (nodes)
//===----------------------------------------------------------------------===//

/// Common base for the leaf nodes.
struct TaskGraphNode : TaskGraphElement {
  int64_t DeviceId = OFFLOAD_DEVICE_DEFAULT;

  /// The leaf's share of the data plan, one entry per recorded map argument
  /// and indexed like the map arrays; empty for a leaf with no map arrays.
  inline llvm::ArrayRef<TaskGraphArgSlotTy> argPlan() const;
  /// The operations the leaf performs, in order.
  inline llvm::ArrayRef<TaskGraphOpTy> ops() const;

  static bool classof(const TaskGraphElement *E) {
    return E->getKind() >= TGKind::TargetNode &&
           E->getKind() <= TGKind::TargetUpdateDataNode;
  }

protected:
  explicit TaskGraphNode(TGKind K) : TaskGraphElement(K) {}

private:
  friend class TaskGraphPlanner;

  // Set when the plan is built.
  TaskGraphArgSlotTy *ArgPlan = nullptr;
  uint32_t NumArgPlans = 0;
  TaskGraphOpTy *OpPlan = nullptr;
  uint32_t NumOps = 0;
};

/// A target kernel launch.  KernelArgs is an opaque KernelArgsTy*, deep-copied
/// by libomp at record time.
struct TaskGraphTargetNode : TaskGraphNode {
  int32_t NumTeams = 0;
  int32_t ThreadLimit = 0;
  void *HostPtr = nullptr;
  void *KernelArgs = nullptr;
  /// The kernel's explicit parameters, in the kernel's order.
  inline llvm::ArrayRef<TaskGraphKernelParamTy> kernelParams() const;

  TaskGraphTargetNode() : TaskGraphNode(TGKind::TargetNode) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::TargetNode;
  }

private:
  friend class TaskGraphPlanner;

  // Set when the plan is built.
  TaskGraphKernelParamTy *Params = nullptr;
  uint32_t NumParams = 0;
};

/// Base for target enter/exit/update data ops.  The specific op is encoded by
/// the Kind.  The arg arrays are libomp-owned (no copy).
struct TaskGraphTargetDataNode : TaskGraphNode {
  int32_t ArgNum = 0;
  void **ArgsBase = nullptr;
  void **Args = nullptr;
  int64_t *ArgSizes = nullptr;
  int64_t *ArgTypes = nullptr;
  void **ArgNames = nullptr;
  void **ArgMappers = nullptr;

  static bool classof(const TaskGraphElement *E) {
    return E->getKind() >= TGKind::TargetEnterDataNode &&
           E->getKind() <= TGKind::TargetUpdateDataNode;
  }

protected:
  explicit TaskGraphTargetDataNode(TGKind K) : TaskGraphNode(K) {}
};

struct TaskGraphTargetEnterDataNode : TaskGraphTargetDataNode {
  TaskGraphTargetEnterDataNode()
      : TaskGraphTargetDataNode(TGKind::TargetEnterDataNode) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::TargetEnterDataNode;
  }
};

struct TaskGraphTargetExitDataNode : TaskGraphTargetDataNode {
  TaskGraphTargetExitDataNode()
      : TaskGraphTargetDataNode(TGKind::TargetExitDataNode) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::TargetExitDataNode;
  }
};

struct TaskGraphTargetUpdateDataNode : TaskGraphTargetDataNode {
  TaskGraphTargetUpdateDataNode()
      : TaskGraphTargetDataNode(TGKind::TargetUpdateDataNode) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::TargetUpdateDataNode;
  }
};

//===----------------------------------------------------------------------===//
// libomptarget services available to a plugin while lowering a graph
//===----------------------------------------------------------------------===//

/// What a plugin needs from libomptarget to lower a recorded graph, passed as a
/// function pointer rather than called directly.
///
/// A plugin cannot name a libomptarget symbol: plugins are static archives
/// linked into both libomptarget.so and LLVMOffload, so a direct call would
/// leave LLVMOffload with an undefined reference.
///
/// Every DeviceId passed here is a concrete device: libomptarget normalizes
/// OFFLOAD_DEVICE_DEFAULT away across the whole tree before a plugin sees it.
struct TaskGraphServicesTy {
  /// Resolve a recorded host function pointer to the plugin-side kernel object
  /// (a GenericKernelTy *), or null if the device has no matching entry.
  void *(*resolveKernel)(int64_t DeviceId, void *HostPtr) = nullptr;
};

//===----------------------------------------------------------------------===//
// The data plan: where a graph's mapped data lives
//
// Built by libomptarget before the graph is offered to a plugin and kept
// current across replays, so that every backend lowers the same decisions
// about which host bytes have which device copy.  Backends differ only in how
// they perform the transfers the plan describes.
//===----------------------------------------------------------------------===//

/// One host range the graph maps, after the maps sharing host memory have been
/// gathered into one.  Gathering keeps each host byte to a single device copy.
///
/// The grouping and the extents are fixed when the graph is finalized; only the
/// addresses are re-derived.  A backend that lowered the graph onto a native
/// graph object can update a node's parameters but cannot add or remove nodes,
/// so the set of ranges has to be an invariant of the graph.
///
/// A range lives in TaskGraphDataPlanTy::Ranges for the graph's lifetime and
/// never moves in memory, so argument entries and backend side tables may hold
/// its address or index by its Id.
class TaskGraphRangeTy {
public:
  /// The range's index in the plan's ranges(), which is in host-address order
  /// when the plan is built.  What a backend indexes a side table by.
  uint32_t getId() const { return Id; }

  /// The host range.  The begin follows relocation; the extent is fixed.
  void *getHostBegin() const { return HostBegin; }
  uint64_t getSize() const { return Size; }
  char *begin() const { return static_cast<char *>(HostBegin); }
  char *end() const { return begin() + Size; }

  /// The device address for this replay: the graph's own storage for a
  /// self-contained range, and otherwise that of the mapping-table entry
  /// covering the range.  Null while the range is absent.
  void *getDevicePtr() const { return DevicePtr; }

private:
  friend class TaskGraphPlanner;
  friend class TaskGraphDataPlanTy;

  uint32_t Id = 0;
  void *HostBegin = nullptr;
  uint64_t Size = 0;
  void *DevicePtr = nullptr;

  // Plan-builder state.

  /// The graph owns the range outright: it creates the range, its maps balance,
  /// and nothing outside the graph had it mapped when the graph was finalized.
  /// Each replay gives such a range device storage of its own and no
  /// mapping-table entry; a claim (see MappingInfoTy) records the storage for
  /// diagnostics.  A replay that finds the range mapped from outside borrows
  /// that mapping instead, and one that finds it partly mapped declines to the
  /// host.  Fixed at finalize.
  bool SelfContained = false;

  /// This replay allocated DevicePtr as a self-contained range's storage, and
  /// frees it when the replay ends.
  bool StorageHeld = false;

  /// The graph created this replay's mapping of the range, so the graph
  /// performs the range's implicit transfers and removes the mapping again.
  /// False means the range was already mapped when the replay began, and the
  /// transfers and the lifetime belong to whoever mapped it, unless the graph
  /// drops its last reference (DropsLast).
  bool PlanInserted = false;

  /// The range was already mapped when the replay began, and the reference the
  /// graph's exit-data maps drop is that mapping's last, so the graph removes
  /// the mapping and performs the copy out it implies.
  bool DropsLast = false;

  /// The graph's enter-data and exit-data maps' net effect on the range's
  /// reference count: +1 for each enter and -1 for each exit, not counting a
  /// struct member's map, which shares its struct's reference.  See persists()
  /// and releases().
  int32_t RefDelta = 0;

  /// Some leaf may bring the range into the device data environment: an
  /// enter-data map or a `target` region's map.  A range named only by
  /// `target update` is never created by the graph; its transfers are switched
  /// off while it is absent.
  bool MayCreate = false;

  /// Some map naming the range carried `present`.
  bool RequiresPresent = false;

  /// An exit-data map named the range with `delete`.
  bool Deleted = false;

  /// Some map naming the range carried `ompx_hold`, so the reference an
  /// unbalanced graph leaves behind goes on the hold count.  Unrelated to Held.
  bool Holds = false;

  /// Some map naming the range carried `close`, so under unified shared memory
  /// the range gets device memory rather than using the host allocation.  The
  /// range is one device copy, so one map asking is enough.
  bool Close = false;

  /// The device uses the host memory directly this replay, under unified
  /// shared memory or automatic zero-copy, so there is nothing to copy.
  bool HostResident = false;

  /// The graph holds a hold reference on the range's mapping until the replay
  /// ends, so the mapping cannot be removed while the graph uses it.
  bool Held = false;

  /// The relocation that last moved the range, numbered by
  /// TaskGraphDataPlanTy::Relocations, and the distance it moved.  Used to
  /// check that every argument in the range moved by the same distance.
  uint64_t MovedInRelocation = 0;
  int64_t MovedBy = 0;

  /// A mapping the graph creates for the range outlives the replay, as one
  /// made by an unbalanced `target enter data` does.
  bool persists() const { return RefDelta > 0 && !Deleted; }
  /// The replay ends one reference down on the mapping it found, as an
  /// unbalanced `target exit data` does.
  bool releases() const { return RefDelta < 0; }
};

llvm::ArrayRef<TaskGraphArgSlotTy> TaskGraphNode::argPlan() const {
  return {ArgPlan, NumArgPlans};
}

/// One operation of a leaf.  A leaf's operations run in the order ops() lists
/// them, which for a target leaf is its copy-ins, its attachments and private
/// initialisations, the kernel, and its copy-outs, and for a data leaf its
/// copies in argument order, a to/from argument's copy-in first, followed by
/// its attachments.
///
/// Every copy-like operation is listed whether or not the current replay
/// performs it; see TaskGraphDataPlanTy::transfer.
struct TaskGraphOpTy {
  enum class KindTy : uint8_t {
    /// Copy a Mapped argument from the host to the device.
    CopyIn,
    /// Copy a Mapped argument from the device to the host.
    CopyOut,
    /// Write an Attach argument's staged pointer into the device copy.
    Attach,
    /// Write a Private argument's staged initial value into its storage.
    PrivateInit,
    /// Launch the target leaf's kernel.
    Launch,
  };
  KindTy Kind;
  /// The argument entry performed, as an index into the leaf's argPlan().
  /// Unused for Launch.
  uint32_t Arg = 0;

  bool isCopyLike() const { return Kind != KindTy::Launch; }
};

/// One explicit parameter of a target leaf's kernel.
struct TaskGraphKernelParamTy {
  enum class KindTy : uint8_t {
    /// The device address of a Mapped argument, at Offset from its range's.
    Mapped,
    /// The address of a Private argument's storage, at Offset from it.
    Private,
    /// A Literal argument's value.
    Literal,
    /// The kernel launch environment, which the backend provides.
    LaunchEnvironment,
  };
  KindTy Kind;
  /// The recorded argument, as an index into the leaf's argPlan().  Unused
  /// for LaunchEnvironment.
  uint32_t Arg = 0;
  /// The kernel is handed a base address, which for a struct member or a
  /// pointer-and-pointee map lies before the mapped begin, so this can be
  /// negative.  The distance is a property of the map's shape and is fixed
  /// for the graph.
  int64_t Offset = 0;
};

/// Where a copy-like operation moves its bytes this replay.
struct TaskGraphTransferTy {
  /// Either end is null while it cannot be resolved, such as the device end
  /// of a range absent this replay; the operation is then disabled.
  void *Dst = nullptr;
  const void *Src = nullptr;
  /// The range the host end lies in, or null when the host end is the plan's
  /// own staging.  A backend that registers or pins host memory per range
  /// finds the host end's buffer here.
  const TaskGraphRangeTy *HostRange = nullptr;
  uint64_t Size = 0;
  bool ToDevice = true;
  /// This replay performs the operation.
  bool Enabled = false;
};

/// What changed about one argument entry since the last replay the backend
/// took in, for a backend that has already lowered the graph.
///
/// Every operation and kernel parameter a backend lowers comes from one
/// argument entry, so the entry is the unit changes are reported against: an
/// attachment depends on the ranges of both its pointer and its pointee, but
/// the plan resolves that and reports the attachment once.  Values themselves
/// are always read from the plan; an update only says which ones to re-read.
///
/// An update only repoints addresses, refreshes bytes the plan supplies and
/// switches operations on or off; it never adds or removes an operation.  A
/// native-graph backend can therefore serve it with node parameter updates and
/// node enablement.  Because nothing is added later, a backend has to lower
/// every copy-like operation whether or not the current replay performs it.  A
/// range named only by `target update` can be absent when the graph is built,
/// with no device address and its transfers disabled, and mapped by the
/// program before a later replay.  A native-graph backend therefore points such
/// an operand at scratch of its own, sized for the largest such transfer, and
/// enables the node once the range resolves.
struct TaskGraphArgUpdateTy {
  enum ChangeTy : uint8_t {
    /// The entry's device address moved: a Mapped entry's or an attachment's
    /// pointer's range moved on the device.
    DeviceAddressChanged = 1 << 0,
    /// The entry's host address moved: relocation moved a Mapped entry's
    /// range.
    HostAddressChanged = 1 << 1,
    /// The bytes the plan supplies for the entry changed: an attachment's or a
    /// private's staged bytes, or a literal's value.  Program memory a copy
    /// reads when it runs is not such bytes.
    DataChanged = 1 << 2,
    /// Whether the replay performs the entry's copy-like operations changed.
    EnabledChanged = 1 << 3,
    /// Every kind of change, including any added later: what a backend has to
    /// assume of every entry when it is told to resync.
    AllChanged = 0xff,
  };
  /// The entry, as an index into TaskGraphDataPlanTy::Args.
  uint32_t Arg = 0;
  /// A mask of ChangeTy.
  uint8_t Changes = 0;

  bool has(ChangeTy Change) const { return Changes & Change; }
};

/// Where a graph's mapped data lives.  libomptarget builds it and keeps it
/// current; a backend reads it through the members below.
class TaskGraphDataPlanTy {
public:
  /// The ranges, indexed by Id.  They never overlap, and ranges that merely
  /// abut are kept apart.  Fixed in number when the plan is built, so a
  /// backend may hold a range's address.
  llvm::ArrayRef<TaskGraphRangeTy> ranges() const { return Ranges; }

  /// Every leaf's argument entries, end to end, as an update names them.
  size_t getNumArgs() const { return Args.size(); }
  const TaskGraphArgSlotTy &getArg(uint32_t Index) const { return Args[Index]; }
  /// \p A's index among the entries, which is what an update names it by.
  uint32_t argIndex(const TaskGraphArgSlotTy &A) const {
    return static_cast<uint32_t>(&A - Args.data());
  }

  /// The bytes every attachment writes and the host image of the private
  /// block.  Refilled in place each replay, at addresses fixed for the
  /// graph's lifetime, so a backend may lower a copy from them once.
  llvm::ArrayRef<char> attachStaging() const { return AttachStaging; }
  llvm::ArrayRef<char> privateStaging() const { return PrivateStaging; }
  /// Device storage for every private copy in the graph, allocated when the
  /// plan is built and freed with it.  Not in the mapping table.
  void *getPrivateBlock() const { return PrivateBlock; }

  /// Where \p Op, a copy-like operation of \p Leaf, moves its bytes this
  /// replay.  The size and direction are fixed for the graph.
  TaskGraphTransferTy transfer(const TaskGraphNode &Leaf,
                               const TaskGraphOpTy &Op) const;

  /// The value \p Leaf's kernel is handed for \p Param this replay, given the
  /// backend's launch environment \p LaunchEnvironment.
  uint64_t paramValue(const TaskGraphTargetNode &Leaf,
                      const TaskGraphKernelParamTy &Param,
                      void *LaunchEnvironment) const;

  /// The last acquisition declined its replay for a reason it has reported
  /// itself, so the caller need not.
  bool declineReported() const { return DeclineReported; }

private:
  friend class TaskGraphPlanner;

  const char *attachStagingFor(const TaskGraphAttachArgTy &A) const {
    return AttachStaging.data() + A.StagingOffset;
  }
  const char *privateStagingFor(const TaskGraphFirstPrivateArgTy &F) const {
    return PrivateStaging.data() + F.StagingOffset;
  }
  char *privateDevicePtr(const TaskGraphFirstPrivateArgTy &F) const {
    return static_cast<char *>(PrivateBlock) + F.StagingOffset;
  }

  std::vector<TaskGraphRangeTy> Ranges;
  /// Never resized after the plan is built, so pointers into it are stable.
  std::vector<TaskGraphArgSlotTy> Args;
  std::vector<char> AttachStaging;
  /// What each attached pointer held on the host when the replay started, at
  /// the same offsets as AttachStaging.  A `from` copy of the range holding
  /// the pointer brings the device value back, so the host value is restored
  /// when the replay ends, as the runtime does at the end of a data region.
  std::vector<char> AttachHostBytes;
  std::vector<char> PrivateStaging;
  /// What each private pointer initialisation's host object held when the
  /// replay began, at the same offsets as PrivateStaging.
  std::vector<char> PrivateHostBytes;
  void *PrivateBlock = nullptr;
  /// Every leaf's operations, end to end; each leaf's OpPlan spans its own.
  std::vector<TaskGraphOpTy> Ops;
  /// Every target leaf's kernel parameters, end to end.
  std::vector<TaskGraphKernelParamTy> KernelParams;

  /// The mapping-table epoch as of the plan's build or its last release of the
  /// mappings.  Only used to report that the program changed the mapping state
  /// between replays; every replay resolves every range regardless.
  uint64_t Epoch = 0;

  /// The graph holds its mappings for a replay in flight.
  bool MappingsHeld = false;

  /// How many replays have re-keyed the ranges.  Compared against a range's
  /// MovedInRelocation; zero stamps no range.
  uint64_t Relocations = 0;

  /// The backend may not have adopted every change.  A replay declined to the
  /// host can change the plan without telling the backend, so the next replay
  /// to reach it has the backend re-read everything.  Set at the start of each
  /// acquisition and cleared by commitTaskGraphUpdates.
  bool Resync = false;

  /// A range that one leaf may copy while another leaf, which the region tree
  /// does not order with it, uses the range.  Found when the plan is built; a
  /// replay in which the copier does copy the range is declined to the host.
  struct CopyConflictTy {
    uint32_t Range;
    uint32_t Copier;
    uint32_t User;
  };
  std::vector<CopyConflictTy> CopyConflicts;

  /// The previous replay was declined for a copy conflict, so another is not
  /// reported again.
  bool ConflictDeclined = false;

  /// See declineReported().  Cleared at the start of each acquisition.
  bool DeclineReported = false;
};

llvm::ArrayRef<TaskGraphOpTy> TaskGraphNode::ops() const {
  return {OpPlan, NumOps};
}

llvm::ArrayRef<TaskGraphKernelParamTy>
TaskGraphTargetNode::kernelParams() const {
  return {Params, NumParams};
}

/// Create a TaskGraphTransferTy describing geometry of a copy operation from
/// a taskgraph \p Leaf and a particular operation \p Op.

inline TaskGraphTransferTy
TaskGraphDataPlanTy::transfer(const TaskGraphNode &Leaf,
                              const TaskGraphOpTy &Op) const {
  using KindTy = TaskGraphOpTy::KindTy;
  assert(Op.isCopyLike() && "not a copy-like operation");
  const TaskGraphArgSlotTy &ArgSlot = Leaf.argPlan()[Op.Arg];
  // Offset non-nullptr Base, or return nullptr.
  auto maybeOffset = [](void *Base, uint64_t Offset) -> char * {
    return Base ? static_cast<char *>(Base) + Offset : nullptr;
  };
  TaskGraphTransferTy T;
  T.Enabled = false;
  auto *Arg = cast<TaskGraphArgBaseTy>(&ArgSlot.get());
  switch (Op.Kind) {
  case KindTy::CopyIn:
  case KindTy::CopyOut: {
    const auto *M = cast<TaskGraphMappedArgTy>(Arg);
    char *Device = maybeOffset(M->Range->DevicePtr, M->Offset);
    char *Host = maybeOffset(M->Range->HostBegin, M->Offset);
    T.ToDevice = Op.Kind == KindTy::CopyIn;
    T.HostRange = M->Range;
    T.Dst = T.ToDevice ? Device : Host;
    T.Src = T.ToDevice ? Host : Device;
    T.Size = M->Size;
    T.Enabled = T.ToDevice ? M->In.Enabled : M->Out.Enabled;
    break;
  }
  case KindTy::Attach: {
    const auto *A = cast<TaskGraphAttachArgTy>(Arg);
    T.Dst = maybeOffset(A->PtrRange->DevicePtr, A->Offset);
    T.Src = attachStagingFor(*A);
    T.Size = A->Size;
    T.Enabled = A->Enabled;
    break;
  }
  case KindTy::PrivateInit: {
    const auto *F = cast<TaskGraphFirstPrivateArgTy>(Arg);
    T.Dst = PrivateBlock ? privateDevicePtr(*F) : nullptr;
    T.Src = privateStagingFor(*F);
    T.Size = F->Size;
    T.Enabled = true;
    break;
  }
  case KindTy::Launch:
    break;
  }
  return T;
}

inline uint64_t
TaskGraphDataPlanTy::paramValue(const TaskGraphTargetNode &Leaf,
                                const TaskGraphKernelParamTy &Param,
                                void *LaunchEnvironment) const {
  using KindTy = TaskGraphKernelParamTy::KindTy;
  switch (Param.Kind) {
  case KindTy::Mapped: {
    const TaskGraphArgSlotTy &S = Leaf.argPlan()[Param.Arg];
    const auto *M = cast<TaskGraphMappedArgTy>(&S.get());
    return reinterpret_cast<uint64_t>(M->Range->DevicePtr) + Param.Offset;
  }
  case KindTy::Private: {
    const TaskGraphArgSlotTy &S = Leaf.argPlan()[Param.Arg];
    const auto *F = cast<TaskGraphFirstPrivateArgTy>(&S.get());
    return reinterpret_cast<uint64_t>(privateDevicePtr(*F)) + Param.Offset;
  }
  case KindTy::Literal:
    return reinterpret_cast<uint64_t>(
        static_cast<const KernelArgsTy *>(Leaf.KernelArgs)
            ->ArgBasePtrs[Param.Arg]);
  case KindTy::LaunchEnvironment:
    return reinterpret_cast<uint64_t>(LaunchEnvironment);
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// The graph
//===----------------------------------------------------------------------===//

class TaskGraphBuilder;

/// Where the build pass links the next streamed element.
struct BuildCursor {
  /// The region being filled, or null at the root.
  TaskGraphRegion *Cur = nullptr;
  /// The next slot in Cur->children().
  uint32_t Index = 0;
};

/// A processed taskgraph, streamed in by libomp and built by TaskGraphBuilder
/// (Shared/TaskGraphBuilder.h).  A plugin is handed a const graph, which it
/// only reads.
class TaskGraphTy {
public:
  TaskGraphTy(int64_t DeviceId, uintptr_t GraphId)
      : DeviceId(DeviceId), GraphId(GraphId) {}
  ~TaskGraphTy() { std::free(Block); }
  TaskGraphTy(const TaskGraphTy &) = delete;
  TaskGraphTy &operator=(const TaskGraphTy &) = delete;

  /// The device the graph is replayed on, concrete once the graph is offered
  /// to a plugin.
  int64_t DeviceId;
  uintptr_t GraphId;
  /// The claiming plugin's own form of the graph: whatever its finalize
  /// returned, opaque to everything but that plugin.  Null for a graph no
  /// plugin claimed.
  void *PluginGraph = nullptr;

  /// Filled in by libomptarget before the graph is offered to a plugin.
  TaskGraphServicesTy Services;

  /// Where the graph's mapped data lives.  Built before the graph is offered to
  /// a plugin and kept current by libomptarget across replays, so a plugin only
  /// ever reads it.
  TaskGraphDataPlanTy Plan;

  /// The top of the tree: the first element streamed outside any region.  A
  /// region in general, but a graph holding a single leaf arrives as that leaf.
  TaskGraphElement *getRoot() { return Root; }
  const TaskGraphElement *getRoot() const { return Root; }

  /// The leaves in preorder.  Empty for a graph that was not well formed.
  ArrayRef<TaskGraphNode *> leaves() { return Leaves; }
  ArrayRef<const TaskGraphNode *> leaves() const {
    return ArrayRef<TaskGraphNode *>(Leaves);
  }

private:
  friend class TaskGraphBuilder;

  TaskGraphElement *Root = nullptr;
  std::vector<TaskGraphNode *> Leaves;

  /// The single block holding the whole tree, null on the measuring pass.
  char *Block = nullptr;
  /// The block's size: accumulated on the measuring pass, given on the build
  /// pass.
  size_t MeasuredBytes = 0;
  /// How much of the block the build pass has used.
  size_t BlockUsed = 0;
  BuildCursor Cursor;
  /// How the stream broke the transmission protocol, if it did.
  const char *Malformed = nullptr;
};

} // namespace target
} // namespace omp
} // namespace llvm

#endif // OMPTARGET_SHARED_TASKGRAPH_H
