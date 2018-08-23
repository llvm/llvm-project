//===--------------------- Scheduler.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// A scheduler for Processor Resource Units and Processor Resource Groups.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_SCHEDULER_H
#define LLVM_TOOLS_LLVM_MCA_SCHEDULER_H

#include "HardwareUnit.h"
#include "Instruction.h"
#include "LSUnit.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include <map>

namespace mca {

/// Used to notify the internal state of a processor resource.
///
/// A processor resource is available if it is not reserved, and there are
/// available slots in the buffer.  A processor resource is unavailable if it
/// is either reserved, or the associated buffer is full. A processor resource
/// with a buffer size of -1 is always available if it is not reserved.
///
/// Values of type ResourceStateEvent are returned by method
/// ResourceState::isBufferAvailable(), which is used to query the internal
/// state of a resource.
///
/// The naming convention for resource state events is:
///  * Event names start with prefix RS_
///  * Prefix RS_ is followed by a string describing the actual resource state.
enum ResourceStateEvent {
  RS_BUFFER_AVAILABLE,
  RS_BUFFER_UNAVAILABLE,
  RS_RESERVED
};

/// Resource allocation strategy used by hardware scheduler resources.
class ResourceStrategy {
  ResourceStrategy(const ResourceStrategy &) = delete;
  ResourceStrategy &operator=(const ResourceStrategy &) = delete;

public:
  ResourceStrategy() {}
  virtual ~ResourceStrategy();

  /// Selects a processor resource unit from a ReadyMask.
  virtual uint64_t select(uint64_t ReadyMask) = 0;

  /// Called by the ResourceManager when a processor resource group, or a
  /// processor resource with multiple units has become unavailable.
  ///
  /// The default strategy uses this information to bias its selection logic.
  virtual void used(uint64_t ResourceMask) {}
};

/// Default resource allocation strategy used by processor resource groups and
/// processor resources with multiple units.
class DefaultResourceStrategy final : public ResourceStrategy {
  /// A Mask of resource unit identifiers.
  ///
  /// There is one bit set for every available resource unit.
  /// It defaults to the value of field ResourceSizeMask in ResourceState.
  const unsigned ResourceUnitMask;

  /// A simple round-robin selector for processor resource units.
  /// Each bit of this mask identifies a sub resource within a group.
  ///
  /// As an example, lets assume that this is a default policy for a
  /// processor resource group composed by the following three units:
  ///   ResourceA -- 0b001
  ///   ResourceB -- 0b010
  ///   ResourceC -- 0b100
  ///
  /// Field NextInSequenceMask is used to select the next unit from the set of
  /// resource units. It defaults to the value of field `ResourceUnitMasks` (in
  /// this example, it defaults to mask '0b111').
  ///
  /// The round-robin selector would firstly select 'ResourceC', then
  /// 'ResourceB', and eventually 'ResourceA'.  When a resource R is used, the
  /// corresponding bit in NextInSequenceMask is cleared.  For example, if
  /// 'ResourceC' is selected, then the new value of NextInSequenceMask becomes
  /// 0xb011.
  ///
  /// When NextInSequenceMask becomes zero, it is automatically reset to the
  /// default value (i.e. ResourceUnitMask).
  uint64_t NextInSequenceMask;

  /// This field is used to track resource units that are used (i.e. selected)
  /// by other groups other than the one associated with this strategy object.
  ///
  /// In LLVM processor resource groups are allowed to partially (or fully)
  /// overlap. That means, a same unit may be visible to multiple groups.
  /// This field keeps track of uses that have originated from outside of
  /// this group. The idea is to bias the selection strategy, so that resources
  /// that haven't been used by other groups get prioritized.
  ///
  /// The end goal is to (try to) keep the resource distribution as much uniform
  /// as possible. By construction, this mask only tracks one-level of resource
  /// usage. Therefore, this strategy is expected to be less accurate when same
  /// units are used multiple times by other groups within a single round of
  /// select.
  ///
  /// Note: an LRU selector would have a better accuracy at the cost of being
  /// slightly more expensive (mostly in terms of runtime cost). Methods
  /// 'select' and 'used', are always in the hot execution path of llvm-mca.
  /// Therefore, a slow implementation of 'select' would have a negative impact
  /// on the overall performance of the tool.
  uint64_t RemovedFromNextInSequence;

  void skipMask(uint64_t Mask);

public:
  DefaultResourceStrategy(uint64_t UnitMask)
      : ResourceStrategy(), ResourceUnitMask(UnitMask),
        NextInSequenceMask(UnitMask), RemovedFromNextInSequence(0) {}
  virtual ~DefaultResourceStrategy() = default;

  uint64_t select(uint64_t ReadyMask) override;
  void used(uint64_t Mask) override;
};

/// A processor resource descriptor.
///
/// There is an instance of this class for every processor resource defined by
/// the machine scheduling model.
/// Objects of class ResourceState dynamically track the usage of processor
/// resource units.
class ResourceState {
  /// An index to the MCProcResourceDesc entry in the processor model.
  const unsigned ProcResourceDescIndex;
  /// A resource mask. This is generated by the tool with the help of
  /// function `mca::createProcResourceMasks' (see Support.h).
  const uint64_t ResourceMask;

  /// A ProcResource can have multiple units.
  ///
  /// For processor resource groups,
  /// this field default to the value of field `ResourceMask`; the number of
  /// bits set is equal to the cardinality of the group.  For normal (i.e.
  /// non-group) resources, the number of bits set in this mask is equivalent
  /// to the number of units declared by the processor model (see field
  /// 'NumUnits' in 'ProcResourceUnits').
  uint64_t ResourceSizeMask;

  /// A mask of ready units.
  uint64_t ReadyMask;

  /// Buffered resources will have this field set to a positive number different
  /// than zero. A buffered resource behaves like a reservation station
  /// implementing its own buffer for out-of-order execution.
  ///
  /// A BufferSize of 1 is used by scheduler resources that force in-order
  /// execution.
  ///
  /// A BufferSize of 0 is used to model in-order issue/dispatch resources.
  /// Since in-order issue/dispatch resources don't implement buffers, dispatch
  /// events coincide with issue events.
  /// Also, no other instruction ca be dispatched/issue while this resource is
  /// in use. Only when all the "resource cycles" are consumed (after the issue
  /// event), a new instruction ca be dispatched.
  const int BufferSize;

  /// Available slots in the buffer (zero, if this is not a buffered resource).
  unsigned AvailableSlots;

  /// This field is set if this resource is currently reserved.
  ///
  /// Resources can be reserved for a number of cycles.
  /// Instructions can still be dispatched to reserved resources. However,
  /// istructions dispatched to a reserved resource cannot be issued to the
  /// underlying units (i.e. pipelines) until the resource is released.
  bool Unavailable;

  /// Checks for the availability of unit 'SubResMask' in the group.
  bool isSubResourceReady(uint64_t SubResMask) const {
    return ReadyMask & SubResMask;
  }

public:
  ResourceState(const llvm::MCProcResourceDesc &Desc, unsigned Index,
                uint64_t Mask);

  unsigned getProcResourceID() const { return ProcResourceDescIndex; }
  uint64_t getResourceMask() const { return ResourceMask; }
  uint64_t getReadyMask() const { return ReadyMask; }
  int getBufferSize() const { return BufferSize; }

  bool isBuffered() const { return BufferSize > 0; }
  bool isInOrder() const { return BufferSize == 1; }

  /// Returns true if this is an in-order dispatch/issue resource.
  bool isADispatchHazard() const { return BufferSize == 0; }
  bool isReserved() const { return Unavailable; }

  void setReserved() { Unavailable = true; }
  void clearReserved() { Unavailable = false; }

  /// Returs true if this resource is not reserved, and if there are at least
  /// `NumUnits` available units.
  bool isReady(unsigned NumUnits = 1) const;

  bool isAResourceGroup() const {
    return llvm::countPopulation(ResourceMask) > 1;
  }

  bool containsResource(uint64_t ID) const { return ResourceMask & ID; }

  void markSubResourceAsUsed(uint64_t ID) {
    assert(isSubResourceReady(ID));
    ReadyMask ^= ID;
  }

  void releaseSubResource(uint64_t ID) {
    assert(!isSubResourceReady(ID));
    ReadyMask ^= ID;
  }

  unsigned getNumUnits() const {
    return isAResourceGroup() ? 1U : llvm::countPopulation(ResourceSizeMask);
  }

  /// Checks if there is an available slot in the resource buffer.
  ///
  /// Returns RS_BUFFER_AVAILABLE if this is not a buffered resource, or if
  /// there is a slot available.
  ///
  /// Returns RS_RESERVED if this buffered resource is a dispatch hazard, and it
  /// is reserved.
  ///
  /// Returns RS_BUFFER_UNAVAILABLE if there are no available slots.
  ResourceStateEvent isBufferAvailable() const;

  /// Reserve a slot in the buffer.
  void reserveBuffer() {
    if (AvailableSlots)
      AvailableSlots--;
  }

  /// Release a slot in the buffer.
  void releaseBuffer() {
    if (BufferSize > 0)
      AvailableSlots++;
    assert(AvailableSlots <= static_cast<unsigned>(BufferSize));
  }

#ifndef NDEBUG
  void dump() const;
#endif
};

/// A resource unit identifier.
///
/// This is used to identify a specific processor resource unit using a pair
/// of indices where the 'first' index is a processor resource mask, and the
/// 'second' index is an index for a "sub-resource" (i.e. unit).
typedef std::pair<uint64_t, uint64_t> ResourceRef;

// First: a MCProcResourceDesc index identifying a buffered resource.
// Second: max number of buffer entries used in this resource.
typedef std::pair<unsigned, unsigned> BufferUsageEntry;

/// A resource manager for processor resource units and groups.
///
/// This class owns all the ResourceState objects, and it is responsible for
/// acting on requests from a Scheduler by updating the internal state of
/// ResourceState objects.
/// This class doesn't know about instruction itineraries and functional units.
/// In future, it can be extended to support itineraries too through the same
/// public interface.
class ResourceManager {
  // The resource manager owns all the ResourceState.
  std::vector<std::unique_ptr<ResourceState>> Resources;
  std::vector<std::unique_ptr<ResourceStrategy>> Strategies;

  // Keeps track of which resources are busy, and how many cycles are left
  // before those become usable again.
  llvm::SmallDenseMap<ResourceRef, unsigned> BusyResources;

  // A table to map processor resource IDs to processor resource masks.
  llvm::SmallVector<uint64_t, 8> ProcResID2Mask;

  // Returns the actual resource unit that will be used.
  ResourceRef selectPipe(uint64_t ResourceID);

  void use(const ResourceRef &RR);
  void release(const ResourceRef &RR);

  unsigned getNumUnits(uint64_t ResourceID) const;

  // Overrides the selection strategy for the processor resource with the given
  // mask.
  void setCustomStrategyImpl(std::unique_ptr<ResourceStrategy> S,
                             uint64_t ResourceMask);

public:
  ResourceManager(const llvm::MCSchedModel &SM);
  virtual ~ResourceManager() = default;

  // Overrides the selection strategy for the resource at index ResourceID in
  // the MCProcResourceDesc table.
  void setCustomStrategy(std::unique_ptr<ResourceStrategy> S,
                         unsigned ResourceID) {
    assert(ResourceID < ProcResID2Mask.size() &&
           "Invalid resource index in input!");
    return setCustomStrategyImpl(std::move(S), ProcResID2Mask[ResourceID]);
  }

  // Returns RS_BUFFER_AVAILABLE if buffered resources are not reserved, and if
  // there are enough available slots in the buffers.
  ResourceStateEvent canBeDispatched(llvm::ArrayRef<uint64_t> Buffers) const;

  // Return the processor resource identifier associated to this Mask.
  unsigned resolveResourceMask(uint64_t Mask) const;

  // Consume a slot in every buffered resource from array 'Buffers'. Resource
  // units that are dispatch hazards (i.e. BufferSize=0) are marked as reserved.
  void reserveBuffers(llvm::ArrayRef<uint64_t> Buffers);

  // Release buffer entries previously allocated by method reserveBuffers.
  void releaseBuffers(llvm::ArrayRef<uint64_t> Buffers);

  // Reserve a processor resource. A reserved resource is not available for
  // instruction issue until it is released.
  void reserveResource(uint64_t ResourceID);

  // Release a previously reserved processor resource.
  void releaseResource(uint64_t ResourceID);

  // Returns true if all resources are in-order, and there is at least one
  // resource which is a dispatch hazard (BufferSize = 0).
  bool mustIssueImmediately(const InstrDesc &Desc) const;

  bool canBeIssued(const InstrDesc &Desc) const;

  void issueInstruction(
      const InstrDesc &Desc,
      llvm::SmallVectorImpl<std::pair<ResourceRef, double>> &Pipes);

  void cycleEvent(llvm::SmallVectorImpl<ResourceRef> &ResourcesFreed);

#ifndef NDEBUG
  void dump() const {
    for (const std::unique_ptr<ResourceState> &Resource : Resources)
      Resource->dump();
  }
#endif
}; // namespace mca

class SchedulerStrategy {
public:
  SchedulerStrategy() = default;
  virtual ~SchedulerStrategy();

  /// Returns true if Lhs should take priority over Rhs.
  ///
  /// This method is used by class Scheduler to select the "best" ready
  /// instruction to issue to the underlying pipelines.
  virtual bool compare(const InstRef &Lhs, const InstRef &Rhs) const = 0;
};

/// Default instruction selection strategy used by class Scheduler.
class DefaultSchedulerStrategy : public SchedulerStrategy {
  /// This method ranks instructions based on their age, and the number of known
  /// users. The lower the rank value, the better.
  int computeRank(const InstRef &Lhs) const {
    return Lhs.getSourceIndex() - Lhs.getInstruction()->getNumUsers();
  }

public:
  DefaultSchedulerStrategy() = default;
  virtual ~DefaultSchedulerStrategy();

  bool compare(const InstRef &Lhs, const InstRef &Rhs) const override {
    int LhsRank = computeRank(Lhs);
    int RhsRank = computeRank(Rhs);

    /// Prioritize older instructions over younger instructions to minimize the
    /// pressure on the reorder buffer.
    if (LhsRank == RhsRank)
      return Lhs.getSourceIndex() < Rhs.getSourceIndex();
    return LhsRank < RhsRank;
  }
};

/// Class Scheduler is responsible for issuing instructions to pipeline
/// resources.
///
/// Internally, it delegates to a ResourceManager the management of processor
/// resources. This class is also responsible for tracking the progress of
/// instructions from the dispatch stage, until the write-back stage.
///
/// An instruction dispatched to the Scheduler is initially placed into either
/// the 'WaitSet' or the 'ReadySet' depending on the availability of the input
/// operands.
///
/// An instruction is moved from the WaitSet to the ReadySet when register
/// operands become available, and all memory dependencies are met.
/// Instructions that are moved from the WaitSet to the ReadySet transition
/// in state from 'IS_AVAILABLE' to 'IS_READY'.
///
/// On every cycle, the Scheduler checks if it can promote instructions from the
/// WaitSet to the ReadySet.
///
/// An Instruction is moved from the ReadySet the `IssuedSet` when it is issued
/// to a (one or more) pipeline(s). This event also causes an instruction state
/// transition (i.e. from state IS_READY, to state IS_EXECUTING). An Instruction
/// leaves the IssuedSet when it reaches the write-back stage.
class Scheduler : public HardwareUnit {
  LSUnit *LSU;

  // Instruction selection strategy for this Scheduler.
  std::unique_ptr<SchedulerStrategy> Strategy;

  // Hardware resources that are managed by this scheduler.
  std::unique_ptr<ResourceManager> Resources;

  std::vector<InstRef> WaitSet;
  std::vector<InstRef> ReadySet;
  std::vector<InstRef> IssuedSet;

  /// Issue an instruction without updating the ready queue.
  void issueInstructionImpl(
      InstRef &IR,
      llvm::SmallVectorImpl<std::pair<ResourceRef, double>> &Pipes);

  // Identify instructions that have finished executing, and remove them from
  // the IssuedSet. References to executed instructions are added to input
  // vector 'Executed'.
  void updateIssuedSet(llvm::SmallVectorImpl<InstRef> &Executed);

  // Try to promote instructions from WaitSet to ReadySet.
  // Add promoted instructions to the 'Ready' vector in input.
  void promoteToReadySet(llvm::SmallVectorImpl<InstRef> &Ready);

public:
  Scheduler(const llvm::MCSchedModel &Model, LSUnit *Lsu)
      : LSU(Lsu), Strategy(llvm::make_unique<DefaultSchedulerStrategy>()),
        Resources(llvm::make_unique<ResourceManager>(Model)) {}
  Scheduler(const llvm::MCSchedModel &Model, LSUnit *Lsu,
            std::unique_ptr<SchedulerStrategy> SelectStrategy)
      : LSU(Lsu), Strategy(std::move(SelectStrategy)),
        Resources(llvm::make_unique<ResourceManager>(Model)) {}
  Scheduler(std::unique_ptr<ResourceManager> RM, LSUnit *Lsu,
            std::unique_ptr<SchedulerStrategy> SelectStrategy)
      : LSU(Lsu), Strategy(std::move(SelectStrategy)),
        Resources(std::move(RM)) {}

  // Stalls generated by the scheduler.
  enum Status {
    SC_AVAILABLE,
    SC_LOAD_QUEUE_FULL,
    SC_STORE_QUEUE_FULL,
    SC_BUFFERS_FULL,
    SC_DISPATCH_GROUP_STALL,
  };

  /// Check if the instruction in 'IR' can be dispatched and returns an answer
  /// in the form of a Status value.
  ///
  /// The DispatchStage is responsible for querying the Scheduler before
  /// dispatching new instructions. This routine is used for performing such
  /// a query.  If the instruction 'IR' can be dispatched, then true is
  /// returned, otherwise false is returned with Event set to the stall type.
  /// Internally, it also checks if the load/store unit is available.
  Status isAvailable(const InstRef &IR) const;

  /// Reserves buffer and LSUnit queue resources that are necessary to issue
  /// this instruction.
  ///
  /// Returns true if instruction IR is ready to be issued to the underlying
  /// pipelines. Note that this operation cannot fail; it assumes that a
  /// previous call to method `isAvailable(IR)` returned `SC_AVAILABLE`.
  void dispatch(const InstRef &IR);

  /// Returns true if IR is ready to be executed by the underlying pipelines.
  /// This method assumes that IR has been previously dispatched.
  bool isReady(const InstRef &IR) const;

  /// Issue an instruction and populates a vector of used pipeline resources,
  /// and a vector of instructions that transitioned to the ready state as a
  /// result of this event.
  void
  issueInstruction(InstRef &IR,
                   llvm::SmallVectorImpl<std::pair<ResourceRef, double>> &Used,
                   llvm::SmallVectorImpl<InstRef> &Ready);

  /// Returns true if IR has to be issued immediately, or if IR is a zero
  /// latency instruction.
  bool mustIssueImmediately(const InstRef &IR) const;

  /// This routine notifies the Scheduler that a new cycle just started.
  ///
  /// It notifies the underlying ResourceManager that a new cycle just started.
  /// Vector `Freed` is populated with resourceRef related to resources that
  /// have changed in state, and that are now available to new instructions.
  /// Instructions executed are added to vector Executed, while vector Ready is
  /// populated with instructions that have become ready in this new cycle.
  void cycleEvent(llvm::SmallVectorImpl<ResourceRef> &Freed,
                  llvm::SmallVectorImpl<InstRef> &Ready,
                  llvm::SmallVectorImpl<InstRef> &Executed);

  /// Convert a resource mask into a valid llvm processor resource identifier.
  unsigned getResourceID(uint64_t Mask) const {
    return Resources->resolveResourceMask(Mask);
  }

  /// Select the next instruction to issue from the ReadySet. Returns an invalid
  /// instruction reference if there are no ready instructions, or if processor
  /// resources are not available.
  InstRef select();

#ifndef NDEBUG
  // Update the ready queues.
  void dump() const;

  // This routine performs a sanity check.  This routine should only be called
  // when we know that 'IR' is not in the scheduler's instruction queues.
  void sanityCheck(const InstRef &IR) const {
    assert(llvm::find(WaitSet, IR) == WaitSet.end());
    assert(llvm::find(ReadySet, IR) == ReadySet.end());
    assert(llvm::find(IssuedSet, IR) == IssuedSet.end());
  }
#endif // !NDEBUG
};
} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_SCHEDULER_H
