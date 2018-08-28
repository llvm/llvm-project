//===--------------------- ResourceManager.h --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// The classes here represent processor resource units and their management
/// strategy.  These classes are managed by the Scheduler.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_RESOURCE_MANAGER_H
#define LLVM_TOOLS_LLVM_MCA_RESOURCE_MANAGER_H

#include "Instruction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSchedule.h"

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
};
} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_RESOURCE_MANAGER_H
