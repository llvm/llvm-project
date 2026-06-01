//===-- MemorySlotInterfaces.td - MemorySlot interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_MEMORYSLOTINTERFACES
#define MLIR_INTERFACES_MEMORYSLOTINTERFACES

include "mlir/IR/OpBase.td"

def PromotableAllocationOpInterface
    : OpInterface<"PromotableAllocationOpInterface"> {
  let description = [{
    Describes an operation allocating a memory slot that can be promoted into
    SSA values.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns a list of memory slots for which promotion should be attempted.
        This only considers the local semantics of the allocator, ignoring
        whether the slot pointer is properly used or not. This allocator is the
        "owner" of the returned slots, meaning no two allocators should return
        the same slot. The content of the memory slot must only be reachable
        using loads and stores to the provided slot pointer, no aliasing is
        allowed.

        Promotion of the slot will lead to the slot pointer no longer being
        used, leaving the content of the memory slot unreachable.

        No IR mutation is allowed in this method.
      }], "::llvm::SmallVector<::mlir::MemorySlot>", "getPromotableSlots",
      (ins)
    >,
    InterfaceMethod<[{
        Provides the default Value of this memory slot. The provided Value
        will be used as the reaching definition of loads done before any store.
        This Value must outlive the promotion and dominate all the uses of this
        slot's pointer. The provided builder can be used to create the default
        value on the fly.

        The builder is located at the beginning of the block where the slot
        pointer is defined.
      }], "::mlir::Value", "getDefaultValue",
      (ins
        "const ::mlir::MemorySlot &":$slot,
        "::mlir::OpBuilder &":$builder)
    >,
    InterfaceMethod<[{
        Hook triggered for every new block argument added to a block.
        This will only be called for slots declared by this operation.
        This function is called after removal of blocking uses, meaning
        only operations that will be deleted remain users of the slot.

        The builder is located at the beginning of the block on call. All IR
        mutations must happen through the builder.
      }],
      "void", "handleBlockArgument",
      (ins
        "const ::mlir::MemorySlot &":$slot,
        "::mlir::BlockArgument":$argument,
        "::mlir::OpBuilder &":$builder
      )
    >,
    InterfaceMethod<[{
        Hook triggered once the promotion of a slot is complete. This can
        also clean up the created default value if necessary. The default
        value may be a null value if no default value was created.
        This will only be called for slots declared by this operation.

        Must return a new promotable allocation op if this operation produced
        multiple promotable slots, nullopt otherwise.
      }],
      "::std::optional<::mlir::PromotableAllocationOpInterface>",
        "handlePromotionComplete",
      (ins
        "const ::mlir::MemorySlot &":$slot,
        "::mlir::Value":$defaultValue,
        "::mlir::OpBuilder &":$builder)
    >,
  ];
}

def PromotableMemOpInterface : OpInterface<"PromotableMemOpInterface"> {
  let description = [{
    Describes an operation that can load from memory slots and/or store
    to memory slots.

    For a memory operation on a slot to be valid, it must strictly operate
    within the bounds of the slot.

    If the same operation does both loads and stores on the same slot, the
    load must semantically happen first.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Gets whether this operation loads from the specified slot.

        No IR mutation is allowed in this method.
      }],
      "bool", "loadsFrom",
      (ins "const ::mlir::MemorySlot &":$slot)
    >,
    InterfaceMethod<[{
        Gets whether this operation stores to the specified slot.

        No IR mutation is allowed in this method.
      }],
      "bool", "storesTo",
      (ins "const ::mlir::MemorySlot &":$slot)
    >,
    InterfaceMethod<[{
        Gets the value stored to the provided memory slot, or returns a null
        value if this operation does not store to this slot. An operation
        storing a value to a slot must always be able to provide the value it
        stores. This method is only called once per slot promotion, and only
        on operations that store to the slot according to the `storesTo` method.
        The returned value must dominate all operations dominated by the storing
        operation.

        The builder is located immediately after the memory operation on call.
        No IR deletion is allowed in this method. IR mutations must not
        introduce new uses of the memory slot. Existing control flow must not
        be modified.
      }],
      "::mlir::Value", "getStored",
      (ins "const ::mlir::MemorySlot &":$slot,
           "::mlir::OpBuilder &":$builder,
           "::mlir::Value":$reachingDef,
           "const ::mlir::DataLayout &":$dataLayout)
    >,
    InterfaceMethod<[{
        Checks that this operation can be promoted to no longer use the provided
        blocking uses, in the context of promoting `slot`.

        If the removal procedure of the use will require that other uses get
        removed, that dependency should be added to the `newBlockingUses`
        argument. Dependent uses must only be uses of results of this operation.

        No IR mutation is allowed in this method.
      }], "bool", "canUsesBeRemoved",
      (ins "const ::mlir::MemorySlot &":$slot,
           "const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &":$blockingUses,
           "::llvm::SmallVectorImpl<::mlir::OpOperand *> &":$newBlockingUses,
           "const ::mlir::DataLayout &":$datalayout)
    >,
    InterfaceMethod<[{
        Transforms IR to ensure that the current operation does not use the
        provided memory slot anymore. `reachingDefinition` contains the value
        currently stored in the provided memory slot, immediately before the
        current operation.

        During the transformation, *no operation should be deleted*.
        The operation can only schedule its own deletion by returning the
        appropriate `DeletionKind`. The deletion must be legal assuming the
        blocking uses passed through the `newBlockingUses` list in
        `canUseBeRemoved` have been removed.

        After calling this method, the blocking uses should have disappeared
        or this operation should have scheduled its own deletion.

        This method will only be called after ensuring promotion is allowed via
        `canUseBeRemoved`. The requested blocking use removal may or may not
        have been done at the point of calling this method, but it will be done
        eventually.

        The builder is located after the promotable operation on call.
      }],
      "::mlir::DeletionKind",
      "removeBlockingUses",
      (ins "const ::mlir::MemorySlot &":$slot,
           "const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> &":$blockingUses,
           "::mlir::OpBuilder &":$builder,
           "::mlir::Value":$reachingDefinition,
           "const ::mlir::DataLayout &":$dataLayout)
    >,
  ];
}

def PromotableOpInterface : OpInterface<"PromotableOpInterface"> {
  let description = [{
    Describes an operation that can be transformed or deleted so it no longer
    uses a provided value (blocking use), in case this would allow the promotion
    of a memory slot.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Checks that this operation can be promoted to no longer use the provided
        blocking uses, in order to allow optimization.

        If the removal procedure of the use will require that other uses get
        removed, that dependency should be added to the `newBlockingUses`
        argument. Dependent uses must only be uses of results of this operation.

        No IR mutation is allowed in this method.
      }], "bool", "canUsesBeRemoved",
      (ins "const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &":$blockingUses,
           "::llvm::SmallVectorImpl<::mlir::OpOperand *> &":$newBlockingUses,
           "const ::mlir::DataLayout &":$datalayout)
    >,
    InterfaceMethod<[{
        Transforms IR to ensure that the current operation does not use the
        provided blocking uses anymore. In contrast to
        `PromotableMemOpInterface`, operations implementing this interface
        must not need access to the reaching definition of the content of the
        slot.

        During the transformation, *no operation should be deleted*.
        The operation can only schedule its own deletion by returning the
        appropriate `DeletionKind`. The deletion must be legal assuming the
        blocking uses passed through the `newBlockingUses` list in
        `canUseBeRemoved` have been removed.

        After calling this method, the blocking uses should have disappeared
        or this operation should have scheduled its own deletion.

        This method will only be called after ensuring promotion is allowed via
        `canUseBeRemoved`. The requested blocking use removal may or may not
        have been done at the point of calling this method, but it will be done
        eventually.

        The builder is located after the promotable operation on call.
      }],
      "::mlir::DeletionKind",
      "removeBlockingUses",
      (ins "const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> &":$blockingUses,
           "::mlir::OpBuilder &":$builder)
    >,
    InterfaceMethod<[{
        Indicates whether the promoted operation needs to visit the SSA values
        that replace the memory slot.

        If true, `visitReplacedValues` will be called once for this operation
        after all reaching definitions have been computed but before any
        blocking uses are removed.

        This should only return true if the intended transformation applies
        to all replaced values. Replaced values must not be deleted.
      }], "bool", "requiresReplacedValues", (ins), [{}],
      [{ return false; }]
    >,
    InterfaceMethod<[{
        Transforms the IR using the SSA values replacing the memory slot.

        Called after all reaching definitions for the slot have been computed
        but before any blocking uses are removed, provided
        `requiresReplacedValues` returns true. `mutatedDefs` contains the
        replacing values. Any new uses of a load result created here will be
        automatically redirected to its reaching definition when the load is
        replaced.

        The builder is positioned after the promotable operation.
        No operations may be deleted during this transformation.
      }],
      "void", "visitReplacedValues",
      (ins "::llvm::ArrayRef<std::pair<::mlir::Operation*, ::mlir::Value>>":$mutatedDefs,
           "::mlir::OpBuilder &":$builder), [{}], [{ return; }]
    >,
  ];
}

def PromotableAliaserInterface : OpInterface<"PromotableAliaserInterface"> {
  let description = [{
    Describes an operation that creates a transparent alias of a memory slot
    accessed through one of its operands. Mem2Reg traverses chains of these
    aliases to project slot values across them. This allows load and store
    operations on the alias to be promoted as if they were directly accessing
    the underlying slot.

    Since an alias remains a blocking use of the underlying slot pointer, the
    operation must also implement either `PromotableOpInterface` or
    `PromotableMemOpInterface`. This ensures that mem2reg can remove the alias
    after the slot has been promoted.
  }];
  let cppNamespace = "::mlir";

  let verify = [{
    if (!::mlir::isa<::mlir::PromotableOpInterface,
                     ::mlir::PromotableMemOpInterface>($_op))
      return $_op->emitOpError(
          "implements `PromotableAliaserInterface` but must also implement "
          "`PromotableOpInterface` or `PromotableMemOpInterface`.");
    return ::mlir::success();
  }];

  let methods = [
    InterfaceMethod<[{
        Populates `newMemorySlots` with the memory slots this operation
        exposes by aliasing `parentSlot` (accessed via
        `aliasedSlotPointerOperand`). Each new slot's pointer must be a
        result of this operation, and its element type may differ from the
        parent's. Leave the vector empty if no alias is exposed. An operation
        can expose multiple aliases for the same parent slot.

        `parentSlot` is provided so that aliasers using opaque pointers can
        derive the new slot's element type from `parentSlot.elemType`.

        Exposing an alias requires implementing the two projection methods
        below to bridge values between the parent and new slot element types.
        If these projections cannot be performed, `newMemorySlots` must be left
        empty.

        No IR mutation is allowed in this method.
      }],
      "void",
      "getPromotableSlotAliases",
      (ins "::mlir::OpOperand &":$aliasedSlotPointerOperand,
           "const ::mlir::MemorySlot &":$parentSlot,
           "::llvm::SmallVectorImpl<::mlir::MemorySlot> &":$newMemorySlots)
    >,
    InterfaceMethod<[{
        Extracts the value of `aliasSlot` from `slotValue` (the value of
        `parentSlot`). Mem2Reg invokes this method when a load on the new slot
        requires the parent's value to be materialized with the new slot's
        element type.
      }],
      "::mlir::Value",
      "projectSlotValueToAliasValue",
      (ins "::mlir::OpOperand &":$aliasedSlotPointerOperand,
           "const ::mlir::MemorySlot &":$parentSlot,
           "const ::mlir::MemorySlot &":$aliasSlot,
           "::mlir::Value":$slotValue,
           "::mlir::OpBuilder &":$builder), [{}],
      [{ return slotValue; }]
    >,
    InterfaceMethod<[{
        Reconstructs the value of `parentSlot` from `aliasValue` (a store to
        `aliasSlot`) and `reachingDef` (the parent slot's value immediately
        preceding the store). For full aliases, `reachingDef` can be ignored.
        For partial sub-aliases, it allows the result to be constructed by
        inserting `aliasValue` into `reachingDef`.
      }],
      "::mlir::Value",
      "projectAliasValueToSlotValue",
      (ins "::mlir::OpOperand &":$aliasedSlotPointerOperand,
           "const ::mlir::MemorySlot &":$parentSlot,
           "const ::mlir::MemorySlot &":$aliasSlot,
           "::mlir::Value":$aliasValue,
           "::mlir::Value":$reachingDef,
           "::mlir::OpBuilder &":$builder), [{}],
      [{ return aliasValue; }]
    >,
  ];
}

def PromotableRegionOpInterface
    : OpInterface<"PromotableRegionOpInterface"> {
  let description = [{
    Describes an operation for which memory slots can be promoted to SSA values
    within the operation's regions.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns true when the provided region of the operation can be analyzed
        for promotion. The provided region must be a child of the operation's
        region.
        The `hasValueStores` flag indicates whether the region contains
        store-like operations that write to the memory slot.
      }], "bool", "isRegionPromotable",
      (ins
        "const ::mlir::MemorySlot &":$slot,
        "::mlir::Region *":$region,
        "bool":$hasValueStores
      )
    >,
    InterfaceMethod<[{
        Called before processing the nested regions in the operation.

        Based on the `reachingDef` value representing the value in the memory
        slot at the entry into the operation, `setupPromotion` fills in the
        `regionsToProcess` with the reaching definition at the entry of
        all its promotable regions.

        `setupPromotion` is allowed to mutate
        the operation in place, including its nested regions, but cannot
        delete existing operations or modify successor-bearing terminators.
        Other mutations are not allowed.

        The `hasValueStores` flag indicates whether the regions contain
        `store`-like operations that write to the memory slot. This field can be
        used to reduce the amount of book-keeping required to track the reaching
        definitions.
      }], "void", "setupPromotion",
      (ins
        "const ::mlir::MemorySlot &":$slot,
        "::mlir::Value":$reachingDef,
        "bool":$hasValueStores,
        "::llvm::SmallMapVector<::mlir::Region *, ::mlir::Value, 2> &":$regionsToProcess
      )
    >,
    InterfaceMethod<[{
        Called once the reaching definitions have been computed for all the
        regions, but before the actual removal of the blocking uses.

        Returns the new reaching definition at the exit of the operation. For
        this purpose, mutation of the operation is allowed under the following
        constraints:
        1. If a region is deleted, all of its content must have been moved out
           (not copied) to a new empty region that remains valid after the
           deletion.
        2. Mutation must not change control flow within or between existing or
           moved regions. This includes adding, removing or reordering blocks.
        3. Mutation must not modify or add operations that interact with the
           value of the slot.

        As an example, in order to add new results to the region operation, it
        is allowed to clone the operation without regions, move (without
        copying) the old region content into the new regions, and delete the
        original operation.

        The `entryReachingDef` is the reaching definition at the entry of the
        region operation.

        The `reachingAtBlockEnd` map contains the reaching definitions after all
        the terminators within the regions of the operation. If a block of the
        region is not present in the map, it is either dead code or within a
        region that does not interact with the value of the slot.

        The `hasValueStores` flag indicates whether the regions contain
        `store`-like operations that write to the memory slot. This field can be
        used to reduce the amount of book-keeping required to track the reaching
        definitions.
      }],
      "::mlir::Value", "finalizePromotion",
      (ins
        "const ::mlir::MemorySlot &":$slot,
        "::mlir::Value":$entryReachingDef,
        "bool":$hasValueStores,
        "const ::llvm::DenseMap<::mlir::Block *, ::mlir::Value> &":$reachingAtBlockEnd,
        "::mlir::OpBuilder &":$builder
      )
    >,
  ];
}

def DestructurableAllocationOpInterface
  : OpInterface<"DestructurableAllocationOpInterface"> {
  let description = [{
    Describes operations allocating memory slots of aggregates that can be
    destructured into multiple smaller allocations.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the list of slots for which destructuring should be attempted,
        specifying in which way the slot should be destructured into subslots.
        The subslots are indexed by attributes. This computes the type of the
        pointer for each subslot to be generated. The type of the memory slot
        must implement `DestructurableTypeInterface`.

        No IR mutation is allowed in this method.
      }],
      "::llvm::SmallVector<::mlir::DestructurableMemorySlot>",
      "getDestructurableSlots",
      (ins)
    >,
    InterfaceMethod<[{
        Destructures this slot into multiple subslots. The newly generated slots
        may belong to a different allocator. The original slot must still exist
        at the end of this call. Only generates subslots for the indices found in
        `usedIndices` since all other subslots are unused.

        The builder is located at the beginning of the block where the slot
        pointer is defined.
      }],
      "::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot>",
      "destructure",
      (ins "const ::mlir::DestructurableMemorySlot &":$slot,
           "const ::llvm::SmallPtrSetImpl<::mlir::Attribute> &":$usedIndices,
           "::mlir::OpBuilder &":$builder,
           "::mlir::SmallVectorImpl<::mlir::DestructurableAllocationOpInterface> &":
             $newAllocators)
    >,
    InterfaceMethod<[{
        Hook triggered once the destructuring of a slot is complete, meaning the
        original slot is no longer being referred to and could be deleted.
        This will only be called for slots declared by this operation.

        Must return a new destructurable allocation op if this hook creates
        a new destructurable op, nullopt otherwise.
      }],
      "::std::optional<::mlir::DestructurableAllocationOpInterface>",
      "handleDestructuringComplete",
      (ins "const ::mlir::DestructurableMemorySlot &":$slot,
           "::mlir::OpBuilder &":$builder)
    >,
  ];
}

def SafeMemorySlotAccessOpInterface
  : OpInterface<"SafeMemorySlotAccessOpInterface"> {
  let description = [{
    Describes operations using memory slots in a safe manner.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns whether all accesses in this operation to the provided slot are
        done in a safe manner. To be safe, the access must only access the slot
        inside the bounds that its type implies.

        If the safety of the accesses depends on the safety of the accesses to
        further memory slots, the result of this method will be conditioned to
        the safety of the accesses to the slots added by this method to
        `mustBeSafelyUsed`.

        No IR mutation is allowed in this method.
      }],
      "::llvm::LogicalResult",
      "ensureOnlySafeAccesses",
      (ins "const ::mlir::MemorySlot &":$slot,
           "::mlir::SmallVectorImpl<::mlir::MemorySlot> &":$mustBeSafelyUsed,
           "const ::mlir::DataLayout &":$dataLayout)
    >
  ];
}

def DestructurableAccessorOpInterface
  : OpInterface<"DestructurableAccessorOpInterface"> {
  let description = [{
    Describes operations that can access a sub-element of a destructurable slot.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        For a given destructurable memory slot, returns whether this operation can
        rewire its uses of the slot to use the slots generated after
        destructuring. This may involve creating new operations.

        This method must also register the indices it will access within the
        `usedIndices` set. If the accessor generates new slots mapping to
        subelements, they must be registered in `mustBeSafelyUsed` to ensure
        they are used in a safe manner.

        No IR mutation is allowed in this method.
      }],
      "bool",
      "canRewire",
      (ins "const ::mlir::DestructurableMemorySlot &":$slot,
           "::llvm::SmallPtrSetImpl<::mlir::Attribute> &":$usedIndices,
           "::mlir::SmallVectorImpl<::mlir::MemorySlot> &":$mustBeSafelyUsed,
           "const ::mlir::DataLayout &":$dataLayout)
    >,
    InterfaceMethod<[{
        Rewires the use of a slot to the generated subslots, without deleting
        any operation. Returns whether the accessor should be deleted.

        Deletion of operations is not allowed, only the accessor can be
        scheduled for deletion by returning the appropriate value.
      }],
      "::mlir::DeletionKind",
      "rewire",
      (ins "const ::mlir::DestructurableMemorySlot &":$slot,
           "::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> &":$subslots,
           "::mlir::OpBuilder &":$builder,
           "const ::mlir::DataLayout &":$dataLayout)
    >
  ];
}

def DestructurableTypeInterface
  : TypeInterface<"DestructurableTypeInterface"> {
  let description = [{
    Describes a type that can be broken down into indexable sub-element types.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Destructures the type into subelements into a map of index attributes to
        types of subelements. Returns nothing if the type cannot be destructured.
      }],
      "::std::optional<::llvm::DenseMap<::mlir::Attribute, ::mlir::Type>>",
      "getSubelementIndexMap",
      (ins)
    >,
    InterfaceMethod<[{
        Indicates which type is held at the provided index, returning a null
        Type if no type could be computed. While this can return information
        even when the type cannot be completely destructured, it must be coherent
        with the types returned by `getSubelementIndexMap` when they exist.
      }],
      "::mlir::Type",
      "getTypeAtIndex",
      (ins "::mlir::Attribute":$index)
    >
  ];
}

#endif // MLIR_INTERFACES_MEMORYSLOTINTERFACES
