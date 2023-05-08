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
      (ins "const ::mlir::MemorySlot &":$slot, "::mlir::OpBuilder &":$builder)
    >,
    InterfaceMethod<[{
        Hook triggered for every new block argument added to a block.
        This will only be called for slots declared by this operation.

        The builder is located at the beginning of the block on call.
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
        also clean up the created default value if necessary.
        This will only be called for slots declared by this operation.
      }],
      "void", "handlePromotionComplete",
      (ins "const ::mlir::MemorySlot &":$slot, "::mlir::Value":$defaultValue)
    >,
  ];
}

def PromotableMemOpInterface : OpInterface<"PromotableMemOpInterface"> {
  let description = [{
    Describes an operation that can load from memory slots and/or store
    to memory slots. Loads and stores must be of whole values of the same
    type as the slot itself.

    For a memory operation on a slot to be valid, it must operate on the slot
    pointer *only as a pointer to an element of the type of the slot*.

    If the same operation does both loads and stores on the same slot, the
    load must semantically happen first.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Gets whether this operation loads from the specified slot.
      }],
      "bool", "loadsFrom",
      (ins "const ::mlir::MemorySlot &":$slot)
    >,
    InterfaceMethod<[{
        Gets the value stored to the provided memory slot, or returns a null
        value if this operation does not store to this slot. An operation
        storing a value to a slot must always be able to provide the value it
        stores. This method is only called on operations that use the slot.
      }],
      "::mlir::Value", "getStored",
      (ins "const ::mlir::MemorySlot &":$slot)
    >,
    InterfaceMethod<[{
        Checks that this operation can be promoted to no longer use the provided
        blocking uses, in the context of promoting `slot`.

        If the removal procedure of the use will require that other uses get
        removed, that dependency should be added to the `newBlockingUses`
        argument. Dependent uses must only be uses of results of this operation.
      }], "bool", "canUsesBeRemoved",
      (ins "const ::mlir::MemorySlot &":$slot,
           "const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &":$blockingUses,
           "::llvm::SmallVectorImpl<::mlir::OpOperand *> &":$newBlockingUses)
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
           "::mlir::Value":$reachingDefinition)
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
      }], "bool", "canUsesBeRemoved",
      (ins "const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &":$blockingUses,
           "::llvm::SmallVectorImpl<::mlir::OpOperand *> &":$newBlockingUses)
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
  ];
}

#endif // MLIR_INTERFACES_MEMORYSLOTINTERFACES
