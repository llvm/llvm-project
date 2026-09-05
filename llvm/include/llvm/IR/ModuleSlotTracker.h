//===-- llvm/IR/ModuleSlotTracker.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_MODULESLOTTRACKER_H
#define LLVM_IR_MODULESLOTTRACKER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <functional>
#include <memory>
#include <utility>

namespace llvm {

class Module;
class Function;
class SlotTracker;
class Value;
class DILocation;
class MDNode;

/// Abstract interface of slot tracker storage.
class LLVM_ABI AbstractSlotTrackerStorage {
public:
  virtual ~AbstractSlotTrackerStorage();

  virtual void createMetadataSlot(const MDNode *) = 0;
  virtual int getMetadataSlot(const MDNode *) = 0;
};

/// Manage lifetime of a slot tracker for printing IR.
///
/// Wrapper around the \a SlotTracker used internally by \a AsmWriter.  This
/// class allows callers to share the cost of incorporating the metadata in a
/// module or a function.
///
/// If the IR changes from underneath \a ModuleSlotTracker, strings like
/// "<badref>" will be printed, or, worse, the wrong slots entirely.
class LLVM_ABI ModuleSlotTracker {
public:
  using MachineMDNodeListType =
      SmallVector<std::pair<unsigned, const MDNode *>, 0>;

private:
  /// Storage for a slot tracker.
  std::unique_ptr<SlotTracker> MachineStorage;
  bool ShouldCreateStorage = false;

  const Module *M = nullptr;
  const Function *F = nullptr;
  SlotTracker *Machine = nullptr;

  std::function<void(AbstractSlotTrackerStorage *, const Module *)>
      ProcessModuleHookFn;
  std::function<void(AbstractSlotTrackerStorage *, const Function *)>
      ProcessFunctionHookFn;

protected:
  /// Renumber module metadata and then additional metadata for canonical
  /// assembly output.
  void renumberMetadataForAssembly(
      ArrayRef<const MDNode *> AdditionalMetadata,
      MachineMDNodeListType *AdditionalMetadataNodes = nullptr) const;

  /// Collect metadata reachable from \p AdditionalMetadata but not from the
  /// module.
  void collectAdditionalMetadata(
      ArrayRef<const MDNode *> AdditionalMetadata,
      MachineMDNodeListType &AdditionalMetadataNodes) const;

public:
  /// Wrap a preinitialized SlotTracker.
  ModuleSlotTracker(SlotTracker &Machine, const Module *M,
                    const Function *F = nullptr);

  /// Construct a slot tracker from a module.
  ///
  /// If \a M is \c nullptr, uses a null slot tracker.
  explicit ModuleSlotTracker(const Module *M);

  /// Destructor to clean up storage.
  virtual ~ModuleSlotTracker();

  /// Lazily creates a slot tracker.
  SlotTracker *getMachine();

  const Module *getModule() const { return M; }
  const Function *getCurrentFunction() const { return F; }

  /// Incorporate the given function.
  ///
  /// Purge the currently incorporated function and incorporate \c F.  If \c F
  /// is currently incorporated, this is a no-op.
  void incorporateFunction(const Function &F);

  /// Return the slot number of the specified local value.
  ///
  /// A function that defines this value should be incorporated prior to calling
  /// this method.
  /// Return -1 if the value is not in the function's SlotTracker.
  int getLocalSlot(const Value *V);

  void setProcessHook(
      std::function<void(AbstractSlotTrackerStorage *, const Module *)>);
  void setProcessHook(
      std::function<void(AbstractSlotTrackerStorage *, const Function *)>);

  void collectMDNodes(MachineMDNodeListType &L) const;

  /// Return whether a debug location should be printed inline instead of by ID.
  virtual bool shouldPrintDebugLocationInline(const DILocation *) const {
    return false;
  }
};

} // end namespace llvm

#endif
