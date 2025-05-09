//===- Target.h - Target information ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declare utilities to interact with LLVM targets by querying an MLIR
// target.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_TARGET_H
#define MLIR_TARGET_LLVM_TARGET_H

#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
class Triple;
class Target;
class TargetMachine;
} // namespace llvm

namespace mlir {
/// Given a target triple. chip and features returns the LLVM data layout.
FailureOr<const llvm::DataLayout>
getLLVMDataLayout(StringRef triple, StringRef chip, StringRef features);

/// Returns the LLVM target triple held by `target`.
llvm::Triple getTargetTriple(TargetAttrInterface target);

/// Returns the LLVM target held by `target`.
FailureOr<const llvm::Target *> getLLVMTarget(TargetAttrInterface target);

/// Helper class for holding LLVM target information. Note: This class requires
/// that the corresponding LLVM target has ben initialized.
class TargetInfo {
public:
  TargetInfo(TargetInfo &&) = default;
  TargetInfo(const TargetInfo &) = delete;
  ~TargetInfo();
  TargetInfo &operator=(TargetInfo &&) = default;
  TargetInfo &operator=(const TargetInfo &) = delete;
  /// Constructs the target info from `target`.
  static FailureOr<TargetInfo> getTargetInfo(StringRef triple, StringRef chip,
                                             StringRef features);

  /// Constructs the target info from `target`.
  static FailureOr<TargetInfo> getTargetInfo(TargetAttrInterface target) {
    return getTargetInfo(target.getTargetTriple(), target.getTargetChip(),
                         target.getTargetFeatures());
  }

  /// Returns the target chip.
  StringRef getTargetChip() const;

  /// Returns the target features.
  StringRef getTargetFeatures() const;

  /// Returns the target triple.
  const llvm::Triple &getTriple() const;

  /// Returns the target.
  const llvm::Target &getTarget() const;

  /// Returns the target machine.
  const llvm::TargetMachine *getTargetMachine() const {
    return targetMachine.get();
  }

  /// Returns the LLVM data layout for the corresponding target.
  const llvm::DataLayout getDataLayout() const;

private:
  TargetInfo(llvm::TargetMachine *targetMachine);
  /// The LLVM target machine.
  mutable std::unique_ptr<llvm::TargetMachine> targetMachine;
};
} // namespace mlir

#endif // MLIR_TARGET_LLVM_TARGET_H
