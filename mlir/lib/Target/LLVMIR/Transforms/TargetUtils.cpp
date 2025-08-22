//===- TargetUtils.cpp - utils for obtaining generic target backend info --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Transforms/Passes.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "mlir-llvm-target-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

llvm::once_flag initializeBackendsOnce;

namespace mlir {
namespace LLVM {
namespace detail {
void initializeBackendsOnce() {
  static llvm::once_flag initializeOnceFlag;
  llvm::call_once(initializeOnceFlag, []() {
    // Ensure that the targets, that LLVM has been configured to support,
    // are loaded into the TargetRegistry.
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
  });
}

FailureOr<std::unique_ptr<llvm::TargetMachine>>
getTargetMachine(mlir::LLVM::TargetAttrInterface attr) {
  StringRef triple = attr.getTriple();
  StringRef chipAKAcpu = attr.getChip();
  std::string features =
      attr.getFeatures() ? attr.getFeatures().getFeaturesString() : "";

  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target || !error.empty()) {
    LDBG("Looking up target '" << triple << "' failed: " << error << "\n");
    return failure();
  }

  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      llvm::Triple(triple), chipAKAcpu, features, {}, {}));
}

FailureOr<llvm::DataLayout>
getDataLayout(mlir::LLVM::TargetAttrInterface attr) {
  FailureOr<std::unique_ptr<llvm::TargetMachine>> targetMachine =
      getTargetMachine(attr);
  if (failed(targetMachine)) {
    LDBG("Failed to retrieve the target machine for data layout.\n");
    return failure();
  }
  return (targetMachine.value())->createDataLayout();
}

} // namespace detail
} // namespace LLVM
} // namespace mlir
