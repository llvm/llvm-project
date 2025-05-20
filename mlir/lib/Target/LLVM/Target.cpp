//===- Target.cpp - Target information --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities to interact with LLVM targets by querying an MLIR
// target.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/Target.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "mlir-llvm-target"

using namespace mlir;

llvm::Triple getTargetTriple(TargetAttrInterface target) {
  return llvm::Triple(target.getTargetTriple());
}

static FailureOr<const llvm::Target *> getLLVMTarget(StringRef triple) {
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (error.empty())
    return target;
  LLVM_DEBUG({
    llvm::dbgs() << "Failed to retrieve the target with: `" << error << "`\n";
  });
  return failure();
}

FailureOr<const llvm::Target *>
mlir::getLLVMTarget(TargetAttrInterface target) {
  return ::getLLVMTarget(target.getTargetTriple());
}

FailureOr<const llvm::DataLayout>
mlir::getLLVMDataLayout(StringRef triple, StringRef chip, StringRef features) {
  FailureOr<const llvm::Target *> target = ::getLLVMTarget(triple);
  if (failed(target))
    return failure();
  std::unique_ptr<llvm::TargetMachine> tgtMachine(
      (*target)->createTargetMachine(llvm::Triple(triple), chip, features, {},
                                     {}));
  return tgtMachine->createDataLayout();
}

FailureOr<TargetInfo> TargetInfo::getTargetInfo(StringRef triple,
                                                StringRef chip,
                                                StringRef features) {
  FailureOr<const llvm::Target *> llvmTgt = ::getLLVMTarget(triple);
  if (failed(llvmTgt))
    return failure();
  return FailureOr<TargetInfo>(TargetInfo((*llvmTgt)->createTargetMachine(
      llvm::Triple(triple), chip, features, {}, {})));
}

TargetInfo::TargetInfo(llvm::TargetMachine *targetMachine)
    : targetMachine(targetMachine) {}

TargetInfo::~TargetInfo() = default;

StringRef TargetInfo::getTargetChip() const {
  return targetMachine->getTargetCPU();
}

StringRef TargetInfo::getTargetFeatures() const {
  return targetMachine->getTargetFeatureString();
}

const llvm::Triple &TargetInfo::getTriple() const {
  return targetMachine->getTargetTriple();
}

const llvm::Target &TargetInfo::getTarget() const {
  return targetMachine->getTarget();
}

const llvm::DataLayout TargetInfo::getDataLayout() const {
  return targetMachine->createDataLayout();
}
