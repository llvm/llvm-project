//===- NVVMAttributes.cpp - NVVM enums and attributes ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the NVVM dialect enums and attributes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

using namespace mlir;
using namespace NVVM;

#include "mlir/Dialect/LLVMIR/NVVMOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// NVVM Address Space Attr
//===----------------------------------------------------------------------===//

unsigned NVVMMemorySpaceAttr::getAddressSpace() const {
  return static_cast<unsigned>(getValue());
}

bool NVVMMemorySpaceAttr::isValidLoad(
    Type type, ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return LLVM::detail::isValidLoadStoreImpl(type, ordering, alignment,
                                            dataLayout, emitError);
}

bool NVVMMemorySpaceAttr::isValidStore(
    Type type, ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return LLVM::detail::isValidLoadStoreImpl(type, ordering, alignment,
                                            dataLayout, emitError);
}

bool NVVMMemorySpaceAttr::isValidAtomicOp(
    ptr::AtomicBinOp op, Type type, ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once `ptr.atomic_rmw` is implemented.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

bool NVVMMemorySpaceAttr::isValidAtomicXchg(
    Type type, ptr::AtomicOrdering successOrdering,
    ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once `ptr.atomic_cmpxchg` is implemented.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

bool NVVMMemorySpaceAttr::isValidAddrSpaceCast(
    Type tgt, Type src, function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once the `ptr.addrspace_cast` op is added to the
  // dialect.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

bool NVVMMemorySpaceAttr::isValidPtrIntCast(
    Type intLikeTy, Type ptrLikeTy,
    function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once the int-cast ops are added to the `ptr`
  // dialect.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

//===----------------------------------------------------------------------===//
// NVVM target attribute.
//===----------------------------------------------------------------------===//
LogicalResult
NVVMTargetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       int optLevel, StringRef triple, StringRef chip,
                       StringRef features, DictionaryAttr flags,
                       ArrayAttr files, bool verifyTarget) {
  if (optLevel < 0 || optLevel > 3) {
    emitError() << "The optimization level must be a number between 0 and 3.";
    return failure();
  }
  if (triple.empty()) {
    emitError() << "The target triple cannot be empty.";
    return failure();
  }
  if (chip.empty()) {
    emitError() << "The target chip cannot be empty.";
    return failure();
  }
  if (files && !llvm::all_of(files, [](::mlir::Attribute attr) {
        return mlir::isa_and_nonnull<StringAttr>(attr);
      })) {
    emitError() << "All the elements in the `link` array must be strings.";
    return failure();
  }
  return success();
}

LogicalResult NVVMTargetAttr::verifyTarget(Operation *gpuModule) {
  if (!getVerifyTarget())
    return success();

  auto gpuModuleOp = llvm::dyn_cast<gpu::GPUModuleOp>(gpuModule);
  if (!gpuModuleOp) {
    return emitError(gpuModule->getLoc(),
                     "NVVM target attribute must be attached to a GPU module");
  }

  std::optional<unsigned> targetFullSmVersion =
      NVVMCheckSMVersion::getTargetFullSmVersionFromStr(getChip());
  if (!targetFullSmVersion)
    return emitError(gpuModule->getLoc())
           << "invalid NVVM target chip \"" << getChip()
           << "\", expected sm_<version>[a|f]";

  if (!NVVMCheckSMVersion::isMinimumSMVersion(*targetFullSmVersion)) {
    return emitError(gpuModule->getLoc(),
                     "Minimum NVVM target SM version is sm_20");
  }

  if (gpuModuleOp
          ->walk([&](Operation *op) {
            if (auto reqOp = llvm::dyn_cast<NVVM::RequiresSMInterface>(op)) {
              const NVVMCheckSMVersion requirement =
                  reqOp.getRequiredMinSMVersion();
              if (!requirement.isCompatibleWith(*targetFullSmVersion)) {
                op->emitOpError() << "is not supported on " << getChip();
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          })
          .wasInterrupted())
    return failure();

  return success();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// NVVMDialect attribute registration
//===----------------------------------------------------------------------===//

void NVVMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/NVVMOpsAttributes.cpp.inc"
      >();
}
