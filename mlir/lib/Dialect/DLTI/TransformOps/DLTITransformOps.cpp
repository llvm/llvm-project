
//===- DLTITransformOps.cpp - Implementation of DLTI transform ops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/TransformOps/DLTITransformOps.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/Utils.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

using namespace mlir;
using namespace mlir::transform;

#define DEBUG_TYPE "dlti-transforms"

//===----------------------------------------------------------------------===//
// QueryOp
//===----------------------------------------------------------------------===//

void transform::QueryOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure transform::QueryOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results, TransformState &state) {
  StringAttr deviceId = getDeviceAttr();
  StringAttr key = getKeyAttr();

  DataLayoutEntryInterface entry;
  if (deviceId) {
    TargetSystemSpecInterface sysSpec = dlti::getTargetSystemSpec(target);
    if (!sysSpec)
      return mlir::emitDefiniteFailure(target->getLoc())
             << "no target system spec associated to: " << target;

    if (auto targetSpec = sysSpec.getDeviceSpecForDeviceID(deviceId))
      entry = targetSpec->getSpecForIdentifier(key);
    else
      return mlir::emitDefiniteFailure(target->getLoc())
             << "no " << deviceId << " target device spec found";
  } else {
    DataLayoutSpecInterface dlSpec = dlti::getDataLayoutSpec(target);
    if (!dlSpec)
      return mlir::emitDefiniteFailure(target->getLoc())
             << "no data layout spec associated to: " << target;

    entry = dlSpec.getSpecForIdentifier(key);
  }

  if (!entry)
    return mlir::emitDefiniteFailure(target->getLoc())
           << "no DLTI entry for key: " << key;

  results.push_back(entry.getValue());

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class DLTITransformDialectExtension
    : public transform::TransformDialectExtension<
          DLTITransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DLTITransformDialectExtension)

  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/DLTI/TransformOps/DLTITransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/DLTI/TransformOps/DLTITransformOps.cpp.inc"

void mlir::dlti::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<DLTITransformDialectExtension>();
}
