//===- DeduceVersionExtensionCapabilityPass.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to deduce minimal version/extension/capability
// requirements for a spirv::ModuleOp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;

namespace {
/// Pass to deduce minimal version/extension/capability requirements for a
/// spirv::ModuleOp.
class UpdateVCEPass final
    : public OperationPass<UpdateVCEPass, spirv::ModuleOp> {
private:
  void runOnOperation() override;
};
} // namespace

void UpdateVCEPass::runOnOperation() {
  spirv::ModuleOp module = getOperation();

  spirv::TargetEnvAttr targetEnv = spirv::lookupTargetEnv(module);
  if (!targetEnv) {
    module.emitError("missing 'spv.target_env' attribute");
    return signalPassFailure();
  }

  spirv::Version allowedVersion = targetEnv.getVersion();

  // Build a set for available extensions in the target environment.
  llvm::SmallSet<spirv::Extension, 4> allowedExtensions;
  for (spirv::Extension ext : targetEnv.getExtensions())
    allowedExtensions.insert(ext);

  // Add extensions implied by the current version.
  for (spirv::Extension ext : spirv::getImpliedExtensions(allowedVersion))
    allowedExtensions.insert(ext);

  // Build a set for available capabilities in the target environment.
  llvm::SmallSet<spirv::Capability, 8> allowedCapabilities;
  for (spirv::Capability cap : targetEnv.getCapabilities()) {
    allowedCapabilities.insert(cap);

    // Add capabilities implied by the current capability.
    for (spirv::Capability c : spirv::getRecursiveImpliedCapabilities(cap))
      allowedCapabilities.insert(c);
  }

  spirv::Version deducedVersion = spirv::Version::V_1_0;
  llvm::SetVector<spirv::Extension> deducedExtensions;
  llvm::SetVector<spirv::Capability> deducedCapabilities;

  // Walk each SPIR-V op to deduce the minimal version/extension/capability
  // requirements.
  WalkResult walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (auto minVersion = dyn_cast<spirv::QueryMinVersionInterface>(op)) {
      deducedVersion = std::max(deducedVersion, minVersion.getMinVersion());
      if (deducedVersion > allowedVersion) {
        return op->emitError("'") << op->getName() << "' requires min version "
                                  << spirv::stringifyVersion(deducedVersion)
                                  << " but target environment allows up to "
                                  << spirv::stringifyVersion(allowedVersion);
      }
    }

    // Deduce this op's extension requirement. For each op, the query interfacce
    // returns a vector of vector for its extension requirements following
    // ((Extension::A OR Extension::B) AND (Extension::C OR Extension::D))
    // convention. Ops not implementing QueryExtensionInterface do not require
    // extensions to be available.
    if (auto extensions = dyn_cast<spirv::QueryExtensionInterface>(op)) {
      for (const auto &ors : extensions.getExtensions()) {
        bool satisfied = false; // True when at least one extension can be used
        for (spirv::Extension ext : ors) {
          if (allowedExtensions.count(ext)) {
            deducedExtensions.insert(ext);
            satisfied = true;
            break;
          }
        }

        if (!satisfied) {
          SmallVector<StringRef, 4> extStrings;
          for (spirv::Extension ext : ors)
            extStrings.push_back(spirv::stringifyExtension(ext));

          return op->emitError("'")
                 << op->getName() << "' requires at least one extension in ["
                 << llvm::join(extStrings, ", ")
                 << "] but none allowed in target environment";
        }
      }
    }

    // Deduce this op's capability requirement. For each op, the queryinterface
    // returns a vector of vector for its capability requirements following
    // ((Capability::A OR Extension::B) AND (Capability::C OR Capability::D))
    // convention. Ops not implementing QueryExtensionInterface do not require
    // extensions to be available.
    if (auto capabilities = dyn_cast<spirv::QueryCapabilityInterface>(op)) {
      for (const auto &ors : capabilities.getCapabilities()) {
        bool satisfied = false; // True when at least one capability can be used
        for (spirv::Capability cap : ors) {
          if (allowedCapabilities.count(cap)) {
            deducedCapabilities.insert(cap);
            satisfied = true;
            break;
          }
        }

        if (!satisfied) {
          SmallVector<StringRef, 4> capStrings;
          for (spirv::Capability cap : ors)
            capStrings.push_back(spirv::stringifyCapability(cap));

          return op->emitError("'")
                 << op->getName() << "' requires at least one capability in ["
                 << llvm::join(capStrings, ", ")
                 << "] but none allowed in target environment";
        }
      }
    }

    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return signalPassFailure();

  // TODO(antiagainst): verify that the deduced version is consistent with
  // SPIR-V ops' maximal version requirements.

  auto triple = spirv::VerCapExtAttr::get(
      deducedVersion, deducedCapabilities.getArrayRef(),
      deducedExtensions.getArrayRef(), &getContext());
  module.setAttr("vce_triple", triple);
}

std::unique_ptr<OpPassBase<spirv::ModuleOp>>
mlir::spirv::createUpdateVersionCapabilityExtensionPass() {
  return std::make_unique<UpdateVCEPass>();
}

static PassRegistration<UpdateVCEPass>
    pass("spirv-update-vce",
         "Deduce and attach minimal (version, capabilities, extensions) "
         "requirements to spv.module ops");
