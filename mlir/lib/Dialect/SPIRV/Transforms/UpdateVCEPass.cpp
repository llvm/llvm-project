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

/// Checks that `candidates` extension requirements are possible to be satisfied
/// with the given `allowedExtensions` and updates `deducedExtensions` if so.
/// Emits errors attaching to the given `op` on failures.
///
///  `candidates` is a vector of vector for extension requirements following
/// ((Extension::A OR Extension::B) AND (Extension::C OR Extension::D))
/// convention.
static LogicalResult checkAndUpdateExtensionRequirements(
    Operation *op, const llvm::SmallSet<spirv::Extension, 4> &allowedExtensions,
    const spirv::SPIRVType::ExtensionArrayRefVector &candidates,
    llvm::SetVector<spirv::Extension> &deducedExtensions) {
  for (const auto &ors : candidates) {
    auto chosen = llvm::find_if(ors, [&](spirv::Extension ext) {
      return allowedExtensions.count(ext);
    });

    if (chosen != ors.end()) {
      deducedExtensions.insert(*chosen);
    } else {
      SmallVector<StringRef, 4> extStrings;
      for (spirv::Extension ext : ors)
        extStrings.push_back(spirv::stringifyExtension(ext));

      return op->emitError("'")
             << op->getName() << "' requires at least one extension in ["
             << llvm::join(extStrings, ", ")
             << "] but none allowed in target environment";
    }
  }
  return success();
}

/// Checks that `candidates`capability requirements are possible to be satisfied
/// with the given `allowedCapabilities` and updates `deducedCapabilities` if
/// so. Emits errors attaching to the given `op` on failures.
///
///  `candidates` is a vector of vector for capability requirements following
/// ((Capability::A OR Capability::B) AND (Capability::C OR Capability::D))
/// convention.
static LogicalResult checkAndUpdateCapabilityRequirements(
    Operation *op,
    const llvm::SmallSet<spirv::Capability, 8> &allowedCapabilities,
    const spirv::SPIRVType::CapabilityArrayRefVector &candidates,
    llvm::SetVector<spirv::Capability> &deducedCapabilities) {
  for (const auto &ors : candidates) {
    auto chosen = llvm::find_if(ors, [&](spirv::Capability cap) {
      return allowedCapabilities.count(cap);
    });

    if (chosen != ors.end()) {
      deducedCapabilities.insert(*chosen);
    } else {
      SmallVector<StringRef, 4> capStrings;
      for (spirv::Capability cap : ors)
        capStrings.push_back(spirv::stringifyCapability(cap));

      return op->emitError("'")
             << op->getName() << "' requires at least one capability in ["
             << llvm::join(capStrings, ", ")
             << "] but none allowed in target environment";
    }
  }
  return success();
}

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
    // Op min version requirements
    if (auto minVersion = dyn_cast<spirv::QueryMinVersionInterface>(op)) {
      deducedVersion = std::max(deducedVersion, minVersion.getMinVersion());
      if (deducedVersion > allowedVersion) {
        return op->emitError("'") << op->getName() << "' requires min version "
                                  << spirv::stringifyVersion(deducedVersion)
                                  << " but target environment allows up to "
                                  << spirv::stringifyVersion(allowedVersion);
      }
    }

    // Op extension requirements
    if (auto extensions = dyn_cast<spirv::QueryExtensionInterface>(op))
      if (failed(checkAndUpdateExtensionRequirements(op, allowedExtensions,
                                                     extensions.getExtensions(),
                                                     deducedExtensions)))
        return WalkResult::interrupt();

    // Op capability requirements
    if (auto capabilities = dyn_cast<spirv::QueryCapabilityInterface>(op))
      if (failed(checkAndUpdateCapabilityRequirements(
              op, allowedCapabilities, capabilities.getCapabilities(),
              deducedCapabilities)))
        return WalkResult::interrupt();

    SmallVector<Type, 4> valueTypes;
    valueTypes.append(op->operand_type_begin(), op->operand_type_end());
    valueTypes.append(op->result_type_begin(), op->result_type_end());

    // Special treatment for global variables, whose type requirements are
    // conveyed by type attributes.
    if (auto globalVar = dyn_cast<spirv::GlobalVariableOp>(op))
      valueTypes.push_back(globalVar.type());

    // Requirements from values' types
    SmallVector<ArrayRef<spirv::Extension>, 4> typeExtensions;
    SmallVector<ArrayRef<spirv::Capability>, 8> typeCapabilities;
    for (Type valueType : valueTypes) {
      typeExtensions.clear();
      valueType.cast<spirv::SPIRVType>().getExtensions(typeExtensions);
      if (failed(checkAndUpdateExtensionRequirements(
              op, allowedExtensions, typeExtensions, deducedExtensions)))
        return WalkResult::interrupt();

      typeCapabilities.clear();
      valueType.cast<spirv::SPIRVType>().getCapabilities(typeCapabilities);
      if (failed(checkAndUpdateCapabilityRequirements(
              op, allowedCapabilities, typeCapabilities, deducedCapabilities)))
        return WalkResult::interrupt();
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
  module.setAttr(spirv::ModuleOp::getVCETripleAttrName(), triple);
}

std::unique_ptr<OpPassBase<spirv::ModuleOp>>
mlir::spirv::createUpdateVersionCapabilityExtensionPass() {
  return std::make_unique<UpdateVCEPass>();
}

static PassRegistration<UpdateVCEPass>
    pass("spirv-update-vce",
         "Deduce and attach minimal (version, capabilities, extensions) "
         "requirements to spv.module ops");
