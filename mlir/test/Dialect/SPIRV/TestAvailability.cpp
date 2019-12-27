//===- TestAvailability.cpp - Pass to test SPIR-V op availability ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// A pass for testing SPIR-V op availability.
struct TestAvailability : public FunctionPass<TestAvailability> {
  void runOnFunction() override;
};
} // end anonymous namespace

void TestAvailability::runOnFunction() {
  auto f = getFunction();
  llvm::outs() << f.getName() << "\n";

  Dialect *spvDialect = getContext().getRegisteredDialect("spv");

  f.getOperation()->walk([&](Operation *op) {
    if (op->getDialect() != spvDialect)
      return WalkResult::advance();

    auto &os = llvm::outs();

    if (auto minVersion = dyn_cast<spirv::QueryMinVersionInterface>(op))
      os << " min version: "
         << spirv::stringifyVersion(minVersion.getMinVersion()) << "\n";

    if (auto maxVersion = dyn_cast<spirv::QueryMaxVersionInterface>(op))
      os << " max version: "
         << spirv::stringifyVersion(maxVersion.getMaxVersion()) << "\n";

    if (auto extension = dyn_cast<spirv::QueryExtensionInterface>(op)) {
      os << " extensions: [";
      for (const auto &exts : extension.getExtensions()) {
        os << " [";
        interleaveComma(exts, os, [&](spirv::Extension ext) {
          os << spirv::stringifyExtension(ext);
        });
        os << "]";
      }
      os << " ]\n";
    }

    if (auto capability = dyn_cast<spirv::QueryCapabilityInterface>(op)) {
      os << " capabilities: [";
      for (const auto &caps : capability.getCapabilities()) {
        os << " [";
        interleaveComma(caps, os, [&](spirv::Capability cap) {
          os << spirv::stringifyCapability(cap);
        });
        os << "]";
      }
      os << " ]\n";
    }
    os.flush();

    return WalkResult::advance();
  });
}

static PassRegistration<TestAvailability> pass("test-spirv-op-availability",
                                               "Test SPIR-V op availability");
