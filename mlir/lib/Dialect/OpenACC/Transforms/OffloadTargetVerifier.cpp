//===- OffloadTargetVerifier.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass verifies that values and symbols used within offload regions are
// legal for the target execution model.
//
// Overview:
// ---------
// Offload regions execute on a target device (e.g., GPU) where not all values
// and symbols from the host context are accessible. This pass checks that
// live-in values (values defined outside but used inside the region) and
// symbol references are valid for device execution.
//
// The pass operates on any operation implementing `OffloadRegionOpInterface`,
// which includes OpenACC compute constructs (`acc.parallel`, `acc.kernels`,
// `acc.serial`) as well as GPU operations like `gpu.launch`.
//
// Verification:
// -------------
// For each offload region, the pass checks:
//
// 1. Live-in Values: Values flowing into the region must be valid for device
//    use. This includes checking that data has been properly mapped via
//    OpenACC data clauses (copyin, copyout, present, etc.) or is a scalar
//    that can be passed by value.
//
// 2. Symbol References: Symbols referenced inside the region must be
//    accessible on the device. This includes checking for proper `declare`
//    attributes on globals or device-resident data attributes.
//
// Requirements:
// -------------
// 1. Target Region Identification: Operations representing offload regions
//    must implement `acc::OffloadRegionOpInterface`.
//
// 2. OpenACCSupport Analysis: The pass relies on the `OpenACCSupport`
//    analysis to determine value and symbol validity. This analysis provides
//    dialect-specific hooks for checking legality through `isValidValueUse`
//    and `isValidSymbolUse` methods. Custom dialect support can be registered
//    by providing a derived `OpenACCSupport` analysis before running this
//    pass.
//
// 3. Device Type: The `device_type` option specifies the target device.
//    For `host` or `multicore` targets, verification of ACC compute
//    constructs is not yet implemented.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_OFFLOADTARGETVERIFIER
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "offload-target-verifier"

using namespace mlir;

namespace {

class OffloadTargetVerifier
    : public acc::impl::OffloadTargetVerifierBase<OffloadTargetVerifier> {
public:
  using OffloadTargetVerifierBase::OffloadTargetVerifierBase;

  /// Returns true if the target device type corresponds to host execution.
  bool isHostTarget() const {
    return deviceType == acc::DeviceType::Host ||
           deviceType == acc::DeviceType::Multicore;
  }

  /// Check live-in values for legality.
  SmallVector<Value>
  getIllegalLiveInValues(Region &region, Liveness &liveness,
                         acc::OpenACCSupport &accSupport) const {
    auto isInvalid = [&](Value val) -> bool {
      return !accSupport.isValidValueUse(val, region);
    };

    SmallVector<Value> illegalValues(llvm::make_filter_range(
        liveness.getLiveIn(&region.front()), isInvalid));

    return illegalValues;
  }

  /// Check symbol uses for legality.
  SmallVector<SymbolTable::SymbolUse>
  getIllegalUsedSymbols(Region &region, acc::OpenACCSupport &accSupport) const {
    auto symUses = SymbolTable::getSymbolUses(&region);

    // When there are no symbols used in the region, there are no illegal ones.
    if (!symUses.has_value())
      return {};

    auto isInvalidSymbol = [&](const SymbolTable::SymbolUse &symUse) -> bool {
      Operation *definingOp = nullptr;
      return !accSupport.isValidSymbolUse(symUse.getUser(),
                                          symUse.getSymbolRef(), &definingOp);
    };

    auto invalidSyms =
        llvm::make_filter_range(symUses.value(), isInvalidSymbol);
    SmallVector<SymbolTable::SymbolUse> invalidSymsList(invalidSyms);
    return invalidSymsList;
  }

  /// Check if the region has illegal live-in values.
  bool hasIllegalLiveInValues(Operation *regionOp,
                              acc::OpenACCSupport &accSupport) const {
    if (regionOp->getNumRegions() == 0)
      return false;

    Liveness liveness(regionOp);
    SmallVector<Value> invalidValues =
        getIllegalLiveInValues(regionOp->getRegion(0), liveness, accSupport);

    bool hasIllegalValues = !invalidValues.empty();

    if (hasIllegalValues) {
      if (softCheck) {
        // Emit warnings for each illegal value.
        auto diag = regionOp->emitWarning("offload target verifier: ")
                    << invalidValues.size() << " illegal live-in value(s)";
        for (auto [idx, invalidValue] : llvm::enumerate(invalidValues)) {
          diag.attachNote(invalidValue.getLoc()) << "value: " << invalidValue;
        }
      } else {
        accSupport.emitNYI(regionOp->getLoc(),
                           "offload target verifier failed due to " +
                               Twine(invalidValues.size()) +
                               " illegal live-in value(s)");
      }
    }

    return hasIllegalValues;
  }

  /// Check if the region has illegal symbol uses.
  bool hasIllegalSymbolUses(Operation *regionOp,
                            acc::OpenACCSupport &accSupport) const {
    if (regionOp->getNumRegions() == 0)
      return false;

    SmallVector<SymbolTable::SymbolUse> invalidSyms =
        getIllegalUsedSymbols(regionOp->getRegion(0), accSupport);

    bool hasIllegalSymbols = !invalidSyms.empty();

    if (hasIllegalSymbols) {
      auto getSymName = [&](SymbolTable::SymbolUse symUse) -> std::string {
        return symUse.getSymbolRef().getLeafReference().str();
      };
      std::string invalidString =
          llvm::join(llvm::map_range(invalidSyms, getSymName), ", ");

      // Emit only warnings when softCheck is enabled.
      if (softCheck)
        regionOp->emitWarning("offload target verifier: illegal symbol(s): ")
            << invalidString;
      else
        accSupport.emitNYI(regionOp->getLoc(),
                           "offload target verifier failed due to illegal "
                           "symbol(s): " +
                               invalidString);
    }

    return hasIllegalSymbols;
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Enter OffloadTargetVerifier()\n");
    func::FuncOp func = getOperation();

    // Try to get cached parent analysis first, fall back to local analysis.
    auto cachedAnalysis =
        getCachedParentAnalysis<acc::OpenACCSupport>(func->getParentOp());
    acc::OpenACCSupport &accSupport = cachedAnalysis
                                          ? cachedAnalysis->get()
                                          : getAnalysis<acc::OpenACCSupport>();

    bool hasErrors = false;

    func.walk([&](Operation *op) {
      // Only process offload region operations.
      if (!isa<acc::OffloadRegionOpInterface>(op))
        return WalkResult::advance();

      // TODO: Host/multicore verification for ACC compute constructs is not yet
      // implemented.
      if (isHostTarget() && isa<ACC_COMPUTE_CONSTRUCT_OPS>(op)) {
        accSupport.emitNYI(op->getLoc(),
                           "host/multicore verification for ACC compute "
                           "constructs");
        return WalkResult::advance();
      }

      // Check for illegal live-in values.
      bool hasIllegalValues = hasIllegalLiveInValues(op, accSupport);
      if (hasIllegalValues)
        hasErrors = true;

      // Check for illegal symbol uses.
      bool hasIllegalSyms = hasIllegalSymbolUses(op, accSupport);
      if (hasIllegalSyms)
        hasErrors = true;

      if (!hasIllegalValues && !hasIllegalSyms && softCheck)
        op->emitRemark("offload target verifier: passed validity check");

      return WalkResult::advance();
    });

    if (hasErrors && !softCheck)
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "Exit OffloadTargetVerifier()\n");
  }
};

} // namespace
