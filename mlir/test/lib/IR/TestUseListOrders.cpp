//===- TestPrintDefUse.cpp - Passes to illustrate the IR def-use chains ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include <numeric>
#include <random>

using namespace mlir;

namespace {
/// This pass tests that:
/// 1) we can shuffle use-lists correctly;
/// 2) use-list orders are preserved after a roundtrip to bytecode.
class TestPreserveUseListOrders
    : public PassWrapper<TestPreserveUseListOrders, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPreserveUseListOrders)

  TestPreserveUseListOrders() = default;
  TestPreserveUseListOrders(const TestPreserveUseListOrders &pass)
      : PassWrapper(pass) {}
  StringRef getArgument() const final { return "test-verify-uselistorder"; }
  StringRef getDescription() const final {
    return "Verify that roundtripping the IR to bytecode preserves the order "
           "of the uselists";
  }
  Option<unsigned> rngSeed{*this, "rng-seed",
                           llvm::cl::desc("Specify an input random seed"),
                           llvm::cl::init(1)};

  LogicalResult initialize(MLIRContext *context) override {
    rng.seed(static_cast<unsigned>(rngSeed));
    return success();
  }

  void runOnOperation() override {
    // Clone the module so that we can plug in this pass to any other
    // independently.
    OwningOpRef<ModuleOp> cloneModule = getOperation().clone();

    // 1. Compute the op numbering of the module.
    computeOpNumbering(*cloneModule);

    // 2. Loop over all the values and shuffle the uses. While doing so, check
    // that each shuffle is correct.
    if (failed(shuffleUses(*cloneModule)))
      return signalPassFailure();

    // 3. Do a bytecode roundtrip to version 3, which supports use-list order
    // preservation.
    auto roundtripModuleOr = doRoundtripToBytecode(*cloneModule, 3);
    // If the bytecode roundtrip failed, try to roundtrip the original module
    // to version 2, which does not support use-list. If this also fails, the
    // original module had an issue unrelated to uselists.
    if (failed(roundtripModuleOr)) {
      auto testModuleOr = doRoundtripToBytecode(getOperation(), 2);
      if (failed(testModuleOr))
        return;

      return signalPassFailure();
    }

    // 4. Recompute the op numbering on the new module. The numbering should be
    // the same as (1), but on the new operation pointers.
    computeOpNumbering(roundtripModuleOr->get());

    // 5. Loop over all the values and verify that the use-list is consistent
    // with the post-shuffle order of step (2).
    if (failed(verifyUseListOrders(roundtripModuleOr->get())))
      return signalPassFailure();
  }

private:
  FailureOr<OwningOpRef<Operation *>> doRoundtripToBytecode(Operation *module,
                                                            uint32_t version) {
    std::string str;
    llvm::raw_string_ostream m(str);
    BytecodeWriterConfig config;
    config.setDesiredBytecodeVersion(version);
    if (failed(writeBytecodeToFile(module, m, config)))
      return failure();

    ParserConfig parseConfig(&getContext(), /*verifyAfterParse=*/true);
    auto newModuleOp = parseSourceString(StringRef(str), parseConfig);
    if (!newModuleOp.get())
      return failure();
    return newModuleOp;
  }

  /// Compute an ordered numbering for all the operations in the IR.
  void computeOpNumbering(Operation *topLevelOp) {
    uint32_t operationID = 0;
    opNumbering.clear();
    topLevelOp->walk<mlir::WalkOrder::PreOrder>(
        [&](Operation *op) { opNumbering.try_emplace(op, operationID++); });
  }

  template <typename ValueT>
  SmallVector<uint64_t> getUseIDs(ValueT val) {
    return SmallVector<uint64_t>(llvm::map_range(val.getUses(), [&](auto &use) {
      return bytecode::getUseID(use, opNumbering.at(use.getOwner()));
    }));
  }

  LogicalResult shuffleUses(Operation *topLevelOp) {
    uint32_t valueID = 0;
    /// Permute randomly the use-list of each value. It is guaranteed that at
    /// least one pair of the use list is permuted.
    auto doShuffleForRange = [&](ValueRange range) -> LogicalResult {
      for (auto val : range) {
        if (val.use_empty() || val.hasOneUse())
          continue;

        /// Get a valid index permutation for the uses of value.
        SmallVector<unsigned> permutation = getRandomPermutation(val);

        /// Store original order and verify that the shuffle was applied
        /// correctly.
        auto useIDs = getUseIDs(val);

        /// Apply shuffle to the uselist.
        val.shuffleUseList(permutation);

        /// Get the new order and verify the shuffle happened correctly.
        auto permutedIDs = getUseIDs(val);
        if (permutedIDs.size() != useIDs.size())
          return failure();
        for (size_t idx = 0; idx < permutation.size(); idx++)
          if (useIDs[idx] != permutedIDs[permutation[idx]])
            return failure();

        referenceUseListOrder.try_emplace(
            valueID++, llvm::map_range(val.getUses(), [&](auto &use) {
              return bytecode::getUseID(use, opNumbering.at(use.getOwner()));
            }));
      }
      return success();
    };

    return walkOverValues(topLevelOp, doShuffleForRange);
  }

  LogicalResult verifyUseListOrders(Operation *topLevelOp) {
    uint32_t valueID = 0;
    /// Check that the use-list for the value range matches the one stored in
    /// the reference.
    auto doValidationForRange = [&](ValueRange range) -> LogicalResult {
      for (auto val : range) {
        if (val.use_empty() || val.hasOneUse())
          continue;
        auto referenceOrder = referenceUseListOrder.at(valueID++);
        for (auto [use, referenceID] :
             llvm::zip(val.getUses(), referenceOrder)) {
          uint64_t uniqueID =
              bytecode::getUseID(use, opNumbering.at(use.getOwner()));
          if (uniqueID != referenceID) {
            use.getOwner()->emitError()
                << "found use-list order mismatch for value: " << val;
            return failure();
          }
        }
      }
      return success();
    };

    return walkOverValues(topLevelOp, doValidationForRange);
  }

  /// Walk over blocks and operations and execute a callable over the ranges of
  /// operands/results respectively.
  template <typename FuncT>
  LogicalResult walkOverValues(Operation *topLevelOp, FuncT callable) {
    auto blockWalk = topLevelOp->walk([&](Block *block) {
      if (failed(callable(block->getArguments())))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (blockWalk.wasInterrupted())
      return failure();

    auto resultsWalk = topLevelOp->walk([&](Operation *op) {
      if (failed(callable(op->getResults())))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    return failure(resultsWalk.wasInterrupted());
  }

  /// Creates a random permutation of the uselist order chain of the provided
  /// value.
  SmallVector<unsigned> getRandomPermutation(Value value) {
    size_t numUses = std::distance(value.use_begin(), value.use_end());
    SmallVector<unsigned> permutation(numUses);
    unsigned zero = 0;
    std::iota(permutation.begin(), permutation.end(), zero);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    return permutation;
  }

  /// Map each value to its use-list order encoded with unique use IDs.
  DenseMap<uint32_t, SmallVector<uint64_t>> referenceUseListOrder;

  /// Map each operation to its global ID.
  DenseMap<Operation *, uint32_t> opNumbering;

  std::default_random_engine rng;
};
} // namespace

namespace mlir {
void registerTestPreserveUseListOrders() {
  PassRegistration<TestPreserveUseListOrders>();
}
} // namespace mlir
