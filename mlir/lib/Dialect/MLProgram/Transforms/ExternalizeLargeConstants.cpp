//===- ExternalizeLargeConstants.cpp - Externalize Large Constants Pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/Transforms/Passes.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace ml_program {
#define GEN_PASS_DEF_MLPROGRAMEXTERNALIZECONSTANTSPASS
#define GEN_PASS_DEF_MLPROGRAMREPOPULATECONSTANTSPASS
#include "mlir/Dialect/MLProgram/Transforms/Passes.h.inc"

/// Repopulate constants in `op` from `externalWeightsModule`.
LogicalResult repopulateConstants(Operation *op,
                                  Operation *externalWeightsModule) {
  MLIRContext *ctx = op->getContext();
  OpBuilder builder(ctx);

  auto weightsModule = dyn_cast<ModuleOp>(externalWeightsModule);
  if (!weightsModule)
    return failure();

  SymbolTable weightsSymbolTable(weightsModule);

  auto walkResult = op->walk([&](GlobalLoadConstOp loadOp) {
    auto globalName = loadOp.getGlobal().getLeafReference();

    auto nestedModule = weightsSymbolTable.lookup<ModuleOp>(globalName);
    if (!nestedModule)
      return WalkResult::advance();

    // Find the first operation in the nested module
    if (nestedModule.getBody()->empty()) {
      loadOp.emitError() << "weight module is empty: " << globalName;
      return WalkResult::interrupt();
    }

    Operation *constOp = &nestedModule.getBody()->front();
    if (constOp->getNumResults() != 1 ||
        constOp->getResult(0).getType() != loadOp.getType()) {
      loadOp.emitError() << "weight type mismatch for: " << globalName;
      return WalkResult::interrupt();
    }

    builder.setInsertionPoint(loadOp);
    Operation *clonedOp = builder.clone(*constOp);
    loadOp.getResult().replaceAllUsesWith(clonedOp->getResult(0));
    loadOp.erase();

    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return failure();

  // Optionally clean up the extern globals if they are no longer used
  // For now we keep them as they might be needed for other things.

  return success();
}

LogicalResult externalizeConstants(ModuleOp module, int64_t minSize,
                                   bool dropLocations,
                                   StringRef outputFilename) {
  if (outputFilename.empty())
    return module.emitError() << "outputFilename is required for externalizing constants";

  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  Location moduleLoc = dropLocations ? UnknownLoc::get(ctx) : module.getLoc();
  auto weightsModule =
      ModuleOp::create(moduleLoc, StringAttr::get(ctx, "top"));
  int weightCounter = 0;
  llvm::DenseMap<Attribute, std::string> deduplicatedConstants;

  auto walkResult = module.walk([&](Operation *op) {
    if (!op->hasTrait<OpTrait::ConstantLike>() || op->getNumResults() != 1)
      return WalkResult::advance();

    ElementsAttr largeElementsAttr;
    if (!matchPattern(op, m_Constant(&largeElementsAttr)) ||
        largeElementsAttr.getNumElements() < minSize)
      return WalkResult::advance();

    Location loc = dropLocations ? UnknownLoc::get(ctx) : op->getLoc();
    std::string weightName;

    // Only capture one of the constants with the same value.
    if (dropLocations) {
      auto it = deduplicatedConstants.find(largeElementsAttr);
      if (it != deduplicatedConstants.end())
        weightName = it->second;
    }

    if (weightName.empty()) {
      // Create unique name for the weight
      // TODO: We can do something better here to make identification better.
      weightName = "_weight_" + std::to_string(weightCounter++);
      if (dropLocations)
        deduplicatedConstants[largeElementsAttr] = weightName;

      // Create extern global in the original module
      builder.setInsertionPointToStart(module.getBody());
      auto externAttr = ExternAttr::get(ctx, op->getResult(0).getType());
      GlobalOp::create(builder, loc, weightName, op->getResult(0).getType(),
                       /*is_mutable=*/false, externAttr,
                       /*sym_visibility=*/nullptr);

      // Create nested module in weightsModule
      builder.setInsertionPointToEnd(weightsModule.getBody());
      auto nestedModule = ModuleOp::create(loc, weightName);
      builder.insert(nestedModule);

      // Clone the constant op into nestedModule
      builder.setInsertionPointToEnd(nestedModule.getBody());
      Operation *clonedOp = builder.clone(*op);
      if (dropLocations)
        clonedOp->setLoc(loc);
    }

    // Replace original op with ml_program.global_load_const
    builder.setInsertionPoint(op);
    auto loadOp = GlobalLoadConstOp::create(
        builder, loc, op->getResult(0).getType(),
        FlatSymbolRefAttr::get(ctx, weightName));

    op->replaceAllUsesWith(loadOp);
    op->erase();

    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    weightsModule->erase();
    return failure();
  }

  if (weightCounter > 0) {
    std::string errorMessage;
    auto output = openOutputFile(outputFilename, &errorMessage);
    if (!output) {
      module.emitError() << "failed to open output file: " << errorMessage;
      weightsModule->erase();
      return failure();
    }
    if (failed(writeBytecodeToFile(weightsModule, output->os()))) {
      module.emitError() << "failed to write bytecode to file";
      weightsModule->erase();
      return failure();
    }
    output->keep();
  }
  weightsModule->erase();
  return success();
}

namespace {

class MLProgramExternalizeConstants
    : public impl::MLProgramExternalizeConstantsPassBase<
          MLProgramExternalizeConstants> {
public:
  using impl::MLProgramExternalizeConstantsPassBase<
      MLProgramExternalizeConstants>::MLProgramExternalizeConstantsPassBase;

  void runOnOperation() override {
    if (failed(externalizeConstants(getOperation(), minSize, dropLocations,
                                    outputFilename)))
      signalPassFailure();
  }
};

class MLProgramRepopulateConstants
    : public impl::MLProgramRepopulateConstantsPassBase<
          MLProgramRepopulateConstants> {
public:
  using impl::MLProgramRepopulateConstantsPassBase<
      MLProgramRepopulateConstants>::MLProgramRepopulateConstantsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    if (inputFilename.empty())
      return;

    ParserConfig config(ctx);
    auto weightsModuleOwned = parseSourceFile<ModuleOp>(inputFilename, config);
    if (!weightsModuleOwned) {
      emitError(module.getLoc()) << "failed to parse weights file: "
                                 << inputFilename;
      return signalPassFailure();
    }

    if (failed(repopulateConstants(module, weightsModuleOwned.get())))
      return signalPassFailure();
  }
};

} // namespace

} // namespace ml_program
} // namespace mlir
