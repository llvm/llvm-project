//===- OpenACCUtilsCGTest.cpp - Unit tests for OpenACC CG utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCParMapping.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCUtilsCGTest : public ::testing::Test {
protected:
  OpenACCUtilsCGTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, arith::ArithDialect,
                        func::FuncDialect, scf::SCFDialect, gpu::GPUDialect,
                        memref::MemRefDialect, DLTIDialect>();
  }

  static ComputeRegionOp buildComputeRegionWithPrivateLocal(
      MLIRContext &context, OpBuilder &b, Location loc, ModuleOp module,
      GPUParallelDimsAttr privatizeParDims, ValueRange launchArgs,
      PrivateLocalOp &privateLocalOut, PrivatizeOp &privatizeOut,
      MemRefType memTy = {}, bool addReductionAccumulator = false) {
    IRRewriter rewriter(&context);
    rewriter.setInsertionPointToStart(module.getBody());

    if (!memTy)
      memTy = MemRefType::get({4}, b.getI32Type());
    Type privateTy = PrivateType::get(&context, memTy);
    privatizeOut = PrivatizeOp::create(rewriter, loc, privateTy, ValueRange{},
                                       privatizeParDims);

    Region sourceRegion;
    Block *srcBlock = new Block();
    sourceRegion.push_back(srcBlock);
    BlockArgument privArg = srcBlock->addArgument(privateTy, loc);
    OpBuilder srcBuilder(&context);
    srcBuilder.setInsertionPointToStart(srcBlock);

    Value c0 = arith::ConstantIndexOp::create(srcBuilder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(srcBuilder, loc, 1);
    YieldOp::create(srcBuilder, loc);

    // Nest one scf.parallel per privatization dimension so that
    // collectPrivateLocalParDims recovers them from the parent loops.
    srcBuilder.setInsertionPoint(srcBlock->getTerminator());
    for (GPUParallelDimAttr parDim : privatizeParDims.getArray()) {
      auto par = scf::ParallelOp::create(srcBuilder, loc, ValueRange{c0},
                                         ValueRange{c1}, ValueRange{c1});
      setParDimsAttr(par, GPUParallelDimsAttr::get(&context, {parDim}));
      srcBuilder.setInsertionPoint(par.getBody()->getTerminator());
    }
    PrivateLocalOp privateLocal =
        PrivateLocalOp::create(srcBuilder, loc, memTy, privArg);
    if (addReductionAccumulator) {
      Value partial =
          arith::ConstantIntOp::create(srcBuilder, loc, b.getI32Type(), 1);
      SmallVector<GPUParallelDimAttr> reductionDims;
      for (GPUParallelDimAttr parDim : privatizeParDims.getArray())
        if (!parDim.isAnyBlock())
          reductionDims.push_back(parDim);
      ReductionAccumulateOp::create(
          srcBuilder, loc, partial, privateLocal.getResult(),
          ReductionOperator::AccAdd,
          GPUParallelDimsAttr::get(&context, reductionDims));
    }

    IRMapping mapping;
    auto cr = buildComputeRegion(
        loc, launchArgs, ValueRange{privatizeOut.getResult()},
        ParallelOp::getOperationName(), sourceRegion, rewriter, mapping,
        /*output=*/{}, /*kernelFuncName=*/{}, /*kernelModuleName=*/{},
        /*stream=*/{}, ValueRange{privArg});

    privateLocalOut = {};
    cr.walk([&](PrivateLocalOp op) { privateLocalOut = op; });
    return cr;
  }

  static ComputeRegionOp createEmptyComputeRegion(MLIRContext &context,
                                                  Location loc,
                                                  ModuleOp module) {
    IRRewriter rewriter(&context);
    rewriter.setInsertionPointToStart(module.getBody());

    Region sourceRegion;
    Block *block = new Block();
    sourceRegion.push_back(block);
    OpBuilder regionBuilder(&context);
    regionBuilder.setInsertionPointToStart(block);
    YieldOp::create(regionBuilder, loc);

    IRMapping mapping;
    return buildComputeRegion(loc, ValueRange{}, ValueRange{},
                              ParallelOp::getOperationName(), sourceRegion,
                              rewriter, mapping);
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

//===----------------------------------------------------------------------===//
// getDataLayout Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, getDataLayoutNoSpecAllowDefault) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // With allowDefault=true, should return a default DataLayout
  auto dl = getDataLayout(module->getOperation(), /*allowDefault=*/true);
  EXPECT_TRUE(dl.has_value());
}

TEST_F(OpenACCUtilsCGTest, getDataLayoutNoSpecDisallowDefault) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // With allowDefault=false and no spec, should return nullopt
  auto dl = getDataLayout(module->getOperation(), /*allowDefault=*/false);
  EXPECT_FALSE(dl.has_value());
}

TEST_F(OpenACCUtilsCGTest, getDataLayoutNullOp) {
  // Null operation should return nullopt
  auto dl = getDataLayout(nullptr, /*allowDefault=*/true);
  EXPECT_FALSE(dl.has_value());
}

TEST_F(OpenACCUtilsCGTest, getDataLayoutWithSpec) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // Add a data layout spec to the module
  auto indexEntry = DataLayoutEntryAttr::get(IndexType::get(&context),
                                             b.getI32IntegerAttr(32));
  auto spec = DataLayoutSpecAttr::get(&context, {indexEntry});
  (*module)->setAttr(DLTIDialect::kDataLayoutAttrName, spec);

  // With explicit spec, should return DataLayout regardless of allowDefault
  auto dl1 = getDataLayout(module->getOperation(), /*allowDefault=*/false);
  EXPECT_TRUE(dl1.has_value());

  auto dl2 = getDataLayout(module->getOperation(), /*allowDefault=*/true);
  EXPECT_TRUE(dl2.has_value());
}

//===----------------------------------------------------------------------===//
// ParDim utilities Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, insertParDimOrdersAndDeduplicates) {
  SmallVector<GPUParallelDimAttr> parDims;
  GPUParallelDimAttr threadX = GPUParallelDimAttr::threadXDim(&context);
  GPUParallelDimAttr blockX = GPUParallelDimAttr::blockXDim(&context);
  GPUParallelDimAttr threadY = GPUParallelDimAttr::threadYDim(&context);

  insertParDim(parDims, threadX);
  insertParDim(parDims, blockX);
  insertParDim(parDims, threadY);
  insertParDim(parDims, threadX);

  ASSERT_EQ(parDims.size(), 3u);
  EXPECT_EQ(parDims[0], blockX);
  EXPECT_EQ(parDims[1], threadY);
  EXPECT_EQ(parDims[2], threadX);
}

TEST_F(OpenACCUtilsCGTest, removeParDimRemovesOnlyMatchingDim) {
  GPUParallelDimAttr threadX = GPUParallelDimAttr::threadXDim(&context);
  GPUParallelDimAttr blockX = GPUParallelDimAttr::blockXDim(&context);
  GPUParallelDimAttr threadY = GPUParallelDimAttr::threadYDim(&context);
  SmallVector<GPUParallelDimAttr> parDims{blockX, threadY, threadX};

  removeParDim(parDims, threadX);
  removeParDim(parDims, GPUParallelDimAttr::blockYDim(&context));

  ASSERT_EQ(parDims.size(), 2u);
  EXPECT_EQ(parDims[0], blockX);
  EXPECT_EQ(parDims[1], threadY);
}

TEST_F(OpenACCUtilsCGTest, parDimsOperationAttributes) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  OwningOpRef<ModuleOp> otherModule = ModuleOp::create(b, loc);
  Operation *op = module->getOperation();
  Operation *otherOp = otherModule->getOperation();
  GPUParallelDimsAttr seqAttr = GPUParallelDimsAttr::seq(&context);
  GPUParallelDimsAttr blockAttr = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context)});

  EXPECT_FALSE(hasParDimsAttr(op));
  setParDimsAttr(op, seqAttr);
  EXPECT_TRUE(hasParDimsAttr(op));
  EXPECT_TRUE(hasSeqParDims(op));
  EXPECT_EQ(getParDimsAttr(op), seqAttr);

  updateParDimsAttr(op, blockAttr);
  EXPECT_FALSE(hasSeqParDims(op));
  EXPECT_EQ(getParDimsAttr(op), blockAttr);

  copyParDimsAttr(op, otherOp);
  EXPECT_EQ(getParDimsAttr(otherOp), blockAttr);
}

TEST_F(OpenACCUtilsCGTest, getParDimsAttrReadsInherentAttribute) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  GPUParallelDimsAttr blockAttr = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context)});
  Type privateTy = PrivateType::get(&context, b.getI32Type());
  auto privatize =
      PrivatizeOp::create(b, loc, privateTy,
                          /*dynamicSizes=*/ValueRange{}, blockAttr);

  EXPECT_TRUE(hasParDimsAttr(privatize));
  EXPECT_EQ(getParDimsAttr(privatize), blockAttr);
  EXPECT_EQ(privatize.getParDimsAttr(), blockAttr);
}

TEST_F(OpenACCUtilsCGTest, setParDimsAttrSetsInherentAttribute) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  Type privateTy = PrivateType::get(&context, b.getI32Type());
  auto privatize =
      PrivatizeOp::create(b, loc, privateTy, /*dynamicSizes=*/ValueRange{});
  GPUParallelDimsAttr blockAttr = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context)});

  setParDimsAttr(privatize, blockAttr);
  EXPECT_EQ(getParDimsAttr(privatize), blockAttr);
  EXPECT_EQ(privatize.getParDimsAttr(), blockAttr);
}

//===----------------------------------------------------------------------===//
// buildComputeRegion Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, buildComputeRegionEmpty) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  IRRewriter rewriter(&context);
  rewriter.setInsertionPointToEnd(module->getBody());

  auto funcTy = b.getFunctionType({}, {});
  auto func = func::FuncOp::create(rewriter, loc, "test", funcTy);
  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);

  Region sourceRegion;
  Block *srcBlock = new Block();
  sourceRegion.push_back(srcBlock);
  OpBuilder srcBuilder(&context);
  srcBuilder.setInsertionPointToStart(srcBlock);
  YieldOp::create(srcBuilder, loc);

  IRMapping mapping;
  auto cr = buildComputeRegion(loc, /*launchArgs=*/{}, /*inputArgs=*/{},
                               SerialOp::getOperationName(), sourceRegion,
                               rewriter, mapping);

  EXPECT_EQ(cr.getOrigin(), SerialOp::getOperationName());
  EXPECT_EQ(cr.getLaunchArgs().size(), 0u);
  EXPECT_EQ(cr.getInputArgs().size(), 0u);
  EXPECT_TRUE(cr.getRegion().hasOneBlock());

  func::ReturnOp::create(rewriter, loc);
}

TEST_F(OpenACCUtilsCGTest, buildComputeRegionWithLaunchArgs) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  IRRewriter rewriter(&context);
  rewriter.setInsertionPointToEnd(module->getBody());

  auto funcTy = b.getFunctionType({}, {});
  auto func = func::FuncOp::create(rewriter, loc, "test", funcTy);
  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);

  auto c128 = arith::ConstantIndexOp::create(rewriter, loc, 128);
  auto threadXDim = GPUParallelDimAttr::threadXDim(&context);
  auto pw = ParWidthOp::create(rewriter, loc, c128, threadXDim);

  Region sourceRegion;
  Block *srcBlock = new Block();
  sourceRegion.push_back(srcBlock);
  OpBuilder srcBuilder(&context);
  srcBuilder.setInsertionPointToStart(srcBlock);
  YieldOp::create(srcBuilder, loc);

  IRMapping mapping;
  auto cr = buildComputeRegion(loc, {pw}, /*inputArgs=*/{},
                               ParallelOp::getOperationName(), sourceRegion,
                               rewriter, mapping);

  EXPECT_EQ(cr.getOrigin(), ParallelOp::getOperationName());
  EXPECT_EQ(cr.getLaunchArgs().size(), 1u);
  EXPECT_EQ(cr.getLaunchArgs()[0], pw.getResult());
  EXPECT_TRUE(llvm::isa<IndexType>(pw.getResult().getType()));
  ASSERT_FALSE(cr.getRegion().empty());
  EXPECT_TRUE(
      llvm::isa<IndexType>(cr.getRegion().front().getArgument(0).getType()));

  func::ReturnOp::create(rewriter, loc);
}

// Test buildComputeRegion with inputArgsToMap: clone a region whose block args
// are the "source" values, while the op's inputArgs are "device" values. The
// mapping should map source -> compute_region block args so the cloned body
// uses the correct values.
TEST_F(OpenACCUtilsCGTest, buildComputeRegionWithInputArgsToMap) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  IRRewriter rewriter(&context);
  rewriter.setInsertionPointToEnd(module->getBody());

  // Source function: one block with one index arg, body uses it (addi), then
  // return (terminator is not cloned).
  auto funcTy = b.getFunctionType({b.getIndexType()}, {});
  auto sourceFunc = func::FuncOp::create(rewriter, loc, "source", funcTy);
  Block *sourceBlock = sourceFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(sourceBlock);
  auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto addOp = arith::AddIOp::create(rewriter, loc, sourceBlock->getArgument(0),
                                     c1.getResult());
  (void)addOp;
  func::ReturnOp::create(rewriter, loc);

  // Set insertion back to module so hostFunc is also added to the module.
  rewriter.setInsertionPointToEnd(module->getBody());

  // Current function: we have a "device" block with one index arg. We will
  // clone sourceFunc's body into a compute_region, with inputArgs = [device
  // arg] and inputArgsToMap = [source block arg], so the clone maps source arg
  // -> compute region block arg.
  auto hostFuncTy = b.getFunctionType({b.getIndexType()}, {});
  auto hostFunc = func::FuncOp::create(rewriter, loc, "host", hostFuncTy);
  Block *deviceBlock = hostFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(deviceBlock);

  Region &sourceRegion = sourceFunc.getBody();
  ValueRange sourceArgsToMap = sourceRegion.front().getArguments();
  ValueRange inputArgs = deviceBlock->getArguments();

  IRMapping mapping;
  auto cr = buildComputeRegion(
      loc, /*launchArgs=*/{}, inputArgs, SerialOp::getOperationName(),
      sourceRegion, rewriter, mapping,
      /*output=*/{}, /*kernelFuncName=*/{}, /*kernelModuleName=*/{},
      /*stream=*/{}, sourceArgsToMap);

  ASSERT_TRUE(cr);
  EXPECT_EQ(cr.getInputArgs().size(), 1u);
  EXPECT_EQ(cr.getInputArgs()[0], deviceBlock->getArgument(0));
  Block &crBlock = cr.getRegion().front();
  EXPECT_EQ(crBlock.getNumArguments(), 1u);
  // The cloned body should use the compute_region's block arg (mapped from
  // source arg). So the only non-constant operand of the addi in the clone
  // should be crBlock.getArgument(0).
  bool foundAddI = false;
  for (Operation &op : crBlock.getOperations()) {
    if (isa<arith::AddIOp>(op)) {
      foundAddI = true;
      EXPECT_EQ(op.getOperand(0), crBlock.getArgument(0));
      break;
    }
  }
  EXPECT_TRUE(foundAddI);

  EXPECT_EQ(cr.getOperand(crBlock.getArgument(0)), deviceBlock->getArgument(0));
  ASSERT_TRUE(cr.getBlockArg(deviceBlock->getArgument(0)).has_value());
  EXPECT_EQ(*cr.getBlockArg(deviceBlock->getArgument(0)),
            crBlock.getArgument(0));

  func::ReturnOp::create(rewriter, loc);
}

//===----------------------------------------------------------------------===//
// SharedMemoryBudget Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, SharedMemoryBudgetAlignAndAllocate) {
  EXPECT_EQ(SharedMemoryBudget::alignOffset(0), 0);
  EXPECT_EQ(SharedMemoryBudget::alignOffset(1), 16);
  EXPECT_EQ(SharedMemoryBudget::alignOffset(16), 16);

  SharedMemoryBudget budget(100);
  EXPECT_TRUE(budget.tryAllocate(50));
  EXPECT_EQ(budget.bytesUsed(), 50);
  EXPECT_TRUE(budget.tryAllocate(34));
  EXPECT_EQ(budget.bytesUsed(), 98);
  EXPECT_FALSE(budget.tryAllocate(10));
}

TEST_F(OpenACCUtilsCGTest, SharedMemoryBudgetInitialBytesUsed) {
  SharedMemoryBudget budget(64, /*initialBytesUsed=*/48);
  EXPECT_FALSE(budget.tryAllocate(32));
  EXPECT_TRUE(budget.tryAllocate(16));
  EXPECT_EQ(budget.bytesUsed(), 64);
}

//===----------------------------------------------------------------------===//
// sumExistingSharedMemoryBytes Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, sumExistingSharedMemoryBytes) {
  Region region;
  Block *block = new Block();
  region.push_back(block);
  b.setInsertionPointToStart(block);

  MemRefType ty = MemRefType::get({4}, b.getI32Type());
  GPUSharedMemoryOp::create(b, loc, ty, b.getI64IntegerAttr(1),
                            b.getI64IntegerAttr(100), ValueRange{},
                            IntegerAttr{}, IntegerAttr{});
  GPUSharedMemoryOp::create(b, loc, ty, b.getI64IntegerAttr(1),
                            b.getI64IntegerAttr(50), ValueRange{},
                            IntegerAttr{}, IntegerAttr{});

  EXPECT_EQ(sumExistingSharedMemoryBytes(region), 162);
}

//===----------------------------------------------------------------------===//
// getPrivateBaseMemRefType Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, getPrivateBaseMemRefType) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  MemRefType memTy = MemRefType::get({10}, b.getI64Type());

  EXPECT_EQ(getPrivateBaseMemRefType(memTy, *module), memTy);
}

//===----------------------------------------------------------------------===//
// getPrivatizeOp Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, getPrivatizeOpFromHandle) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  MemRefType memTy = MemRefType::get({4}, b.getI32Type());
  Type privateTy = PrivateType::get(&context, memTy);
  auto privatize =
      PrivatizeOp::create(b, loc, privateTy, /*dynamicSizes=*/ValueRange{});
  auto privateLocal = PrivateLocalOp::create(b, loc, memTy, privatize);
  auto computeRegion = createEmptyComputeRegion(context, loc, *module);

  EXPECT_EQ(getPrivatizeOp(privateLocal, computeRegion), privatize);
}

TEST_F(OpenACCUtilsCGTest, getPrivatizeOpFromComputeRegionBlockArg) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());
  GPUParallelDimsAttr gangDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context)});
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto bx =
      ParWidthOp::create(b, loc, c1, GPUParallelDimAttr::blockXDim(&context));

  PrivateLocalOp privateLocal;
  PrivatizeOp privatize;
  auto cr = buildComputeRegionWithPrivateLocal(
      context, b, loc, *module, gangDims, ValueRange{bx.getResult()},
      privateLocal, privatize);

  EXPECT_EQ(getPrivatizeOp(privateLocal, cr), privatize);
}

//===----------------------------------------------------------------------===//
// collectPrivateLocalParDims Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, collectPrivateLocalParDimsFromParentLoops) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  IRRewriter rewriter(&context);
  rewriter.setInsertionPointToStart(module->getBody());

  auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto bx = ParWidthOp::create(rewriter, loc, c1,
                               GPUParallelDimAttr::blockXDim(&context));
  auto tx = ParWidthOp::create(rewriter, loc, c1,
                               GPUParallelDimAttr::threadXDim(&context));

  MemRefType memTy = MemRefType::get({4}, b.getI32Type());
  Type privateTy = PrivateType::get(&context, memTy);
  auto privatize = PrivatizeOp::create(rewriter, loc, privateTy, ValueRange{});

  Region sourceRegion;
  Block *srcBlock = new Block();
  sourceRegion.push_back(srcBlock);
  BlockArgument privArg = srcBlock->addArgument(privateTy, loc);
  OpBuilder srcBuilder(&context);
  srcBuilder.setInsertionPointToStart(srcBlock);
  auto c0 = arith::ConstantIndexOp::create(srcBuilder, loc, 0);
  auto c1Body = arith::ConstantIndexOp::create(srcBuilder, loc, 1);
  YieldOp::create(srcBuilder, loc);
  srcBuilder.setInsertionPoint(srcBlock->getTerminator());

  // Outer gang (block_x) loop containing an inner vector (thread_x) loop.
  auto gangLoop = scf::ParallelOp::create(
      srcBuilder, loc, ValueRange{c0}, ValueRange{c1Body}, ValueRange{c1Body});
  setParDimsAttr(gangLoop,
                 GPUParallelDimsAttr::get(
                     &context, {GPUParallelDimAttr::blockXDim(&context)}));
  srcBuilder.setInsertionPoint(gangLoop.getBody()->getTerminator());
  auto vectorLoop = scf::ParallelOp::create(
      srcBuilder, loc, ValueRange{c0}, ValueRange{c1Body}, ValueRange{c1Body});
  setParDimsAttr(vectorLoop,
                 GPUParallelDimsAttr::get(
                     &context, {GPUParallelDimAttr::threadXDim(&context)}));
  srcBuilder.setInsertionPoint(vectorLoop.getBody()->getTerminator());
  PrivateLocalOp::create(srcBuilder, loc, memTy, privArg);

  IRMapping mapping;
  auto cr = buildComputeRegion(
      loc, ValueRange{bx.getResult(), tx.getResult()},
      ValueRange{privatize.getResult()}, ParallelOp::getOperationName(),
      sourceRegion, rewriter, mapping, /*output=*/{}, /*kernelFuncName=*/{},
      /*kernelModuleName=*/{}, /*stream=*/{}, ValueRange{privArg});

  PrivateLocalOp clonedLocal;
  cr.walk([&](PrivateLocalOp op) { clonedLocal = op; });

  DefaultACCToGPUMappingPolicy policy;
  SmallVector<GPUParallelDimAttr> parDims =
      collectPrivateLocalParDims(clonedLocal, cr);
  ASSERT_EQ(parDims.size(), 2u);
  EXPECT_TRUE(policy.isGang(parDims[0]));
  EXPECT_TRUE(policy.isVector(parDims[1]));
}

TEST_F(OpenACCUtilsCGTest, collectPrivateLocalParDimsFromLaunchFallback) {
  // With no enclosing parallel loops, collectPrivateLocalParDims falls back to
  // the block-level launch dimensions.
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());
  GPUParallelDimsAttr gangDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context)});
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto bx =
      ParWidthOp::create(b, loc, c1, GPUParallelDimAttr::blockXDim(&context));

  MemRefType memTy = MemRefType::get({4}, b.getI32Type());
  Type privateTy = PrivateType::get(&context, memTy);
  auto privatize = PrivatizeOp::create(b, loc, privateTy, ValueRange{});

  Region sourceRegion;
  Block *srcBlock = new Block();
  sourceRegion.push_back(srcBlock);
  BlockArgument privArg = srcBlock->addArgument(privateTy, loc);
  OpBuilder srcBuilder(&context);
  srcBuilder.setInsertionPointToStart(srcBlock);
  PrivateLocalOp::create(srcBuilder, loc, memTy, privArg);
  YieldOp::create(srcBuilder, loc);

  IRRewriter rewriter(&context);
  rewriter.setInsertionPointToStart(module->getBody());
  IRMapping mapping;
  auto cr = buildComputeRegion(
      loc, ValueRange{bx.getResult()}, ValueRange{privatize.getResult()},
      ParallelOp::getOperationName(), sourceRegion, rewriter, mapping,
      /*output=*/{}, /*kernelFuncName=*/{}, /*kernelModuleName=*/{},
      /*stream=*/{}, ValueRange{privArg});
  (void)gangDims;

  PrivateLocalOp clonedLocal;
  cr.walk([&](PrivateLocalOp op) { clonedLocal = op; });

  DefaultACCToGPUMappingPolicy policy;
  SmallVector<GPUParallelDimAttr> parDims =
      collectPrivateLocalParDims(clonedLocal, cr);
  ASSERT_EQ(parDims.size(), 1u);
  EXPECT_TRUE(policy.isGang(parDims[0]));
}

TEST_F(OpenACCUtilsCGTest,
       collectPrivateLocalParDimsStopsAtComputeRegionBoundary) {
  // Parallel loops enclosing the compute region must not contribute par-dims.
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  IRRewriter rewriter(&context);
  rewriter.setInsertionPointToStart(module->getBody());

  auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto bx = ParWidthOp::create(rewriter, loc, c1,
                               GPUParallelDimAttr::blockXDim(&context));
  auto tx = ParWidthOp::create(rewriter, loc, c1,
                               GPUParallelDimAttr::threadXDim(&context));

  // Outer loop outside the compute region carries block_y.
  // This is a synthetic test only and normally this scenario would hopefully
  // not be encountered.
  auto outerLoop = scf::ParallelOp::create(rewriter, loc, ValueRange{c0},
                                           ValueRange{c1}, ValueRange{c1});
  setParDimsAttr(outerLoop,
                 GPUParallelDimsAttr::get(
                     &context, {GPUParallelDimAttr::blockYDim(&context)}));
  rewriter.setInsertionPoint(outerLoop.getBody()->getTerminator());

  MemRefType memTy = MemRefType::get({4}, b.getI32Type());
  Type privateTy = PrivateType::get(&context, memTy);
  auto privatize = PrivatizeOp::create(rewriter, loc, privateTy, ValueRange{});

  Region sourceRegion;
  Block *srcBlock = new Block();
  sourceRegion.push_back(srcBlock);
  BlockArgument privArg = srcBlock->addArgument(privateTy, loc);
  OpBuilder srcBuilder(&context);
  srcBuilder.setInsertionPointToStart(srcBlock);
  auto c0Body = arith::ConstantIndexOp::create(srcBuilder, loc, 0);
  auto c1Body = arith::ConstantIndexOp::create(srcBuilder, loc, 1);
  YieldOp::create(srcBuilder, loc);
  srcBuilder.setInsertionPoint(srcBlock->getTerminator());

  // Inner vector (thread_x) loop inside the compute region.
  auto vectorLoop =
      scf::ParallelOp::create(srcBuilder, loc, ValueRange{c0Body},
                              ValueRange{c1Body}, ValueRange{c1Body});
  setParDimsAttr(vectorLoop,
                 GPUParallelDimsAttr::get(
                     &context, {GPUParallelDimAttr::threadXDim(&context)}));
  srcBuilder.setInsertionPoint(vectorLoop.getBody()->getTerminator());
  PrivateLocalOp::create(srcBuilder, loc, memTy, privArg);

  IRMapping mapping;
  auto cr = buildComputeRegion(
      loc, ValueRange{bx.getResult(), tx.getResult()},
      ValueRange{privatize.getResult()}, ParallelOp::getOperationName(),
      sourceRegion, rewriter, mapping, /*output=*/{}, /*kernelFuncName=*/{},
      /*kernelModuleName=*/{}, /*stream=*/{}, ValueRange{privArg});

  PrivateLocalOp clonedLocal;
  cr.walk([&](PrivateLocalOp op) { clonedLocal = op; });

  DefaultACCToGPUMappingPolicy policy;
  SmallVector<GPUParallelDimAttr> parDims =
      collectPrivateLocalParDims(clonedLocal, cr);
  ASSERT_EQ(parDims.size(), 1u);
  EXPECT_TRUE(policy.isVector(parDims[0]));
  EXPECT_FALSE(policy.isGang(parDims[0]));
}

TEST_F(OpenACCUtilsCGTest, collectPrivateLocalParDimsFromReductionUsers) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  MemRefType memTy = MemRefType::get({}, b.getI32Type());
  Type privateTy = PrivateType::get(&context, memTy);
  auto privatize =
      PrivatizeOp::create(b, loc, privateTy, /*dynamicSizes=*/ValueRange{});
  auto privateLocal = PrivateLocalOp::create(b, loc, memTy, privatize);
  auto computeRegion = createEmptyComputeRegion(context, loc, *module);

  GPUParallelDimsAttr accDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context),
                 GPUParallelDimAttr::threadXDim(&context)});
  Value partial = arith::ConstantIntOp::create(b, loc, b.getI32Type(), 1);
  ReductionAccumulateOp::create(b, loc, partial, privateLocal.getResult(),
                                ReductionOperator::AccAdd, accDims);

  DefaultACCToGPUMappingPolicy policy;
  SmallVector<GPUParallelDimAttr> parDims =
      collectPrivateLocalParDims(privateLocal, computeRegion);
  ASSERT_EQ(parDims.size(), 2u);
  EXPECT_TRUE(policy.isGang(parDims[0]));
  EXPECT_TRUE(policy.isVector(parDims[1]));
}

//===----------------------------------------------------------------------===//
// Shared-memory private_local eligibility Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, isPrivateLocalSharedMemoryCandidateGangPrivate) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());
  GPUParallelDimsAttr gangDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context)});
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto bx =
      ParWidthOp::create(b, loc, c1, GPUParallelDimAttr::blockXDim(&context));

  PrivateLocalOp privateLocal;
  PrivatizeOp privatize;
  auto cr = buildComputeRegionWithPrivateLocal(
      context, b, loc, *module, gangDims, ValueRange{bx.getResult()},
      privateLocal, privatize);

  DefaultACCToGPUMappingPolicy policy;
  FailureOr<bool> isCandidate =
      isPrivateLocalSharedMemoryCandidate(privateLocal, cr, *module, policy);
  ASSERT_TRUE(succeeded(isCandidate));
  EXPECT_TRUE(*isCandidate);
}

TEST_F(OpenACCUtilsCGTest, isPrivateLocalSharedMemoryCandidateThreadXPrivate) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());
  GPUParallelDimsAttr vectorDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::threadXDim(&context)});
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto tx =
      ParWidthOp::create(b, loc, c1, GPUParallelDimAttr::threadXDim(&context));

  PrivateLocalOp privateLocal;
  PrivatizeOp privatize;
  auto cr = buildComputeRegionWithPrivateLocal(
      context, b, loc, *module, vectorDims, ValueRange{tx.getResult()},
      privateLocal, privatize);

  DefaultACCToGPUMappingPolicy policy;
  FailureOr<bool> isCandidate =
      isPrivateLocalSharedMemoryCandidate(privateLocal, cr, *module, policy);
  ASSERT_TRUE(succeeded(isCandidate));
  EXPECT_FALSE(*isCandidate);
}

TEST_F(OpenACCUtilsCGTest,
       isPrivateLocalSharedMemoryCandidateWorkerPrivateConstant) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());
  GPUParallelDimsAttr workerDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::threadYDim(&context)});
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto c4 = arith::ConstantIndexOp::create(b, loc, 4);
  auto bx =
      ParWidthOp::create(b, loc, c1, GPUParallelDimAttr::blockXDim(&context));
  auto ty =
      ParWidthOp::create(b, loc, c4, GPUParallelDimAttr::threadYDim(&context));

  PrivateLocalOp privateLocal;
  PrivatizeOp privatize;
  auto cr = buildComputeRegionWithPrivateLocal(
      context, b, loc, *module, workerDims,
      ValueRange{bx.getResult(), ty.getResult()}, privateLocal, privatize);

  DefaultACCToGPUMappingPolicy policy;
  FailureOr<bool> isCandidate =
      isPrivateLocalSharedMemoryCandidate(privateLocal, cr, *module, policy);
  ASSERT_TRUE(succeeded(isCandidate));
  EXPECT_TRUE(*isCandidate);
}

TEST_F(OpenACCUtilsCGTest, getSharedMemoryBytesGangWorkerReductionAccumulator) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());
  GPUParallelDimsAttr gangWorkerDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context),
                 GPUParallelDimAttr::threadYDim(&context)});
  auto c2 = arith::ConstantIndexOp::create(b, loc, 2);
  auto c4 = arith::ConstantIndexOp::create(b, loc, 4);
  auto bx =
      ParWidthOp::create(b, loc, c2, GPUParallelDimAttr::blockXDim(&context));
  auto ty =
      ParWidthOp::create(b, loc, c4, GPUParallelDimAttr::threadYDim(&context));

  PrivateLocalOp privateLocal;
  PrivatizeOp privatize;
  MemRefType scalarTy = MemRefType::get({}, b.getI32Type());
  auto cr = buildComputeRegionWithPrivateLocal(
      context, b, loc, *module, gangWorkerDims,
      ValueRange{bx.getResult(), ty.getResult()}, privateLocal, privatize,
      scalarTy, /*addReductionAccumulator=*/true);

  DefaultACCToGPUMappingPolicy policy;
  std::optional<int64_t> upperBound =
      getPrivateLocalSharedMemoryUpperBoundBytes(privateLocal, cr, *module,
                                                 policy);
  ASSERT_TRUE(upperBound.has_value());
  EXPECT_EQ(*upperBound, 16);
}

TEST_F(OpenACCUtilsCGTest,
       isPrivateLocalSharedMemoryCandidateWorkerPrivateDynamicFails) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());
  GPUParallelDimsAttr workerDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::threadYDim(&context)});
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto bx =
      ParWidthOp::create(b, loc, c1, GPUParallelDimAttr::blockXDim(&context));
  // A non-constant num_workers: the launch operand exists but is not an
  // arith.constant, which is what triggers the diagnostic / failure path.
  auto dynNumWorkers = arith::AddIOp::create(b, loc, c1, c1);
  auto ty = ParWidthOp::create(b, loc, dynNumWorkers,
                               GPUParallelDimAttr::threadYDim(&context));

  PrivateLocalOp privateLocal;
  PrivatizeOp privatize;
  auto cr = buildComputeRegionWithPrivateLocal(
      context, b, loc, *module, workerDims,
      ValueRange{bx.getResult(), ty.getResult()}, privateLocal, privatize);

  DefaultACCToGPUMappingPolicy policy;
  OpenACCSupport support;
  FailureOr<bool> silent =
      isPrivateLocalSharedMemoryCandidate(privateLocal, cr, *module, policy);
  ASSERT_TRUE(succeeded(silent));
  EXPECT_FALSE(*silent);

  FailureOr<bool> diagnosed = isPrivateLocalSharedMemoryCandidate(
      privateLocal, cr, *module, policy, &support);
  EXPECT_TRUE(failed(diagnosed));
}

TEST_F(OpenACCUtilsCGTest, getPrivateLocalSharedMemoryUpperBoundBytes) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());
  GPUParallelDimsAttr gangDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context)});
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto bx =
      ParWidthOp::create(b, loc, c1, GPUParallelDimAttr::blockXDim(&context));

  PrivateLocalOp privateLocal;
  PrivatizeOp privatize;
  auto cr = buildComputeRegionWithPrivateLocal(
      context, b, loc, *module, gangDims, ValueRange{bx.getResult()},
      privateLocal, privatize);

  DefaultACCToGPUMappingPolicy policy;
  std::optional<int64_t> upperBound =
      getPrivateLocalSharedMemoryUpperBoundBytes(privateLocal, cr, *module,
                                                 policy);
  ASSERT_TRUE(upperBound.has_value());
  EXPECT_EQ(*upperBound, 16);
}
