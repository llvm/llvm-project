//===- FileLineColLocBreakpointManagerTest.cpp - --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::tracing;

static Operation *createOp(MLIRContext *context, Location loc,
                           StringRef operationName,
                           unsigned int numRegions = 0) {
  context->allowUnregisteredDialects();
  return Operation::create(loc, OperationName(operationName, context),
                           std::nullopt, std::nullopt, std::nullopt,
                           OpaqueProperties(nullptr), std::nullopt, numRegions);
}

namespace {
struct FileLineColLocTestingAction
    : public ActionImpl<FileLineColLocTestingAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FileLineColLocTestingAction)
  static constexpr StringLiteral tag = "file-line-col-loc-testing-action";
  FileLineColLocTestingAction(ArrayRef<IRUnit> irUnits)
      : ActionImpl<FileLineColLocTestingAction>(irUnits) {}
};

TEST(FileLineColLocBreakpointManager, OperationMatch) {
  // This test will process a sequence of operation and check various situation
  // with a breakpoint hitting or not based on the location attached to the
  // operation. When a breakpoint hits, the action is skipped and the counter is
  // not incremented.
  ExecutionContext executionCtx(
      [](const ActionActiveStack *) { return ExecutionContext::Skip; });
  int counter = 0;
  auto counterInc = [&]() { counter++; };

  // Setup

  MLIRContext context;
  // Miscellaneous information to define operations
  std::vector<StringRef> fileNames = {
      StringRef("foo.bar"), StringRef("baz.qux"), StringRef("quux.corge")};
  std::vector<std::pair<unsigned, unsigned>> lineColLoc = {{42, 7}, {24, 3}};
  Location callee = UnknownLoc::get(&context),
           caller = UnknownLoc::get(&context), loc = UnknownLoc::get(&context);

  // Set of operations over where we are going to be testing the functionality
  std::vector<Operation *> operations = {
      createOp(&context, CallSiteLoc::get(callee, caller),
               "callSiteLocOperation"),
      createOp(&context,
               FileLineColLoc::get(&context, fileNames[0], lineColLoc[0].first,
                                   lineColLoc[0].second),
               "fileLineColLocOperation"),
      createOp(&context, FusedLoc::get(&context, {}, Attribute()),
               "fusedLocOperation"),
      createOp(&context, NameLoc::get(StringAttr::get(&context, fileNames[2])),
               "nameLocOperation"),
      createOp(&context, OpaqueLoc::get<void *>(nullptr, loc),
               "opaqueLocOperation"),
      createOp(&context,
               FileLineColLoc::get(&context, fileNames[1], lineColLoc[1].first,
                                   lineColLoc[1].second),
               "anotherFileLineColLocOperation"),
      createOp(&context, UnknownLoc::get(&context), "unknownLocOperation"),
  };

  FileLineColLocBreakpointManager breakpointManager;
  executionCtx.addBreakpointManager(&breakpointManager);

  // Test

  // Basic case is that no breakpoint is set and the counter is incremented for
  // every op.
  auto checkNoMatch = [&]() {
    counter = 0;
    for (auto enumeratedOp : llvm::enumerate(operations)) {
      executionCtx(counterInc,
                   FileLineColLocTestingAction({enumeratedOp.value()}));
      EXPECT_EQ(counter, static_cast<int>(enumeratedOp.index() + 1));
    }
  };
  checkNoMatch();

  // Set a breakpoint matching only the second operation in the list.
  auto *breakpoint = breakpointManager.addBreakpoint(
      fileNames[0], lineColLoc[0].first, lineColLoc[0].second);
  auto checkMatchIdxs = [&](const DenseSet<int> &idxs) {
    counter = 0;
    int reference = 0;
    for (int i = 0; i < (int)operations.size(); ++i) {
      executionCtx(counterInc, FileLineColLocTestingAction({operations[i]}));
      if (!idxs.contains(i))
        reference++;
      EXPECT_EQ(counter, reference);
    }
  };
  checkMatchIdxs({1});

  // Check that disabling the breakpoing brings us back to the original
  // behavior.
  breakpoint->disable();
  checkNoMatch();

  // Adding a breakpoint that won't match any location shouldn't affect the
  // behavior.
  breakpointManager.addBreakpoint(StringRef("random.file"), 3, 14);
  checkNoMatch();

  // Set a breakpoint matching only the fifth operation in the list.
  breakpointManager.addBreakpoint(fileNames[1], lineColLoc[1].first,
                                  lineColLoc[1].second);
  counter = 0;
  checkMatchIdxs({5});

  // Re-enable the breakpoint matching only the second operation in the list.
  // We now expect matching of operations 1 and 5.
  breakpoint->enable();
  checkMatchIdxs({1, 5});

  for (auto *op : operations) {
    op->destroy();
  }
}

TEST(FileLineColLocBreakpointManager, BlockMatch) {
  // This test will process a block and check various situation with
  // a breakpoint hitting or not based on the location attached.
  // When a breakpoint hits, the action is skipped and the counter is not
  // incremented.
  ExecutionContext executionCtx(
      [](const ActionActiveStack *) { return ExecutionContext::Skip; });
  int counter = 0;
  auto counterInc = [&]() { counter++; };

  // Setup

  MLIRContext context;
  std::vector<StringRef> fileNames = {StringRef("grault.garply"),
                                      StringRef("waldo.fred")};
  std::vector<std::pair<unsigned, unsigned>> lineColLoc = {{42, 7}, {24, 3}};
  Operation *frontOp = createOp(&context,
                                FileLineColLoc::get(&context, fileNames.front(),
                                                    lineColLoc.front().first,
                                                    lineColLoc.front().second),
                                "firstOperation");
  Operation *backOp = createOp(&context,
                               FileLineColLoc::get(&context, fileNames.back(),
                                                   lineColLoc.back().first,
                                                   lineColLoc.back().second),
                               "secondOperation");
  Block block;
  block.push_back(frontOp);
  block.push_back(backOp);

  FileLineColLocBreakpointManager breakpointManager;
  executionCtx.addBreakpointManager(&breakpointManager);

  // Test

  executionCtx(counterInc, FileLineColLocTestingAction({&block}));
  EXPECT_EQ(counter, 1);

  auto *breakpoint = breakpointManager.addBreakpoint(
      fileNames.front(), lineColLoc.front().first, lineColLoc.front().second);
  counter = 0;
  executionCtx(counterInc, FileLineColLocTestingAction({&block}));
  EXPECT_EQ(counter, 0);
  breakpoint->disable();
  executionCtx(counterInc, FileLineColLocTestingAction({&block}));
  EXPECT_EQ(counter, 1);

  breakpoint = breakpointManager.addBreakpoint(
      fileNames.back(), lineColLoc.back().first, lineColLoc.back().second);
  counter = 0;
  executionCtx(counterInc, FileLineColLocTestingAction({&block}));
  EXPECT_EQ(counter, 0);
  breakpoint->disable();
  executionCtx(counterInc, FileLineColLocTestingAction({&block}));
  EXPECT_EQ(counter, 1);
}

TEST(FileLineColLocBreakpointManager, RegionMatch) {
  // This test will process a region and check various situation with
  // a breakpoint hitting or not based on the location attached.
  // When a breakpoint hits, the action is skipped and the counter is not
  // incremented.
  ExecutionContext executionCtx(
      [](const ActionActiveStack *) { return ExecutionContext::Skip; });
  int counter = 0;
  auto counterInc = [&]() { counter++; };

  // Setup

  MLIRContext context;
  StringRef fileName("plugh.xyzzy");
  unsigned line = 42, col = 7;
  Operation *containerOp =
      createOp(&context, FileLineColLoc::get(&context, fileName, line, col),
               "containerOperation", 1);
  Region &region = containerOp->getRegion(0);

  FileLineColLocBreakpointManager breakpointManager;
  executionCtx.addBreakpointManager(&breakpointManager);

  // Test
  counter = 0;
  executionCtx(counterInc, FileLineColLocTestingAction({&region}));
  EXPECT_EQ(counter, 1);
  auto *breakpoint = breakpointManager.addBreakpoint(fileName, line, col);
  executionCtx(counterInc, FileLineColLocTestingAction({&region}));
  EXPECT_EQ(counter, 1);
  breakpoint->disable();
  executionCtx(counterInc, FileLineColLocTestingAction({&region}));
  EXPECT_EQ(counter, 2);

  containerOp->destroy();
}
} // namespace
