//===- Diagnostic.cpp - Dialect unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::detail;

namespace {

TEST(DiagnosticLifetime, TestCopiesConstCharStar) {
  const auto *expectedMessage = "Error 1, don't mutate this";

  // Copy expected message into a mutable container, and call the constructor.
  std::string myStr(expectedMessage);

  mlir::MLIRContext context;
  Diagnostic diagnostic(mlir::UnknownLoc::get(&context),
                        DiagnosticSeverity::Note);
  diagnostic << myStr.c_str();

  // Mutate underlying pointer, but ensure diagnostic still has orig. message
  myStr[0] = '^';

  std::string resultMessage;
  llvm::raw_string_ostream stringStream(resultMessage);
  diagnostic.print(stringStream);
  ASSERT_STREQ(expectedMessage, resultMessage.c_str());
}

TEST(DiagnosticLifetime, TestLazyCopyStringLiteral) {
  char charArr[21] = "Error 1, mutate this";
  mlir::MLIRContext context;
  Diagnostic diagnostic(mlir::UnknownLoc::get(&context),
                        DiagnosticSeverity::Note);

  // Diagnostic contains optimization which assumes string literals are
  // represented by `const char[]` type. This is imperfect as we can sometimes
  // trick the type system as seen below.
  //
  // Still we use this to check the diagnostic is lazily storing the pointer.
  auto addToDiagnosticAsConst = [&diagnostic](const char(&charArr)[21]) {
    diagnostic << charArr;
  };
  addToDiagnosticAsConst(charArr);

  // Mutate the underlying pointer and ensure the string does change
  charArr[0] = '^';

  std::string resultMessage;
  llvm::raw_string_ostream stringStream(resultMessage);
  diagnostic.print(stringStream);
  ASSERT_STREQ("^rror 1, mutate this", resultMessage.c_str());
}

// Register counting handlers first so they receive diagnostics only after
// the verifier's registration is erased.

TEST(SourceMgrDiagnosticVerifierHandler, ScopedRegistration) {
  MLIRContext own, other;
  unsigned seen = 0;
  ScopedDiagnosticHandler counter(&other, [&](Diagnostic &) { ++seen; });
  llvm::SourceMgr sourceMgr;
  llvm::raw_null_ostream out;
  SourceMgrDiagnosticVerifierHandler verifier(
      sourceMgr, &own, out,
      SourceMgrDiagnosticVerifierHandler::Level::OnlyExpected);
  {
    std::unique_ptr<ScopedDiagnosticHandler> registration =
        verifier.registerInContext(&other);
    emitRemark(UnknownLoc::get(&other), "while registered");
    EXPECT_EQ(seen, 0u);
  }
  emitRemark(UnknownLoc::get(&other), "after the handle");
  EXPECT_EQ(seen, 1u);
  EXPECT_TRUE(succeeded(verifier.verify()));
}

TEST(SourceMgrDiagnosticVerifierHandler, ErasesOwnRegistration) {
  MLIRContext own;
  unsigned seen = 0;
  ScopedDiagnosticHandler counter(&own, [&](Diagnostic &) { ++seen; });
  {
    llvm::SourceMgr sourceMgr;
    llvm::raw_null_ostream out;
    SourceMgrDiagnosticVerifierHandler verifier(
        sourceMgr, &own, out,
        SourceMgrDiagnosticVerifierHandler::Level::OnlyExpected);
    emitRemark(UnknownLoc::get(&own), "while alive");
    EXPECT_EQ(seen, 0u);
  }
  emitRemark(UnknownLoc::get(&own), "after the verifier");
  EXPECT_EQ(seen, 1u);
}

TEST(ParallelDiagnosticHandler, NestedStackPreservesOrder) {
  MLIRContext ctx;
  std::vector<std::string> seen;
  ScopedDiagnosticHandler capture(&ctx, [&](Diagnostic &diag) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    diag.print(os);
    seen.push_back(msg);
    return success();
  });

  {
    ParallelDiagnosticHandler outer(&ctx);
    outer.setOrderIDForThread(1);
    {
      ParallelDiagnosticHandler inner(&ctx);
      // Outer sits at the bottom of the stack, so inner pushes above it.
      // Inner ids are deliberately disjoint from outer's 0/1.
      inner.setOrderIDForThread(12);
      emitError(UnknownLoc::get(&ctx), "inner-12");
      inner.setOrderIDForThread(11);
      emitError(UnknownLoc::get(&ctx), "inner-11");
      inner.eraseOrderIDForThread();
    } // inner emits sorted: inner-11, inner-12.
    // Inner popped, so outer is on top again: the inner flush was
    // attributed to outer's element, and this is held with id 1, so the
    // id-0 diagnostic below sorts first.
    emitError(UnknownLoc::get(&ctx), "outer-A");
    outer.setOrderIDForThread(0);
    emitError(UnknownLoc::get(&ctx), "outer-B");
    outer.eraseOrderIDForThread();
  } // outer emits sorted: outer-B, then the id-1 group in arrival order.
  ASSERT_EQ(seen.size(), 4u);
  EXPECT_NE(seen[0].find("outer-B"), std::string::npos);
  EXPECT_NE(seen[1].find("inner-11"), std::string::npos);
  EXPECT_NE(seen[2].find("inner-12"), std::string::npos);
  EXPECT_NE(seen[3].find("outer-A"), std::string::npos);
}

TEST(ParallelDiagnosticHandler, DeepNestingOverflowPreservesOrder) {
  MLIRContext ctx;
  std::vector<std::string> seen;
  ScopedDiagnosticHandler capture(&ctx, [&](Diagnostic &diag) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    diag.print(os);
    seen.push_back(msg);
    return success();
  });

  // Five nested handlers: the thread-local stack holds four, so the
  // innermost falls back to the map.
  constexpr unsigned kDepth = 5;
  std::vector<std::unique_ptr<ParallelDiagnosticHandler>> handlers;
  for (unsigned i = 0; i < kDepth; ++i) {
    handlers.push_back(std::make_unique<ParallelDiagnosticHandler>(&ctx));
    handlers.back()->setOrderIDForThread(0);
  }
  handlers.back()->setOrderIDForThread(2);
  emitError(UnknownLoc::get(&ctx), "x-2");
  handlers.back()->setOrderIDForThread(1);
  emitError(UnknownLoc::get(&ctx), "x-1");
  for (auto it = handlers.rbegin(); it != handlers.rend(); ++it)
    (*it)->eraseOrderIDForThread();
  // Destroy innermost first: the handler impl is a pretty-stack-trace entry,
  // which requires LIFO destruction.
  while (!handlers.empty())
    handlers.pop_back();

  ASSERT_EQ(seen.size(), 2u);
  EXPECT_NE(seen[0].find("x-1"), std::string::npos);
  EXPECT_NE(seen[1].find("x-2"), std::string::npos);
}

TEST(ParallelDiagnosticHandler, InterleavedNestingFindsDeeperEntry) {
  MLIRContext ctx;
  std::vector<std::string> seen;
  ScopedDiagnosticHandler capture(&ctx, [&](Diagnostic &diag) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    diag.print(os);
    seen.push_back(msg);
    return success();
  });

  {
    ParallelDiagnosticHandler a(&ctx); // Registered first, fires last.
    ParallelDiagnosticHandler b(&ctx); // Registered later, fires first.
    // b's loop is outer on this thread, a's nested inside it: the stack
    // holds b below a, so b's callback must scan past the top to find
    // itself.
    b.setOrderIDForThread(0);
    a.setOrderIDForThread(7);
    emitError(UnknownLoc::get(&ctx), "d1");
    a.eraseOrderIDForThread();
    b.setOrderIDForThread(1);
    emitError(UnknownLoc::get(&ctx), "d2");
    b.eraseOrderIDForThread();
  } // b flushes both, sorted.

  // b holds both, sorted: d1, d2. If b's lookup only saw the top of the
  // stack, d1 would land in a and flush after d2.
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_NE(seen[0].find("d1"), std::string::npos);
  EXPECT_NE(seen[1].find("d2"), std::string::npos);
}

TEST(ParallelDiagnosticHandler, MultithreadedOrderingHonored) {
  MLIRContext ctx;
  // Element i emits "diag-i"; whatever the thread interleaving, the flush
  // must come out in index order.
  constexpr unsigned numElements = 1000;
  std::vector<std::string> seen;
  ScopedDiagnosticHandler capture(&ctx, [&](Diagnostic &diag) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    diag.print(os);
    seen.push_back(msg);
    return success();
  });

  std::vector<unsigned> indices(numElements);
  std::iota(indices.begin(), indices.end(), 0);
  ASSERT_TRUE(succeeded(
      failableParallelForEach(&ctx, indices, [&](unsigned i) -> LogicalResult {
        emitError(UnknownLoc::get(&ctx), "diag-" + Twine(i));
        return success();
      })));

  ASSERT_EQ(seen.size(), numElements);
  for (size_t i = 0; i < seen.size(); ++i) {
    size_t pos = seen[i].find("diag-");
    ASSERT_NE(pos, std::string::npos);
    EXPECT_EQ(std::stoul(seen[i].substr(pos + 5)), i);
  }
}

TEST(SourceMgrDiagnosticVerifierHandler, OutlivesRegisteredContext) {
  // Match mlir-opt's per-buffer context lifetime.
  MLIRContext context;
  llvm::SourceMgr sourceMgr;
  llvm::raw_null_ostream out;
  SourceMgrDiagnosticVerifierHandler verifier(
      sourceMgr, &context, out,
      SourceMgrDiagnosticVerifierHandler::Level::OnlyExpected);
  {
    MLIRContext perBuffer;
    std::unique_ptr<ScopedDiagnosticHandler> registration =
        verifier.registerInContext(&perBuffer);
    emitRemark(UnknownLoc::get(&perBuffer), "in the temporary context");
  }
  emitRemark(UnknownLoc::get(&context), "verifier still usable");
  EXPECT_TRUE(succeeded(verifier.verify()));
}

} // namespace
