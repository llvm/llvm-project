//===- SPIRVPartialOrderingVisitorTests.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVUtils.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/Support/SourceMgr.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <queue>

using namespace llvm;
using namespace llvm::SPIRV;

class SPIRVPartialOrderingVisitorTest : public testing::Test {
protected:
  void TearDown() override { M.reset(); }

  void run(StringRef Assembly) {
    assert(M == nullptr &&
           "Calling runAnalysis multiple times is unsafe. See getAnalysis().");

    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);
    assert(M && "Bad assembly. Bad test?");

    llvm::Function *F = M->getFunction("main");
    Visitor = std::make_unique<PartialOrderingVisitor>(*F);
  }

  void
  checkBasicBlockRank(std::vector<std::pair<const char *, size_t>> &&Expected) {
    llvm::Function *F = M->getFunction("main");
    auto It = Expected.begin();
    Visitor->partialOrderVisit(*F->begin(), [&](BasicBlock *BB) {
      const auto &[Name, Rank] = *It;
      EXPECT_TRUE(It != Expected.end())
          << "Unexpected block \"" << BB->getName() << " visited.";
      EXPECT_TRUE(BB->getName() == Name)
          << "Error: expected block \"" << Name << "\" got \"" << BB->getName()
          << "\"";
      EXPECT_EQ(Rank, Visitor->GetNodeRank(BB))
          << "Bad rank for BB \"" << BB->getName() << "\"";
      It++;
      return true;
    });
    ASSERT_TRUE(It == Expected.end())
        << "Expected block \"" << It->first
        << "\" but reached the end of the function instead.";
  }

protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  std::unique_ptr<PartialOrderingVisitor> Visitor;
};

TEST_F(SPIRVPartialOrderingVisitorTest, EmptyFunction) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ret void
    }
  )";

  run(Assembly);
  checkBasicBlockRank({{"", 0}});
}

TEST_F(SPIRVPartialOrderingVisitorTest, BasicBlockSwap) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      br label %middle
    exit:
      ret void
    middle:
      br label %exit
    }
  )";

  run(Assembly);
  checkBasicBlockRank({{"entry", 0}, {"middle", 1}, {"exit", 2}});
}

// Skip condition:
//         +-> A -+
//  entry -+      +-> C
//         +------+
TEST_F(SPIRVPartialOrderingVisitorTest, SkipCondition) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br i1 %1, label %c, label %a
    c:
      ret void
    a:
      br label %c
    }
  )";

  run(Assembly);
  checkBasicBlockRank({{"entry", 0}, {"a", 1}, {"c", 2}});
}

// Simple loop:
// entry -> header <-----------------+
//           | `-> body -> continue -+
//           `-> end
TEST_F(SPIRVPartialOrderingVisitorTest, LoopOrdering) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br label %header
    end:
      ret void
    body:
      br label %continue
    continue:
      br label %header
    header:
      br i1 %1, label %body, label %end
    }
  )";

  run(Assembly);
  checkBasicBlockRank(
      {{"entry", 0}, {"header", 1}, {"body", 2}, {"continue", 3}, {"end", 4}});
}

// Diamond condition:
//         +-> A -+
//  entry -+      +-> C
//         +-> B -+
//
// A and B order can be flipped with no effect, but it must be remain
// deterministic/stable.
TEST_F(SPIRVPartialOrderingVisitorTest, DiamondCondition) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br i1 %1, label %a, label %b
    c:
      ret void
    b:
      br label %c
    a:
      br label %c
    }
  )";

  run(Assembly);
  checkBasicBlockRank({{"entry", 0}, {"a", 1}, {"b", 1}, {"c", 2}});
}

// Crossing conditions:
//             +------+  +-> C -+
//         +-> A -+   |  |      |
//  entry -+      +--_|_-+      +-> E
//         +-> B -+   |         |
//             +------+----> D -+
//
// A & B have the same rank.
// C & D have the same rank, but are after A & B.
// E if the last block.
TEST_F(SPIRVPartialOrderingVisitorTest, CrossingCondition) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br i1 %1, label %a, label %b
    e:
      ret void
    c:
      br label %e
    b:
      br i1 %1, label %d, label %c
    d:
      br label %e
    a:
      br i1 %1, label %c, label %d
    }
  )";

  run(Assembly);
  checkBasicBlockRank(
      {{"entry", 0}, {"a", 1}, {"b", 1}, {"c", 2}, {"d", 2}, {"e", 3}});
}

TEST_F(SPIRVPartialOrderingVisitorTest, LoopDiamond) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br label %header
    header:
      br i1 %1, label %body, label %end
    body:
      br i1 %1, label %inside_a, label %break
    inside_a:
      br label %inside_b
    inside_b:
      br i1 %1, label %inside_c, label %inside_d
    inside_c:
      br label %continue
    inside_d:
      br label %continue
    break:
      br label %end
    continue:
      br label %header
    end:
      ret void
    }
  )";

  run(Assembly);
  checkBasicBlockRank({{"entry", 0},
                       {"header", 1},
                       {"body", 2},
                       {"inside_a", 3},
                       {"inside_b", 4},
                       {"inside_c", 5},
                       {"inside_d", 5},
                       {"continue", 6},
                       {"break", 7},
                       {"end", 8}});
}

TEST_F(SPIRVPartialOrderingVisitorTest, LoopNested) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br label %a
    a:
      br i1 %1, label %h, label %b
    b:
      br label %c
    c:
      br i1 %1, label %d, label %e
    d:
      br label %g
    e:
      br label %f
    f:
      br label %c
    g:
      br label %a
    h:
      ret void
    }
  )";

  run(Assembly);
  checkBasicBlockRank({{"entry", 0},
                       {"a", 1},
                       {"b", 2},
                       {"c", 3},
                       {"e", 4},
                       {"f", 5},
                       {"d", 6},
                       {"g", 7},
                       {"h", 8}});
}

TEST_F(SPIRVPartialOrderingVisitorTest, IfNested) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      br i1 true, label %a, label %d
    a:
      br i1 true, label %b, label %c
    b:
      br label %c
    c:
      br label %j
    d:
      br i1 true, label %e, label %f
    e:
      br label %i
    f:
      br i1 true, label %g, label %h
    g:
      br label %h
    h:
      br label %i
    i:
      br label %j
    j:
      ret void
    }
  )";
  run(Assembly);
  checkBasicBlockRank({{"entry", 0},
                       {"a", 1},
                       {"d", 1},
                       {"b", 2},
                       {"e", 2},
                       {"f", 2},
                       {"c", 3},
                       {"g", 3},
                       {"h", 4},
                       {"i", 5},
                       {"j", 6}});
}

TEST_F(SPIRVPartialOrderingVisitorTest, CheckDeathIrreducible) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br label %a
    b:
      br i1 %1, label %a, label %c
    c:
      br label %b
    a:
      br i1 %1, label %b, label %c
    }
  )";

  ASSERT_DEATH(
      { run(Assembly); },
      "No valid candidate in the queue. Is the graph reducible?");
}
