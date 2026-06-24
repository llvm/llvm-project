//===- SPIRVSortBlocksTests.cpp ----------------------------===//
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

class SPIRVSortBlocksTest : public testing::Test {
protected:
  void TearDown() override { M.reset(); }

  bool run(StringRef Assembly) {
    assert(M == nullptr &&
           "Calling runAnalysis multiple times is unsafe. See getAnalysis().");

    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);
    assert(M && "Bad assembly. Bad test?");
    llvm::Function *F = M->getFunction("main");
    return sortBlocks(*F);
  }

  void checkBasicBlockOrder(std::vector<const char *> &&Expected) {
    llvm::Function *F = M->getFunction("main");
    auto It = F->begin();
    for (const char *Name : Expected) {
      ASSERT_TRUE(It != F->end())
          << "Expected block \"" << Name
          << "\" but reached the end of the function instead.";
      ASSERT_TRUE(It->getName() == Name)
          << "Error: expected block \"" << Name << "\" got \"" << It->getName()
          << "\"";
      It++;
    }
    ASSERT_TRUE(It == F->end())
        << "No more blocks were expected, but function has more.";
  }

protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
};

TEST_F(SPIRVSortBlocksTest, DefaultRegion) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ret void
    }
  )";

  // No sorting is required.
  EXPECT_FALSE(run(Assembly));
}

TEST_F(SPIRVSortBlocksTest, BasicBlockSwap) {
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

  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder({"entry", "middle", "exit"});
}

// Skip condition:
//         +-> A -+
//  entry -+      +-> C
//         +------+
TEST_F(SPIRVSortBlocksTest, SkipCondition) {
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

  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder({"entry", "a", "c"});
}

// Simple loop:
// entry -> header <-----------------+
//           | `-> body -> continue -+
//           `-> end
TEST_F(SPIRVSortBlocksTest, LoopOrdering) {
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

  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder({"entry", "header", "end", "body", "continue"});
}

// Diamond condition:
//         +-> A -+
//  entry -+      +-> C
//         +-> B -+
//
// A and B order can be flipped with no effect, but it must be remain
// deterministic/stable.
TEST_F(SPIRVSortBlocksTest, DiamondCondition) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br i1 %1, label %b, label %a
    c:
      ret void
    b:
      br label %c
    a:
      br label %c
    }
  )";

  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder({"entry", "a", "b", "c"});
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
TEST_F(SPIRVSortBlocksTest, CrossingCondition) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br i1 %1, label %b, label %a
    e:
      ret void
    c:
      br label %e
    b:
      br i1 %1, label %d, label %c
    d:
      br label %e
    a:
      br i1 %1, label %d, label %c
    }
  )";

  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder({"entry", "a", "b", "c", "d", "e"});
}

// Irreducible CFG
// digraph {
//    entry -> A;
//
//    A -> B;
//    A -> C;
//
//    B -> A;
//    B -> C;
//
//    C -> B;
// }
//
// Order starts with Entry and A. Order of B and C can change, but must remain
// stable.
// In such case, rank will be defined by the arbitrary traversal order. What's
// important is to have a stable value.
TEST_F(SPIRVSortBlocksTest, IrreducibleOrdering) {
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

  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder({"entry", "a", "b", "c"});
}

TEST_F(SPIRVSortBlocksTest, IrreducibleOrderingBeforeReduction) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br label %a

    c:
      br i1 %1, label %e, label %d

    e:
      ret void

    b:
      br i1 %1, label %c, label %d

    a:
      br label %b

    d:
      br i1 %1, label %b, label %c

    }
  )";

  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder({"entry", "a", "b", "c", "d", "e"});
}

TEST_F(SPIRVSortBlocksTest, LoopDiamond) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      %1 = icmp ne i32 0, 0
      br label %header
    header:
      br i1 %1, label %body, label %end
    body:
      br i1 %1, label %break, label %inside_a
    inside_a:
      br label %inside_b
    inside_b:
      br i1 %1, label %inside_d, label %inside_c
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

  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder({"entry", "header", "body", "inside_a", "inside_b",
                        "inside_c", "inside_d", "continue", "break", "end"});
}

TEST_F(SPIRVSortBlocksTest, LoopNested) {
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

  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder({"entry", "a", "b", "c", "e", "f", "d", "g", "h"});
}

TEST_F(SPIRVSortBlocksTest, IfNested) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      br i1 true, label %d, label %a
    i:
      br label %j
    j:
      ret void
    a:
      br i1 true, label %b, label %c
    d:
      br i1 true, label %f, label %e
    e:
      br label %i
    b:
      br label %c
    f:
      br i1 true, label %h, label %g
    g:
      br label %h
    c:
      br label %j
    h:
      br label %i
    }
  )";
  EXPECT_TRUE(run(Assembly));
  checkBasicBlockOrder(
      {"entry", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"});
}

// Same as above, but this time blocks are already sorted, so no need to reorder
// them.
TEST_F(SPIRVSortBlocksTest, IfNestedSorted) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
    entry:
      br i1 true, label %d, label %z
    z:
      br i1 true, label %b, label %c
    b:
      br label %c
    c:
      br label %j
    d:
      br i1 true, label %f, label %e
    e:
      br label %i
    f:
      br i1 true, label %h, label %g
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
  EXPECT_FALSE(run(Assembly));
}
