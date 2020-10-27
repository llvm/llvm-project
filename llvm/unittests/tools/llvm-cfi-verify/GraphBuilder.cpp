//===- llvm/unittests/llvm-cfi-verify/GraphBuilder.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../tools/llvm-cfi-verify/lib/GraphBuilder.h"
#include "../tools/llvm-cfi-verify/lib/FileAnalysis.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <sstream>

using Instr = ::llvm::cfi_verify::FileAnalysis::Instr;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Matches;
using ::testing::Pair;
using ::testing::PrintToString;
using ::testing::Property;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::testing::Value;

namespace llvm {
namespace cfi_verify {
// Printing helpers for gtest.
std::string HexStringifyContainer(const std::vector<uint64_t> &C) {
  std::stringstream Stream;
  if (C.empty()) {
    return "{ }";
  }

  Stream << "{ ";
  const auto &LastElemIt = std::end(C) - 1;

  for (auto It = std::begin(C); It != LastElemIt; ++It) {
    Stream << "0x" << std::hex << *It << ", ";
  }
  Stream << "0x" << std::hex << *LastElemIt << " }";
  return Stream.str();
}

void PrintTo(const ConditionalBranchNode &BranchNode, ::std::ostream *os) {
  *os << "ConditionalBranchNode<Address: 0x" << std::hex << BranchNode.Address
      << ", Target: 0x" << BranchNode.Target << ", Fallthrough: 0x"
      << BranchNode.Fallthrough
      << ", CFIProtection: " << BranchNode.CFIProtection << ">";
}

void PrintTo(const GraphResult &Result, ::std::ostream *os) {
  *os << "Result BaseAddress: 0x" << std::hex << Result.BaseAddress << "\n";

  if (Result.ConditionalBranchNodes.empty())
    *os << "  (No conditional branch nodes)\n";

  for (const auto &Node : Result.ConditionalBranchNodes) {
    *os << "  ";
    PrintTo(Node, os);
    *os << "\n    Fallthrough Path: " << std::hex
        << HexStringifyContainer(Result.flattenAddress(Node.Fallthrough))
        << "\n";
    *os << "    Target Path: " << std::hex
        << HexStringifyContainer(Result.flattenAddress(Node.Target)) << "\n";
  }

  if (Result.OrphanedNodes.empty())
    *os << "  (No orphaned nodes)";

  for (const auto &Orphan : Result.OrphanedNodes) {
    *os << "  Orphan (0x" << std::hex << Orphan
        << ") Path: " << HexStringifyContainer(Result.flattenAddress(Orphan))
        << "\n";
  }
}

namespace {
class ELFx86TestFileAnalysis : public FileAnalysis {
public:
  ELFx86TestFileAnalysis()
      : FileAnalysis(Triple("x86_64--"), SubtargetFeatures()) {}

  // Expose this method publicly for testing.
  void parseSectionContents(ArrayRef<uint8_t> SectionBytes,
                            object::SectionedAddress Address) {
    FileAnalysis::parseSectionContents(SectionBytes, Address);
  }

  Error initialiseDisassemblyMembers() {
    return FileAnalysis::initialiseDisassemblyMembers();
  }
};

class BasicGraphBuilderTest : public ::testing::Test {
protected:
  void SetUp() override {
    IgnoreDWARFFlag = true;
    SuccessfullyInitialised = true;
    if (auto Err = Analysis.initialiseDisassemblyMembers()) {
      handleAllErrors(std::move(Err), [&](const UnsupportedDisassembly &E) {
        SuccessfullyInitialised = false;
        outs()
            << "Note: CFIVerifyTests are disabled due to lack of x86 support "
               "on this build.\n";
      });
    }
  }

  bool SuccessfullyInitialised;
  ELFx86TestFileAnalysis Analysis;
};

MATCHER_P2(HasPath, Result, Matcher, "has path " + PrintToString(Matcher)) {
  const auto &Path = Result.flattenAddress(arg);
  *result_listener << "the path is " << PrintToString(Path);
  return Matches(Matcher)(Path);
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphTestSinglePathFallthroughUd2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02, // 0: jne 4 [+2]
          0x0f, 0x0b, // 2: ud2
          0xff, 0x10, // 4: callq *(%rax)
      },
      {0xDEADBEEF, 0x0});
  const auto Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 4, 0x0});

  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, SizeIs(1));
  EXPECT_THAT(Result.ConditionalBranchNodes,
              Each(Field(&ConditionalBranchNode::CFIProtection, Eq(true))));
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(AllOf(Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF)),
                     Field(&ConditionalBranchNode::Target,
                           HasPath(Result, ElementsAre(0xDEADBEEF + 4))),
                     Field(&ConditionalBranchNode::Fallthrough,
                           HasPath(Result, ElementsAre(0xDEADBEEF + 2))))))
      << PrintToString(Result);
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphTestSinglePathJumpUd2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02, // 0: jne 4 [+2]
          0xff, 0x10, // 2: callq *(%rax)
          0x0f, 0x0b, // 4: ud2
      },
      {0xDEADBEEF, 0x0});
  const auto Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 2, 0x0});

  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, SizeIs(1));
  EXPECT_THAT(Result.ConditionalBranchNodes,
              Each(Field(&ConditionalBranchNode::CFIProtection, Eq(true))));
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(AllOf(Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF)),
                     Field(&ConditionalBranchNode::Target,
                           HasPath(Result, ElementsAre(0xDEADBEEF + 4))),
                     Field(&ConditionalBranchNode::Fallthrough,
                           HasPath(Result, ElementsAre(0xDEADBEEF + 2))))))
      << PrintToString(Result);
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphTestDualPathDualUd2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x03, // 0: jne 5 [+3]
          0x90,       // 2: nop
          0xff, 0x10, // 3: callq *(%rax)
          0x0f, 0x0b, // 5: ud2
          0x75, 0xf9, // 7: jne 2 [-7]
          0x0f, 0x0b, // 9: ud2
      },
      {0xDEADBEEF, 0x0});
  const auto Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 3, 0x0});

  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, SizeIs(2));
  EXPECT_THAT(Result.ConditionalBranchNodes,
              Each(Field(&ConditionalBranchNode::CFIProtection, Eq(true))));
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(AllOf(
          Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF)),
          Field(&ConditionalBranchNode::Fallthrough,
                HasPath(Result, ElementsAre(0xDEADBEEF + 2, 0xDEADBEEF + 3))),
          Field(&ConditionalBranchNode::Target,
                HasPath(Result, ElementsAre(0xDEADBEEF + 5))))))
      << PrintToString(Result);
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(AllOf(
          Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF + 7)),
          Field(&ConditionalBranchNode::Fallthrough,
                HasPath(Result, ElementsAre(0xDEADBEEF + 9))),
          Field(&ConditionalBranchNode::Target,
                HasPath(Result, ElementsAre(0xDEADBEEF + 2, 0xDEADBEEF + 3))))))
      << PrintToString(Result);
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphTestDualPathSingleUd2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x05, // 0: jne 7 [+5]
          0x90,       // 2: nop
          0xff, 0x10, // 3: callq *(%rax)
          0x75, 0xfb, // 5: jne 2 [-5]
          0x0f, 0x0b, // 7: ud2
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 3, 0x0});

  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, SizeIs(2));
  EXPECT_THAT(Result.ConditionalBranchNodes,
              Each(Field(&ConditionalBranchNode::CFIProtection, Eq(true))));
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(AllOf(
          Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF)),
          Field(&ConditionalBranchNode::Fallthrough,
                HasPath(Result, ElementsAre(0xDEADBEEF + 2, 0xDEADBEEF + 3))),
          Field(&ConditionalBranchNode::Target,
                HasPath(Result, ElementsAre(0xDEADBEEF + 7))))))
      << PrintToString(Result);
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(AllOf(
          Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF + 5)),
          Field(&ConditionalBranchNode::Fallthrough,
                HasPath(Result, ElementsAre(0xDEADBEEF + 7))),
          Field(&ConditionalBranchNode::Target,
                HasPath(Result, ElementsAre(0xDEADBEEF + 2, 0xDEADBEEF + 3))))))
      << PrintToString(Result);
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphFailures) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x90,       // 0: nop
          0x75, 0xfe, // 1: jne 1 [-2]
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF, 0x0});
  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, IsEmpty());

  Result = GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 1, 0x0});
  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, IsEmpty());

  Result = GraphBuilder::buildFlowGraph(Analysis, {0xDEADC0DE, 0x0});
  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, IsEmpty());
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphNoXrefs) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0xeb, 0xfe, // 0: jmp 0 [-2]
          0xff, 0x10, // 2: callq *(%rax)
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 2, 0x0});
  EXPECT_THAT(Result.ConditionalBranchNodes, IsEmpty());
  EXPECT_THAT(Result.OrphanedNodes, ElementsAre(0xDEADBEEF + 2));
  EXPECT_THAT(Result.IntermediateNodes, IsEmpty());
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphConditionalInfiniteLoop) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0xfe, // 0: jne 0 [-2]
          0xff, 0x10, // 2: callq *(%rax)
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 2, 0x0});
  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, SizeIs(1));
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Each(AllOf(Field(&ConditionalBranchNode::CFIProtection, Eq(false)),
                 Field(&ConditionalBranchNode::Target,
                       HasPath(Result, ElementsAre(0xDEADBEEF))),
                 Field(&ConditionalBranchNode::Fallthrough,
                       HasPath(Result, ElementsAre(0xDEADBEEF + 2))))))
      << PrintToString(Result);
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphUnconditionalInfiniteLoop) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02, // 0: jne 4 [+2]
          0xeb, 0xfc, // 2: jmp 0 [-4]
          0xff, 0x10, // 4: callq *(%rax)
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 4, 0x0});
  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, SizeIs(1));
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(
          AllOf(Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF)),
                Field(&ConditionalBranchNode::Fallthrough,
                      HasPath(Result, ElementsAre(0xDEADBEEF + 2, 0xDEADBEEF))),
                Field(&ConditionalBranchNode::Target,
                      HasPath(Result, ElementsAre(0xDEADBEEF + 4))))))
      << PrintToString(Result);
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphNoFlowsToIndirection) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x00, // 0: jne 2 [+0]
          0xeb, 0xfc, // 2: jmp 0 [-4]
          0xff, 0x10, // 4: callq *(%rax)
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 4, 0x0});
  EXPECT_THAT(Result.OrphanedNodes, ElementsAre(0xDEADBEEF + 4));
  EXPECT_THAT(Result.ConditionalBranchNodes, IsEmpty());
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphLengthExceededUpwards) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x06, // 0: jne 8 [+6]
          0x90,       // 2: nop
          0x90,       // 3: nop
          0x90,       // 4: nop
          0x90,       // 5: nop
          0xff, 0x10, // 6: callq *(%rax)
          0x0f, 0x0b, // 8: ud2
      },
      {0xDEADBEEF, 0x0});
  uint64_t PrevSearchLengthForConditionalBranch =
      SearchLengthForConditionalBranch;
  SearchLengthForConditionalBranch = 2;

  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 6, 0x0});
  EXPECT_THAT(Result.OrphanedNodes, SizeIs(1));
  EXPECT_THAT(Result.OrphanedNodes,
              Each(HasPath(Result, ElementsAre(0xDEADBEEF + 4, 0xDEADBEEF + 5,
                                               0xDEADBEEF + 6))))
      << PrintToString(Result);
  EXPECT_THAT(Result.ConditionalBranchNodes, IsEmpty());

  SearchLengthForConditionalBranch = PrevSearchLengthForConditionalBranch;
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphLengthExceededDownwards) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02, // 0: jne 4 [+2]
          0xff, 0x10, // 2: callq *(%rax)
          0x90,       // 4: nop
          0x90,       // 5: nop
          0x90,       // 6: nop
          0x90,       // 7: nop
          0x0f, 0x0b, // 8: ud2
      },
      {0xDEADBEEF, 0x0});
  uint64_t PrevSearchLengthForUndef = SearchLengthForUndef;
  SearchLengthForUndef = 2;

  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 2, 0x0});
  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Each(AllOf(
          Field(&ConditionalBranchNode::CFIProtection, Eq(false)),
          Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF)),
          Field(&ConditionalBranchNode::Target,
                HasPath(Result, ElementsAre(0xDEADBEEF + 4, 0xDEADBEEF + 5))),
          Field(&ConditionalBranchNode::Fallthrough,
                HasPath(Result, ElementsAre(0xDEADBEEF + 2))))))
      << PrintToString(Result);

  SearchLengthForUndef = PrevSearchLengthForUndef;
}

// This test ensures when avoiding doing repeated work we still generate the
// paths correctly. We don't need to recalculate the flow from 0x2 -> 0x3 as it
// should only need to be generated once.
TEST_F(BasicGraphBuilderTest, BuildFlowGraphWithRepeatedWork) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x05, // 0: jne 7 [+5]
          0x90,       // 2: nop
          0xff, 0x10, // 3: callq *(%rax)
          0x75, 0xfb, // 5: jne 2 [-5]
          0x0f, 0x0b, // 7: ud2
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 3, 0x0});
  EXPECT_THAT(Result.OrphanedNodes, IsEmpty());
  EXPECT_THAT(Result.ConditionalBranchNodes, SizeIs(2));
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(AllOf(
          Field(&ConditionalBranchNode::CFIProtection, Eq(true)),
          Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF)),
          Field(&ConditionalBranchNode::Target,
                HasPath(Result, ElementsAre(0xDEADBEEF + 7))),
          Field(&ConditionalBranchNode::Fallthrough,
                HasPath(Result, ElementsAre(0xDEADBEEF + 2, 0xDEADBEEF + 3))))))
      << PrintToString(Result);
  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(AllOf(
          Field(&ConditionalBranchNode::CFIProtection, Eq(true)),
          Field(&ConditionalBranchNode::Address, Eq(0xDEADBEEF + 5)),
          Field(&ConditionalBranchNode::Target,
                HasPath(Result, ElementsAre(0xDEADBEEF + 2, 0xDEADBEEF + 3))),
          Field(&ConditionalBranchNode::Fallthrough,
                HasPath(Result, ElementsAre(0xDEADBEEF + 7))))))
      << PrintToString(Result);
  EXPECT_THAT(Result.IntermediateNodes, SizeIs(1));
  EXPECT_THAT(Result.IntermediateNodes,
              UnorderedElementsAre(Pair(0xDEADBEEF + 2, 0xDEADBEEF + 3)));
}

TEST_F(BasicGraphBuilderTest, BuildFlowGraphComplexExample) {
  if (!SuccessfullyInitialised)
    return;
  // The following code has this graph:
  //  +----------+      +--------------+
  //  |    20    | <--- |      0       |
  //  +----------+      +--------------+
  //    |                 |
  //    v                 v
  //  +----------+      +--------------+
  //  |    21    |      |      2       |
  //  +----------+      +--------------+
  //    |                 |
  //    v                 v
  //  +----------+      +--------------+
  //  | 22 (ud2) |  +-> |      7       |
  //  +----------+  |   +--------------+
  //    ^           |     |
  //    |           |     v
  //  +----------+  |   +--------------+
  //  |    4     |  |   |      8       |
  //  +----------+  |   +--------------+
  //    |           |     |
  //    v           |     v
  //  +----------+  |   +--------------+    +------------+
  //  |    6     | -+   | 9 (indirect) | <- |     13     |
  //  +----------+      +--------------+    +------------+
  //                      ^                   |
  //                      |                   v
  //                    +--------------+    +------------+
  //                    |      11      |    | 15 (error) |
  //                    +--------------+    +------------+
  // Or, in image format: https://i.imgur.com/aX5fCoi.png

  Analysis.parseSectionContents(
      {
          0x75, 0x12,                   // 0: jne 20 [+18]
          0xeb, 0x03,                   // 2: jmp 7 [+3]
          0x75, 0x10,                   // 4: jne 22 [+16]
          0x90,                         // 6: nop
          0x90,                         // 7: nop
          0x90,                         // 8: nop
          0xff, 0x10,                   // 9: callq *(%rax)
          0xeb, 0xfc,                   // 11: jmp 9 [-4]
          0x75, 0xfa,                   // 13: jne 9 [-6]
          0xe8, 0x78, 0x56, 0x34, 0x12, // 15: callq OUTOFBOUNDS [+0x12345678]
          0x90,                         // 20: nop
          0x90,                         // 21: nop
          0x0f, 0x0b,                   // 22: ud2
      },
      {0x1000, 0x0});
  uint64_t PrevSearchLengthForUndef = SearchLengthForUndef;
  SearchLengthForUndef = 5;

  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0x1000 + 9, 0x0});

  EXPECT_THAT(Result.OrphanedNodes, SizeIs(1));
  EXPECT_THAT(Result.ConditionalBranchNodes, SizeIs(3));

  EXPECT_THAT(
      Result.OrphanedNodes,
      Each(AllOf(Eq(0x1000u + 11),
                 HasPath(Result, ElementsAre(0x1000 + 11, 0x1000 + 9)))))
      << PrintToString(Result);

  EXPECT_THAT(Result.ConditionalBranchNodes,
              Contains(AllOf(
                  Field(&ConditionalBranchNode::CFIProtection, Eq(true)),
                  Field(&ConditionalBranchNode::Address, Eq(0x1000u)),
                  Field(&ConditionalBranchNode::Target,
                        HasPath(Result, ElementsAre(0x1000 + 20, 0x1000 + 21,
                                                    0x1000 + 22))),
                  Field(&ConditionalBranchNode::Fallthrough,
                        HasPath(Result, ElementsAre(0x1000 + 2, 0x1000 + 7,
                                                    0x1000 + 8, 0x1000 + 9))))))
      << PrintToString(Result);

  EXPECT_THAT(Result.ConditionalBranchNodes,
              Contains(AllOf(
                  Field(&ConditionalBranchNode::CFIProtection, Eq(true)),
                  Field(&ConditionalBranchNode::Address, Eq(0x1000u + 4)),
                  Field(&ConditionalBranchNode::Target,
                        HasPath(Result, ElementsAre(0x1000 + 22))),
                  Field(&ConditionalBranchNode::Fallthrough,
                        HasPath(Result, ElementsAre(0x1000 + 6, 0x1000 + 7,
                                                    0x1000 + 8, 0x1000 + 9))))))
      << PrintToString(Result);

  EXPECT_THAT(
      Result.ConditionalBranchNodes,
      Contains(AllOf(Field(&ConditionalBranchNode::CFIProtection, Eq(false)),
                     Field(&ConditionalBranchNode::Address, Eq(0x1000u + 13)),
                     Field(&ConditionalBranchNode::Target,
                           HasPath(Result, ElementsAre(0x1000 + 9))),
                     Field(&ConditionalBranchNode::Fallthrough,
                           HasPath(Result, ElementsAre(0x1000 + 15))))))
      << PrintToString(Result);

  SearchLengthForUndef = PrevSearchLengthForUndef;
}

} // anonymous namespace
} // end namespace cfi_verify
} // end namespace llvm
