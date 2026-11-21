//===- unittests/MC/TargetRegistry.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The target registry code lives in Support, but it relies on linking in all
// LLVM targets. We keep this test with the MC tests, which already do that, to
// keep the SupportTests target small.

#include "llvm/MC/TargetRegistry.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCRelocationInfo.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(TargetRegistry, TargetHasArchType) {
  // Presence of at least one target will be asserted when done with the loop,
  // else this would pass by accident if InitializeAllTargetInfos were omitted.
  int Count = 0;

  llvm::InitializeAllTargetInfos();

  for (const Target &T : TargetRegistry::targets()) {
    StringRef Name = T.getName();
    // There is really no way (at present) to ask a Target whether it targets
    // a specific architecture, because the logic for that is buried in a
    // predicate.
    // We can't ask the predicate "Are you a function that always returns
    // false?"
    // So given that the cpp backend truly has no target arch, it is skipped.
    if (Name != "cpp") {
      Triple::ArchType Arch = Triple::getArchTypeForLLVMName(Name);
      EXPECT_NE(Arch, Triple::UnknownArch);
      ++Count;
    }
  }
  ASSERT_NE(Count, 0);
}

TEST(TargetRegistry, IsValidFeatureListFormat) {
  // Valid strings

  // Empty string is a valid feature string
  EXPECT_TRUE(Target::isValidFeatureListFormat(""));

  EXPECT_TRUE(Target::isValidFeatureListFormat("+some_feature"));
  EXPECT_TRUE(Target::isValidFeatureListFormat("-some_feature"));
  EXPECT_TRUE(
      Target::isValidFeatureListFormat("+feature1,-feature2,+feature3"));
  EXPECT_TRUE(Target::isValidFeatureListFormat("+123"));

  // Strings with single trailing comma are also valid
  EXPECT_TRUE(Target::isValidFeatureListFormat("-feature,"));
  EXPECT_TRUE(
      Target::isValidFeatureListFormat("-feature1,+feature2,+feature3,"));

  // Invalid strings

  // Feature don't start with '+' or '-'
  EXPECT_FALSE(Target::isValidFeatureListFormat("invalid_string"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("+good,bad"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("bad,+good"));

  // String has spaces
  EXPECT_FALSE(Target::isValidFeatureListFormat(" "));
  EXPECT_FALSE(Target::isValidFeatureListFormat(", "));
  EXPECT_FALSE(Target::isValidFeatureListFormat(" avx"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("+avx, -sse"));

  // Redundant commas
  EXPECT_FALSE(Target::isValidFeatureListFormat("+feature1,,+feature2"));
  EXPECT_FALSE(Target::isValidFeatureListFormat(",+feature"));
  EXPECT_FALSE(
      Target::isValidFeatureListFormat("+feature1,,,+feature2,,+feature3"));

  // Feature consists only of '+' or '-'
  EXPECT_FALSE(Target::isValidFeatureListFormat("+"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("-"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("+avx,-"));

  // Only commas
  EXPECT_FALSE(Target::isValidFeatureListFormat(","));
  EXPECT_FALSE(Target::isValidFeatureListFormat(",,"));
  EXPECT_FALSE(Target::isValidFeatureListFormat(",,,"));
}

TEST(TargetRegistry, SubtargetCopyPreservesHwMode) {
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();

  for (StringRef TripleName :
       {"x86_64-unknown-linux-gnu", "riscv64-unknown-elf"}) {
    SCOPED_TRACE(TripleName);
    Triple TT(TripleName);
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget)
      continue;
    std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOptions;
    std::unique_ptr<MCAsmInfo> MAI(
        TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
    std::unique_ptr<MCSubtargetInfo> STI(
        TheTarget->createMCSubtargetInfo(TT, "", ""));
    ASSERT_TRUE(MRI && MAI && STI);
    ASSERT_NE(STI->getHwMode(), 0u);
    ASSERT_NE(STI->getHwModeSet(), 0u);
    MCContext Ctx(TT, *MAI, *MRI, *STI);
    MCSubtargetInfo &Copy = Ctx.getSubtargetCopy(*STI);
    EXPECT_EQ(Copy.getHwMode(), STI->getHwMode());
    EXPECT_EQ(Copy.getHwModeSet(), STI->getHwModeSet());
  }
}

// Stands in for a client's DisInfo. It starts with zeros so that a symbolizer
// that reads it as a SectionSymbolsTy sees an empty list instead of crashing.
struct SymbolizerProbe {
  void *Zeros[4] = {};
  unsigned Calls = 0;
};

static int probeGetOpInfo(void *DisInfo, uint64_t PC, uint64_t Offset,
                          uint64_t OpSize, uint64_t InstSize, int TagType,
                          void *TagBuf) {
  ++static_cast<SymbolizerProbe *>(DisInfo)->Calls;
  return 0;
}

static const char *probeSymbolLookUp(void *DisInfo, uint64_t ReferenceValue,
                                     uint64_t *ReferenceType,
                                     uint64_t ReferencePC,
                                     const char **ReferenceName) {
  ++static_cast<SymbolizerProbe *>(DisInfo)->Calls;
  *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
  *ReferenceName = nullptr;
  return nullptr;
}

// A symbolizer created with callbacks must pass DisInfo to them instead of
// reading it itself.
TEST(TargetRegistry, SymbolizerUsesClientCallbacks) {
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  int Count = 0;
  for (const Target &T : TargetRegistry::targets()) {
    SCOPED_TRACE(T.getName());
    Triple TT;
    TT.setArch(Triple::getArchTypeForLLVMName(T.getName()));
    std::unique_ptr<MCRegisterInfo> MRI(T.createMCRegInfo(TT));
    if (!MRI)
      continue;
    MCTargetOptions MCOptions;
    std::unique_ptr<MCAsmInfo> MAI(T.createMCAsmInfo(*MRI, TT, MCOptions));
    std::unique_ptr<MCSubtargetInfo> STI(T.createMCSubtargetInfo(TT, "", ""));
    if (!MAI || !STI)
      continue;
    MCContext Ctx(TT, *MAI, *MRI, *STI);
    std::unique_ptr<MCRelocationInfo> RelInfo(
        T.createMCRelocationInfo(TT, Ctx));
    if (!RelInfo)
      continue;

    SymbolizerProbe Probe;
    std::unique_ptr<MCSymbolizer> Symbolizer(
        T.createMCSymbolizer(TT, probeGetOpInfo, probeSymbolLookUp, &Probe,
                             &Ctx, std::move(RelInfo)));
    ASSERT_TRUE(Symbolizer);
    MCInst Inst;
    std::string Comments;
    raw_string_ostream CommentStream(Comments);
    Symbolizer->tryAddingSymbolicOperand(Inst, CommentStream, /*Value=*/0x1000,
                                         /*Address=*/0x2000, /*IsBranch=*/true,
                                         /*Offset=*/0, /*OpSize=*/4,
                                         /*InstSize=*/4);
    EXPECT_GT(Probe.Calls, 0u);
    ++Count;
  }
  ASSERT_NE(Count, 0);
}

} // end namespace
