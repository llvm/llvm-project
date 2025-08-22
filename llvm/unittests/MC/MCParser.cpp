//===- llvm/unittest/MC/MCParser.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class MCAsmParserTest : public ::testing::Test {
public:
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;

  MCAsmParserTest() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();

    std::string TripleName = sys::getDefaultTargetTriple();
    Triple TT(Triple::normalize(TripleName));

    std::string ErrorStr;

    const Target *TheTarget = TargetRegistry::lookupTarget(TT, ErrorStr);

    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOptions;
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
  }
};
} // namespace

TEST_F(MCAsmParserTest, CheckNullTerminator) {
  if (!MAI)
    GTEST_SKIP();
  AsmLexer Lexer(*MAI);
  const char *Source = "ret\0 ";
  StringRef SourceRef(Source, 4); // Include NULL terminator in buffer length
  EXPECT_DEATH(Lexer.setBuffer(SourceRef),
               "Buffer provided to AsmLexer lacks null terminator.");
}
