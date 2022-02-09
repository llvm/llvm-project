//===- llvm/unittest/MC/MCInstPrinter.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class MCInstPrinterTest : public ::testing::Test {
public:
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<const MCInstrInfo> MII;
  std::unique_ptr<MCInstPrinter> Printer;

  MCInstPrinterTest() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();

    std::string TripleName = "x86_64-pc-linux";
    std::string ErrorStr;

    const Target *TheTarget =
        TargetRegistry::lookupTarget(TripleName, ErrorStr);

    // If we didn't build x86, do not run the test.
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TripleName));
    MCTargetOptions MCOptions;
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
    MII.reset(TheTarget->createMCInstrInfo());
    Printer.reset(TheTarget->createMCInstPrinter(
        Triple(TripleName), MAI->getAssemblerDialect(), *MAI, *MII, *MRI));
  }

  template <typename T> std::string formatHex(T i) {
    std::string Buffer;
    raw_string_ostream OS(Buffer);
    OS << Printer->formatHex(i);
    OS.flush();
    return Buffer;
  }
};
} // namespace

TEST_F(MCInstPrinterTest, formatHex) {
  if (!Printer)
    return;

  EXPECT_EQ("0x1", formatHex<int64_t>(1));
  EXPECT_EQ("0x7fffffffffffffff",
            formatHex(std::numeric_limits<int64_t>::max()));
  EXPECT_EQ("-0x8000000000000000",
            formatHex(std::numeric_limits<int64_t>::min()));
}
