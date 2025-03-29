//===- AArch64InstPrinterTest.cpp - AArch64InstPrinter unit tests----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"

#include "MCTargetDesc/AArch64InstPrinter.h"

#include "gtest/gtest.h"

using namespace llvm;

class AArch64InstPrinterTest : public AArch64InstPrinter {
public:
  AArch64InstPrinterTest(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                         const MCRegisterInfo &MRI)
      : AArch64InstPrinter(MAI, MII, MRI) {}
  void printAlignedLabel(const MCInst *MI, uint64_t Address, unsigned OpNum,
                         const MCSubtargetInfo &STI, raw_ostream &O) {
    AArch64InstPrinter::printAlignedLabel(MI, Address, OpNum, STI, O);
  }
};

static std::string AArch64InstPrinterTestPrintAlignedLabel(uint64_t value) {
  MCAsmInfo MAI;
  MCInstrInfo MII;
  MCRegisterInfo MRI;
  MCSubtargetInfo STI(Triple(""), "", "", "", {},
                      ArrayRef((SubtargetFeatureKV *)NULL, (size_t)0),
                      ArrayRef((SubtargetSubTypeKV *)NULL, (size_t)0), NULL,
                      NULL, NULL, NULL, NULL, NULL);
  MCContext Ctx(Triple(""), &MAI, &MRI, &STI);
  MCInst MI;

  MI.addOperand(MCOperand::createExpr(MCConstantExpr::create(value, Ctx)));

  std::string str;
  raw_string_ostream O(str);
  AArch64InstPrinterTest(MAI, MII, MRI).printAlignedLabel(&MI, 0, 0, STI, O);
  return str;
}

TEST(AArch64InstPrinterTest, PrintAlignedLabel) {
  EXPECT_EQ(AArch64InstPrinterTestPrintAlignedLabel(0x0), "0x0");
  EXPECT_EQ(AArch64InstPrinterTestPrintAlignedLabel(0xffffffff001200eb),
            "0xffffffff001200eb");
  EXPECT_EQ(AArch64InstPrinterTestPrintAlignedLabel(0x7c01445bcc10f),
            "0x7c01445bcc10f");
}
