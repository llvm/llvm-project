//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualOutputConfig.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::vfs;

namespace {

TEST(VirtualOutputConfigTest, construct) {
  // Test defaults.
  EXPECT_FALSE(OutputConfig().getText());
  EXPECT_FALSE(OutputConfig().getCRLF());
  EXPECT_TRUE(OutputConfig().getDiscardOnSignal());
  EXPECT_TRUE(OutputConfig().getAtomicWrite());
  EXPECT_TRUE(OutputConfig().getImplyCreateDirectories());
  EXPECT_FALSE(OutputConfig().getOnlyIfDifferent());
  EXPECT_FALSE(OutputConfig().getAppend());

  // Test inverted defaults.
  EXPECT_TRUE(OutputConfig().getNoText());
  EXPECT_TRUE(OutputConfig().getNoCRLF());
  EXPECT_FALSE(OutputConfig().getNoDiscardOnSignal());
  EXPECT_FALSE(OutputConfig().getNoAtomicWrite());
  EXPECT_FALSE(OutputConfig().getNoImplyCreateDirectories());
  EXPECT_TRUE(OutputConfig().getNoOnlyIfDifferent());
  EXPECT_TRUE(OutputConfig().getNoAppend());
}

TEST(VirtualOutputConfigTest, set) {
  // Check a flag that defaults to false. Try both 'get's, all three 'set's,
  // and turning back off after turning it on.
  ASSERT_TRUE(OutputConfig().getNoText());
  EXPECT_TRUE(OutputConfig().setText().getText());
  EXPECT_FALSE(OutputConfig().setText().getNoText());
  EXPECT_TRUE(OutputConfig().setText(true).getText());
  EXPECT_FALSE(OutputConfig().setText().setNoText().getText());
  EXPECT_FALSE(OutputConfig().setText().setText(false).getText());

  // Check a flag that defaults to true. Try both 'get's, all three 'set's, and
  // turning back on after turning it off.
  ASSERT_TRUE(OutputConfig().getDiscardOnSignal());
  EXPECT_FALSE(OutputConfig().setNoDiscardOnSignal().getDiscardOnSignal());
  EXPECT_TRUE(OutputConfig().setNoDiscardOnSignal().getNoDiscardOnSignal());
  EXPECT_FALSE(OutputConfig().setDiscardOnSignal(false).getDiscardOnSignal());
  EXPECT_TRUE(OutputConfig()
                  .setNoDiscardOnSignal()
                  .setDiscardOnSignal()
                  .getDiscardOnSignal());
  EXPECT_TRUE(OutputConfig()
                  .setNoDiscardOnSignal()
                  .setDiscardOnSignal(true)
                  .getDiscardOnSignal());

  // Set multiple flags.
  OutputConfig Config;
  Config.setText().setNoDiscardOnSignal().setNoImplyCreateDirectories();
  EXPECT_TRUE(Config.getText());
  EXPECT_TRUE(Config.getNoDiscardOnSignal());
  EXPECT_TRUE(Config.getNoImplyCreateDirectories());
}

TEST(VirtualOutputConfigTest, equals) {
  EXPECT_TRUE(OutputConfig() == OutputConfig());
  EXPECT_FALSE(OutputConfig() != OutputConfig());
  EXPECT_EQ(OutputConfig().setAtomicWrite(), OutputConfig().setAtomicWrite());
  EXPECT_NE(OutputConfig().setAtomicWrite(), OutputConfig().setNoAtomicWrite());
}

static std::string toString(OutputConfig Config) {
  std::string Printed;
  raw_string_ostream OS(Printed);
  Config.print(OS);
  return Printed;
}

TEST(VirtualOutputConfigTest, print) {
  EXPECT_EQ("{}", toString(OutputConfig()));
  EXPECT_EQ("{Text}", toString(OutputConfig().setText()));
  EXPECT_EQ("{Text,NoDiscardOnSignal}",
            toString(OutputConfig().setText().setNoDiscardOnSignal()));
  EXPECT_EQ("{Text,NoDiscardOnSignal}",
            toString(OutputConfig().setNoDiscardOnSignal().setText()));
}

TEST(VirtualOutputConfigTest, BinaryAndTextWithCRLF) {
  // Test defaults.
  EXPECT_TRUE(OutputConfig().getBinary());
  EXPECT_FALSE(OutputConfig().getTextWithCRLF());
  EXPECT_FALSE(OutputConfig().getText());
  EXPECT_FALSE(OutputConfig().getCRLF());

  // Test setting.
  EXPECT_TRUE(OutputConfig().setTextWithCRLF().getTextWithCRLF());
  EXPECT_TRUE(OutputConfig().setTextWithCRLF().getText());
  EXPECT_TRUE(OutputConfig().setTextWithCRLF().getCRLF());
  EXPECT_TRUE(OutputConfig().setText().setCRLF().getTextWithCRLF());
  EXPECT_FALSE(OutputConfig().setText().getBinary());
  EXPECT_FALSE(OutputConfig().setTextWithCRLF().getBinary());
  EXPECT_FALSE(OutputConfig().setTextWithCRLF().setBinary().getText());
  EXPECT_FALSE(OutputConfig().setTextWithCRLF().setBinary().getCRLF());

  // Test setTextWithCRLF(bool).
  EXPECT_TRUE(OutputConfig().setBinary().setTextWithCRLF(true).getText());
  EXPECT_TRUE(OutputConfig().setBinary().setTextWithCRLF(true).getCRLF());
  EXPECT_TRUE(
      OutputConfig().setTextWithCRLF().setTextWithCRLF(false).getBinary());

  // Test printing.
  EXPECT_EQ("{Text,CRLF}", toString(OutputConfig().setTextWithCRLF()));
}

TEST(VirtualOutputConfigTest, OpenFlags) {
  using namespace llvm::sys::fs;

  // Confirm the default is binary.
  ASSERT_EQ(OutputConfig().setBinary(), OutputConfig());

  // Most flags are not supported / have no effect.
  EXPECT_EQ(OutputConfig(), OutputConfig().setOpenFlags(OF_None));
  EXPECT_EQ(OutputConfig(), OutputConfig().setOpenFlags(OF_Delete));
  EXPECT_EQ(OutputConfig(), OutputConfig().setOpenFlags(OF_ChildInherit));
  EXPECT_EQ(OutputConfig(), OutputConfig().setOpenFlags(OF_UpdateAtime));

  // Check setting OF_Text and OF_CRLF.
  for (OutputConfig Init : {
           OutputConfig(),
           OutputConfig().setText(),
           OutputConfig().setTextWithCRLF(),
           OutputConfig().setAppend(),

           // Should be overridden despite being invalid.
           OutputConfig().setCRLF(),
       }) {
    EXPECT_EQ(OutputConfig(), Init.setOpenFlags(OF_None));
    EXPECT_EQ(OutputConfig(), Init.setOpenFlags(OF_CRLF));
    EXPECT_EQ(OutputConfig().setText(), Init.setOpenFlags(OF_Text));
    EXPECT_EQ(OutputConfig().setTextWithCRLF(),
              Init.setOpenFlags(OF_TextWithCRLF));
    EXPECT_EQ(OutputConfig().setAppend(), Init.setOpenFlags(OF_Append));
  }
}

} // anonymous namespace
