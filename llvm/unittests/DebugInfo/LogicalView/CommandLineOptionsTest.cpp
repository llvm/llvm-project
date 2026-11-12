//===- llvm/unittest/DebugInfo/LogicalView/CommandLineOptionsTest.cpp -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVOptions.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::logicalview;

namespace {

// '--attribute' options.
TEST(CommandLineOptionsTest, attributeOptions) {
  auto CheckStandardAttributes = [&](LVOptions &Options, bool Value) {
    EXPECT_EQ(Options.getAttributeBase(), 1);
    EXPECT_EQ(Options.getAttributeCoverage(), Value);
    EXPECT_EQ(Options.getAttributeDirectories(), 1);
    EXPECT_EQ(Options.getAttributeDiscriminator(), 1);
    EXPECT_EQ(Options.getAttributeFilename(), 0);
    EXPECT_EQ(Options.getAttributeFiles(), 1);
    EXPECT_EQ(Options.getAttributeFormat(), 1);
    EXPECT_EQ(Options.getAttributeLevel(), 1);
    EXPECT_EQ(Options.getAttributeProducer(), 1);
    EXPECT_EQ(Options.getAttributePublics(), 1);
    EXPECT_EQ(Options.getAttributeRange(), 1);
    EXPECT_EQ(Options.getAttributeReference(), 1);
    EXPECT_EQ(Options.getAttributeZero(), 1);
  };

  auto CheckExtendedAttributes = [&](LVOptions &Options, bool Value) {
    EXPECT_EQ(Options.getAttributeArgument(), 1);
    EXPECT_EQ(Options.getAttributeDiscarded(), 1);
    EXPECT_EQ(Options.getAttributeEncoded(), 1);
    EXPECT_EQ(Options.getAttributeGaps(), Value);
    EXPECT_EQ(Options.getAttributeGenerated(), 1);
    EXPECT_EQ(Options.getAttributeGlobal(), 1);
    EXPECT_EQ(Options.getAttributeInserted(), 1);
    EXPECT_EQ(Options.getAttributeLinkage(), 1);
    EXPECT_EQ(Options.getAttributeLocal(), 1);
    EXPECT_EQ(Options.getAttributeLocation(), Value);
    EXPECT_EQ(Options.getAttributeOffset(), 1);
    EXPECT_EQ(Options.getAttributePathname(), 1);
    EXPECT_EQ(Options.getAttributeQualified(), 1);
    EXPECT_EQ(Options.getAttributeQualifier(), 1);
    EXPECT_EQ(Options.getAttributeRegister(), Value);
    EXPECT_EQ(Options.getAttributeSize(), 1);
    EXPECT_EQ(Options.getAttributeSubrange(), 1);
    EXPECT_EQ(Options.getAttributeSystem(), 1);
    EXPECT_EQ(Options.getAttributeTypename(), 1);
  };

  // Location information is only relevant when printing symbols.
  // It means the following attributes are dependent on --print=symbols:
  // Coverage, gaps, location and register attributes.
  // '--attribute=pathname' supersedes '--attribute=filename'.

  // Set standard and extended attributes.
  LVOptions OptionsOne;
  OptionsOne.setAttributeStandard();
  OptionsOne.setAttributeExtended();
  OptionsOne.resolveDependencies();
  CheckStandardAttributes(OptionsOne, false);
  CheckExtendedAttributes(OptionsOne, false);

  // Set standard and extended attributes; enable location attributes.
  LVOptions OptionsTwo;
  OptionsTwo.setAttributeStandard();
  OptionsTwo.setAttributeExtended();
  OptionsTwo.setPrintSymbols();
  OptionsTwo.resolveDependencies();
  CheckStandardAttributes(OptionsTwo, true);
  CheckExtendedAttributes(OptionsTwo, true);

  // Set all attributes.
  LVOptions OptionsThree;
  OptionsThree.setAttributeAll();
  OptionsThree.resolveDependencies();
  EXPECT_EQ(OptionsThree.getAttributeExtended(), 1);
  EXPECT_EQ(OptionsThree.getAttributeStandard(), 1);

  // Set filename attribute.
  LVOptions OptionsFour;
  OptionsFour.setAttributeFilename();
  OptionsFour.resolveDependencies();
  EXPECT_EQ(OptionsFour.getAttributeFilename(), 1);
  EXPECT_EQ(OptionsFour.getAttributePathname(), 0);

  // Set pathname attribute.
  OptionsFour.setAttributePathname();
  OptionsFour.resolveDependencies();
  EXPECT_EQ(OptionsFour.getAttributeFilename(), 0);
  EXPECT_EQ(OptionsFour.getAttributePathname(), 1);

  // The location attribute depends on: coverage, gaps or register.
  LVOptions OptionsFive;
  OptionsFive.setPrintSymbols();
  OptionsFive.resetAttributeLocation();
  OptionsFive.resolveDependencies();
  EXPECT_EQ(OptionsFive.getAttributeLocation(), 0);

  OptionsFive.resetAttributeLocation();
  OptionsFive.setAttributeCoverage();
  OptionsFive.resolveDependencies();
  EXPECT_EQ(OptionsFive.getAttributeLocation(), 1);

  OptionsFive.resetAttributeLocation();
  OptionsFive.setAttributeGaps();
  OptionsFive.resolveDependencies();
  EXPECT_EQ(OptionsFive.getAttributeLocation(), 1);

  OptionsFive.resetAttributeLocation();
  OptionsFive.setAttributeRegister();
  OptionsFive.resolveDependencies();
  EXPECT_EQ(OptionsFive.getAttributeLocation(), 1);
}

// '--compare' options.
TEST(CommandLineOptionsTest, compareOptions) {
  LVOptions OptionsOne;
  OptionsOne.setCompareAll();
  OptionsOne.resolveDependencies();
  EXPECT_EQ(OptionsOne.getCompareLines(), 1);
  EXPECT_EQ(OptionsOne.getCompareScopes(), 1);
  EXPECT_EQ(OptionsOne.getCompareSymbols(), 1);
  EXPECT_EQ(OptionsOne.getCompareTypes(), 1);

  // The compare scopes attribute depends on: symbols, types or lines.
  LVOptions OptionsTwo;
  OptionsTwo.resetCompareScopes();
  OptionsTwo.resolveDependencies();
  EXPECT_EQ(OptionsTwo.getCompareScopes(), 0);

  OptionsTwo.resetCompareScopes();
  OptionsTwo.setCompareLines();
  OptionsTwo.resolveDependencies();
  EXPECT_EQ(OptionsTwo.getCompareScopes(), 1);

  OptionsTwo.resetCompareScopes();
  OptionsTwo.setCompareSymbols();
  OptionsTwo.resolveDependencies();
  EXPECT_EQ(OptionsTwo.getCompareScopes(), 1);

  OptionsTwo.resetCompareScopes();
  OptionsTwo.setCompareTypes();
  OptionsTwo.resolveDependencies();
  EXPECT_EQ(OptionsTwo.getCompareScopes(), 1);

  // The compare option, set/reset other attributes.
  LVOptions OptionsThree;
  OptionsThree.setCompareAll();
  OptionsThree.resolveDependencies();
  EXPECT_EQ(OptionsThree.getAttributeArgument(), 1);
  EXPECT_EQ(OptionsThree.getAttributeEncoded(), 1);
  EXPECT_EQ(OptionsThree.getAttributeInserted(), 1);
  EXPECT_EQ(OptionsThree.getAttributeMissing(), 1);
  EXPECT_EQ(OptionsThree.getAttributeQualified(), 1);
}

// '--internal' options.
TEST(CommandLineOptionsTest, internalOptions) {
  LVOptions OptionsOne;
  OptionsOne.setInternalAll();
  OptionsOne.resolveDependencies();
  EXPECT_EQ(OptionsOne.getInternalCmdline(), 1);
  EXPECT_EQ(OptionsOne.getInternalID(), 1);
  EXPECT_EQ(OptionsOne.getInternalIntegrity(), 1);
  EXPECT_EQ(OptionsOne.getInternalNone(), 1);
  EXPECT_EQ(OptionsOne.getInternalTag(), 1);
}

// '--output' options.
TEST(CommandLineOptionsTest, outputOptions) {
  LVOptions OptionsOne;
  OptionsOne.setOutputAll();
  OptionsOne.resolveDependencies();
  EXPECT_EQ(OptionsOne.getOutputJson(), 1);
  EXPECT_EQ(OptionsOne.getOutputSplit(), 1);
  EXPECT_EQ(OptionsOne.getOutputText(), 1);

  // The pathname attribute is set with split output.
  LVOptions OptionsTwo;
  OptionsTwo.resetAttributePathname();
  OptionsTwo.setOutputSplit();
  OptionsTwo.resolveDependencies();
  EXPECT_EQ(OptionsTwo.getAttributePathname(), 1);

  // Setting an output folder, it sets split option.
  LVOptions OptionsThree;
  OptionsThree.resolveDependencies();
  EXPECT_EQ(OptionsThree.getOutputSplit(), 0);

  OptionsThree.setOutputFolder("folder-name");
  OptionsThree.resolveDependencies();
  EXPECT_EQ(OptionsThree.getOutputSplit(), 1);
  EXPECT_STREQ(OptionsThree.getOutputFolder().c_str(), "folder-name");

  // Assume '--output=text' as default.
  LVOptions OptionsFour;
  OptionsFour.resolveDependencies();
  EXPECT_EQ(OptionsFour.getOutputText(), 1);
}

// '--print' options.
TEST(CommandLineOptionsTest, printOptions) {
  LVOptions OptionsOne;
  OptionsOne.setPrintAll();
  OptionsOne.resolveDependencies();
  EXPECT_EQ(OptionsOne.getPrintInstructions(), 1);
  EXPECT_EQ(OptionsOne.getPrintLines(), 1);
  EXPECT_EQ(OptionsOne.getPrintScopes(), 1);
  EXPECT_EQ(OptionsOne.getPrintSizes(), 1);
  EXPECT_EQ(OptionsOne.getPrintSymbols(), 1);
  EXPECT_EQ(OptionsOne.getPrintSummary(), 1);
  EXPECT_EQ(OptionsOne.getPrintTypes(), 1);
  EXPECT_EQ(OptionsOne.getPrintWarnings(), 1);

  // '--print=elements' is a shortcut for:
  // '--print=instructions,lines,scopes,symbols,types'.
  LVOptions OptionsTwo;
  OptionsTwo.setPrintElements();
  OptionsTwo.resolveDependencies();
  EXPECT_EQ(OptionsTwo.getPrintInstructions(), 1);
  EXPECT_EQ(OptionsTwo.getPrintLines(), 1);
  EXPECT_EQ(OptionsTwo.getPrintScopes(), 1);
  EXPECT_EQ(OptionsTwo.getPrintSizes(), 0);
  EXPECT_EQ(OptionsTwo.getPrintSymbols(), 1);
  EXPECT_EQ(OptionsTwo.getPrintSummary(), 0);
  EXPECT_EQ(OptionsTwo.getPrintTypes(), 1);
  EXPECT_EQ(OptionsTwo.getPrintWarnings(), 0);
}

// '--report' options.
TEST(CommandLineOptionsTest, reportOptions) {
  LVOptions OptionsOne;
  OptionsOne.setReportAll();
  OptionsOne.resolveDependencies();
  EXPECT_EQ(OptionsOne.getReportChildren(), 1);
  EXPECT_EQ(OptionsOne.getReportList(), 1);
  EXPECT_EQ(OptionsOne.getReportParents(), 1);
  EXPECT_EQ(OptionsOne.getReportView(), 1);

  // '--report=view' is a shortcut for '--report=parents,children'.
  LVOptions OptionsTwo;
  OptionsTwo.setReportView();
  OptionsTwo.resolveDependencies();
  EXPECT_EQ(OptionsTwo.getReportChildren(), 1);
  EXPECT_EQ(OptionsTwo.getReportParents(), 1);
}

// '--select' options.
TEST(CommandLineOptionsTest, selectOptions) {
  LVOptions OptionsOne;
  OptionsOne.setSelectIgnoreCase();
  OptionsOne.setSelectUseRegex();
  OptionsOne.resolveDependencies();
  EXPECT_EQ(OptionsOne.getSelectIgnoreCase(), 1);
  EXPECT_EQ(OptionsOne.getSelectUseRegex(), 1);
}

// '--warning' options.
TEST(CommandLineOptionsTest, warningOptions) {
  LVOptions OptionsOne;
  OptionsOne.setWarningAll();
  OptionsOne.resolveDependencies();
  EXPECT_EQ(OptionsOne.getWarningCoverages(), 1);
  EXPECT_EQ(OptionsOne.getWarningLines(), 1);
  EXPECT_EQ(OptionsOne.getWarningLocations(), 1);
  EXPECT_EQ(OptionsOne.getWarningRanges(), 1);
}

} // namespace
