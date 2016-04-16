//===- unittest/ProfileData/ProfileSummaryTest.cpp --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/ProfileData/SampleProf.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace sampleprof;

struct ProfileSummaryTest : ::testing::Test {
  InstrProfSummary IPS;
  SampleProfileSummary SPS;

  ProfileSummaryTest()
      : IPS({100000, 900000, 999999}), SPS({100000, 900000, 999999}) {}
  void SetUp() {
    InstrProfRecord Record1("func1", 0x1234, {97531, 5, 99999});
    InstrProfRecord Record2("func2", 0x1234, {57341, 10000, 10, 1});
    IPS.addRecord(Record1);
    IPS.addRecord(Record2);

    IPS.computeDetailedSummary();

    FunctionSamples FooSamples;
    FooSamples.addTotalSamples(7711);
    FooSamples.addHeadSamples(610);
    FooSamples.addBodySamples(1, 0, 610);
    FooSamples.addBodySamples(2, 0, 600);
    FooSamples.addBodySamples(4, 0, 60000);
    FooSamples.addBodySamples(8, 0, 60351);
    FooSamples.addBodySamples(10, 0, 605);

    FunctionSamples BarSamples;
    BarSamples.addTotalSamples(20301);
    BarSamples.addHeadSamples(1437);
    BarSamples.addBodySamples(1, 0, 1437);

    SPS.addRecord(FooSamples);
    SPS.addRecord(BarSamples);

    SPS.computeDetailedSummary();
  }

};

TEST_F(ProfileSummaryTest, summary_from_module) {
  LLVMContext Context;
  Module M1("M1", Context);
  EXPECT_FALSE(ProfileSummary::getProfileSummary(&M1));
  M1.setProfileSummary(IPS.getMD(Context));
  EXPECT_TRUE(IPS == *ProfileSummary::getProfileSummary(&M1));

  Module M2("M2", Context);
  EXPECT_FALSE(ProfileSummary::getProfileSummary(&M2));
  M2.setProfileSummary(SPS.getMD(Context));
  EXPECT_TRUE(SPS == *ProfileSummary::getProfileSummary(&M2));
}
