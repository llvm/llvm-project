//===- unittests/IR/TimePassesTest.cpp - TimePassesHandler tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassTimingInfo.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace {

class MyPass1 : public PassInfoMixin<MyPass1> {};
class MyPass2 : public PassInfoMixin<MyPass2> {};

TEST(TimePassesTest, CustomOut) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext Context;
  Module M("TestModule", Context);
  MyPass1 Pass1;
  MyPass2 Pass2;

  SmallString<0> TimePassesStr;
  raw_svector_ostream ReportStream(TimePassesStr);

  // Setup time-passes handler and redirect output to the stream.
  std::unique_ptr<TimePassesHandler> TimePasses =
      llvm::make_unique<TimePassesHandler>(true);
  TimePasses->setOutStream(ReportStream);
  TimePasses->registerCallbacks(PIC);

  // Running some passes to trigger the timers.
  PI.runBeforePass(Pass1, M);
  PI.runBeforePass(Pass2, M);
  PI.runAfterPass(Pass2, M);
  PI.runAfterPass(Pass1, M);

  // Generating report.
  TimePasses->print();

  // There should be Pass1 and Pass2 in the report
  EXPECT_FALSE(TimePassesStr.empty());
  EXPECT_TRUE(TimePassesStr.str().contains("report"));
  EXPECT_TRUE(TimePassesStr.str().contains("Pass1"));
  EXPECT_TRUE(TimePassesStr.str().contains("Pass2"));

  // Clear and generate report again.
  TimePassesStr.clear();
  TimePasses->print();
  // Since we did not run any passes since last print, report should be empty.
  EXPECT_TRUE(TimePassesStr.empty());

  // Now run just a single pass to populate timers again.
  PI.runBeforePass(Pass2, M);
  PI.runAfterPass(Pass2, M);

  // Generate report by deleting the handler.
  TimePasses.reset();

  // There should be Pass2 in this report and no Pass1.
  EXPECT_FALSE(TimePassesStr.str().empty());
  EXPECT_TRUE(TimePassesStr.str().contains("report"));
  EXPECT_FALSE(TimePassesStr.str().contains("Pass1"));
  EXPECT_TRUE(TimePassesStr.str().contains("Pass2"));
}

} // end anonymous namespace
