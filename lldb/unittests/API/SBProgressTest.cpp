//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "gtest/gtest.h"

#include "TestingSupport/SubsystemRAII.h"

#include "lldb/API/LLDB.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"

using namespace lldb_private;
using namespace lldb;

class SBProgressTest : public testing::Test {
protected:
  void SetUp() override {
    debugger = SBDebugger::Create(/*source_init_files=*/false);
    debugger.SetAsync(false);
  }

  void TearDown() override { SBDebugger::Destroy(debugger); }

  SubsystemRAII<SBDebugger> subsystems;
  SBDebugger debugger;
};

TEST_F(SBProgressTest, Constructor) {
  // Calling the constructor with null values does not crash.
  SBProgress progress(nullptr, nullptr, debugger);
  SBProgress progress2(nullptr, nullptr, /*total_units=*/10, debugger);

  SBListener listener("Test Listener");
  SBBroadcaster broadcaster = debugger.GetBroadcaster();
  broadcaster.AddListener(listener, lldb::eBroadcastBitExternalProgress);

  progress.Increment(2, nullptr);
  progress.Increment(2, "Other");
  progress.Finalize();

  SBEvent event;
  EXPECT_TRUE(listener.GetNextEvent(event));

  const SBStructuredData sdata = SBDebugger::GetProgressDataFromEvent(event);
  SBStream stream;
  sdata.GetAsJSON(stream);
  llvm::StringRef data(stream.GetData(), stream.GetSize());
  llvm::Expected<llvm::json::Value> progress_data = llvm::json::parse(data);

  ASSERT_THAT_EXPECTED(progress_data, llvm::Succeeded());
  auto *proj_json = progress_data->getAsObject();
  ASSERT_NE(proj_json, nullptr);
  for (llvm::StringRef field : {"message", "title", "details"}) {
    auto field_val = proj_json->getString(field);
    ASSERT_TRUE(field_val.has_value());
    EXPECT_EQ(field_val.value(), llvm::StringRef(""));
  }

  EXPECT_TRUE(event.GetDescription(stream));
}
