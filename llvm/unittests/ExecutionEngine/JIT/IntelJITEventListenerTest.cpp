//===- JITEventListenerTest.cpp - Tests for Intel JITEventListener --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "JITEventListenerTestCommon.h"

using namespace llvm;

// Because we want to keep the implementation details of the Intel API used to
// communicate with Amplifier out of the public header files, the header below
// is included from the source tree instead.
#include "../../../lib/ExecutionEngine/IntelJITEvents/IntelJITEventsWrapper.h"

#include <map>
#include <list>

namespace {

// map of function ("method") IDs to source locations
NativeCodeMap ReportedDebugFuncs;

} // namespace

/// Mock implementaion of Intel JIT API jitprofiling library
namespace test_jitprofiling {

int NotifyEvent(iJIT_JVM_EVENT EventType, void *EventSpecificData) {
  switch (EventType) {
    case iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED: {
      EXPECT_TRUE(0 != EventSpecificData);
      iJIT_Method_Load* msg = static_cast<iJIT_Method_Load*>(EventSpecificData);

      ReportedDebugFuncs[msg->method_id];

      for(unsigned int i = 0; i < msg->line_number_size; ++i) {
        EXPECT_TRUE(0 != msg->line_number_table);
        std::pair<std::string, unsigned int> loc(
          std::string(msg->source_file_name),
          msg->line_number_table[i].LineNumber);
        ReportedDebugFuncs[msg->method_id].push_back(loc);
      }
    }
    break;
    case iJVM_EVENT_TYPE_METHOD_UNLOAD_START: {
      EXPECT_TRUE(0 != EventSpecificData);
      unsigned int UnloadId
        = *reinterpret_cast<unsigned int*>(EventSpecificData);
      EXPECT_TRUE(1 == ReportedDebugFuncs.erase(UnloadId));
    }
    default:
      break;
  }
  return 0;
}

iJIT_IsProfilingActiveFlags IsProfilingActive(void) {
  // for testing, pretend we have an Intel Parallel Amplifier XE 2011
  // instance attached
  return iJIT_SAMPLING_ON;
}

unsigned int GetNewMethodID(void) {
  static unsigned int id = 0;
  return ++id;
}

} //namespace test_jitprofiling

class IntelJITEventListenerTest
  : public JITEventListenerTestBase<IntelJITEventsWrapper> {
public:
  IntelJITEventListenerTest()
  : JITEventListenerTestBase<IntelJITEventsWrapper>(
      new IntelJITEventsWrapper(test_jitprofiling::NotifyEvent, 0,
        test_jitprofiling::IsProfilingActive, 0, 0,
        test_jitprofiling::GetNewMethodID))
  {
    EXPECT_TRUE(0 != MockWrapper);

    Listener.reset(JITEventListener::createIntelJITEventListener(
      MockWrapper.take()));
    EXPECT_TRUE(0 != Listener);
    EE->RegisterJITEventListener(Listener.get());
  }
};

TEST_F(IntelJITEventListenerTest, NoDebugInfo) {
  TestNoDebugInfo(ReportedDebugFuncs);
}

TEST_F(IntelJITEventListenerTest, SingleLine) {
  TestSingleLine(ReportedDebugFuncs);
}

TEST_F(IntelJITEventListenerTest, MultipleLines) {
  TestMultipleLines(ReportedDebugFuncs);
}

// This testcase is disabled because the Intel JIT API does not support a single
// JITted function with source lines associated with multiple files
/*
TEST_F(IntelJITEventListenerTest, MultipleFiles) {
  TestMultipleFiles(ReportedDebugFuncs);
}
*/

testing::Environment* const jit_env =
  testing::AddGlobalTestEnvironment(new JITEnvironment);
