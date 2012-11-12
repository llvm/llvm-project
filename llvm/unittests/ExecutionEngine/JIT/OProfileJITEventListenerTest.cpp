//===- OProfileJITEventListenerTest.cpp - Unit tests for OProfileJITEventsListener --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/OProfileWrapper.h"
#include "JITEventListenerTestCommon.h"

#include <map>
#include <list>

using namespace llvm;

namespace {

struct OprofileNativeFunction {
  const char* Name;
  uint64_t Addr;
  const void* CodePtr;
  unsigned int CodeSize;

  OprofileNativeFunction(const char* name,
                         uint64_t addr,
                         const void* code,
                         unsigned int size)
  : Name(name)
  , Addr(addr)
  , CodePtr(code)
  , CodeSize(size) {
  }
};

typedef std::list<OprofileNativeFunction> NativeFunctionList;
typedef std::list<debug_line_info> NativeDebugList;
NativeFunctionList NativeFunctions;

NativeCodeMap ReportedDebugFuncs;

} // namespace

/// Mock implementaion of opagent library
namespace test_opagent {

op_agent_t globalAgent = reinterpret_cast<op_agent_t>(42);

op_agent_t open_agent()
{
  // return non-null op_agent_t
  return globalAgent;
}

int close_agent(op_agent_t agent)
{
  EXPECT_EQ(globalAgent, agent);
  return 0;
}

int write_native_code(op_agent_t agent,
                      const char* name,
                      uint64_t addr,
                      void const* code,
                      unsigned int size)
{
  EXPECT_EQ(globalAgent, agent);
  OprofileNativeFunction func(name, addr, code, size);
  NativeFunctions.push_back(func);

  // Verify no other registration has take place for the same address
  EXPECT_TRUE(ReportedDebugFuncs.find(addr) == ReportedDebugFuncs.end());

  ReportedDebugFuncs[addr];
  return 0;
}

int write_debug_line_info(op_agent_t agent,
                          void const* code,
                          size_t num_entries,
                          struct debug_line_info const* info)
{
  EXPECT_EQ(globalAgent, agent);

  //verify code has been loaded first
  uint64_t addr = reinterpret_cast<uint64_t>(code);
  NativeCodeMap::iterator i = ReportedDebugFuncs.find(addr);
  EXPECT_TRUE(i != ReportedDebugFuncs.end());

  NativeDebugList NativeInfo(info, info + num_entries);

  SourceLocations locs;
  for(NativeDebugList::iterator i = NativeInfo.begin();
      i != NativeInfo.end();
      ++i) {
    locs.push_back(std::make_pair(std::string(i->filename), i->lineno));
  }
  ReportedDebugFuncs[addr] = locs;

  return 0;
}

int unload_native_code(op_agent_t agent, uint64_t addr) {
  EXPECT_EQ(globalAgent, agent);

  //verify that something for the given JIT addr has been loaded first
  NativeCodeMap::iterator i = ReportedDebugFuncs.find(addr);
  EXPECT_TRUE(i != ReportedDebugFuncs.end());
  ReportedDebugFuncs.erase(i);
  return 0;
}

int version() {
  return 1;
}

bool is_oprofile_running() {
  return true;
}

} //namespace test_opagent

class OProfileJITEventListenerTest
: public JITEventListenerTestBase<OProfileWrapper>
{
public:
  OProfileJITEventListenerTest()
  : JITEventListenerTestBase<OProfileWrapper>(
    new OProfileWrapper(test_opagent::open_agent,
      test_opagent::close_agent,
      test_opagent::write_native_code,
      test_opagent::write_debug_line_info,
      test_opagent::unload_native_code,
      test_opagent::version,
      test_opagent::version,
      test_opagent::is_oprofile_running))
  {
    EXPECT_TRUE(0 != MockWrapper);

    Listener.reset(JITEventListener::createOProfileJITEventListener(
      MockWrapper.get()));
    EXPECT_TRUE(0 != Listener);
    EE->RegisterJITEventListener(Listener.get());
  }
};

TEST_F(OProfileJITEventListenerTest, NoDebugInfo) {
  TestNoDebugInfo(ReportedDebugFuncs);
}

TEST_F(OProfileJITEventListenerTest, SingleLine) {
  TestSingleLine(ReportedDebugFuncs);
}

TEST_F(OProfileJITEventListenerTest, MultipleLines) {
  TestMultipleLines(ReportedDebugFuncs);
}

TEST_F(OProfileJITEventListenerTest, MultipleFiles) {
  TestMultipleFiles(ReportedDebugFuncs);
}

testing::Environment* const jit_env =
  testing::AddGlobalTestEnvironment(new JITEnvironment);
