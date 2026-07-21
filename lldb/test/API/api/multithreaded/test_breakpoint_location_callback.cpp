
// LLDB C++ API Test: verify that the function registered with
// SBBreakpoint.SetCallback() is invoked when a breakpoint is hit.

#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFileSpecList.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"

#include "common.h"

using namespace std;
using namespace lldb;

mutex g_mutex;
condition_variable g_condition;
int g_breakpoint_hit_count = 0;

bool BPCallback(void *baton, SBProcess &process, SBThread &thread,
                SBBreakpointLocation &location) {
  lock_guard<mutex> lock(g_mutex);
  g_breakpoint_hit_count += 1;
  g_condition.notify_all();
  return true;
}

void test(SBDebugger &dbg, vector<string> args) {
  dbg.SetAsync(false);
  SBTarget target = dbg.CreateTarget(args.at(0).c_str());
  if (!target.IsValid())
    throw Exception("invalid target");

  // Only look for the breakpoint in the main module.
  SBFileSpec main_module(args.at(0).c_str(), /*resolve=*/true);
  SBFileSpecList module_list;
  module_list.Append(main_module);

  SBBreakpoint breakpoint = target.BreakpointCreateByName(
      "next", eFunctionNameTypeFull, module_list, SBFileSpecList());
  if (!breakpoint.IsValid())
    throw Exception("invalid breakpoint");

  if (breakpoint.GetNumLocations() != 1)
    throw Exception("unexpected amount of breakpoint locations");
  SBBreakpointLocation breakpoint_location = breakpoint.GetLocationAtIndex(0);
  breakpoint_location.SetCallback(BPCallback, 0);

  std::unique_ptr<char> working_dir(get_working_dir());
  SBProcess process = target.LaunchSimple(0, 0, working_dir.get());

  {
    unique_lock<mutex> lock(g_mutex);
    g_condition.wait_for(lock, chrono::seconds(5));
    if (g_breakpoint_hit_count != 1)
      throw Exception("Breakpoint hit count expected to be 1");
  }
}
