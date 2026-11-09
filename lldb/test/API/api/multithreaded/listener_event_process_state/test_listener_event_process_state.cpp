
// LLDB C++ API Test: verify the event description as obtained by calling
// SBEvent::GetCStringFromEvent that is received by an
// SBListener object registered with a process with a breakpoint.

#include <atomic>
#include <iostream>
#include <string>
#include <thread>

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBThread.h"

#include "common.h"

using namespace lldb;
using namespace std;

// listener thread control
extern atomic<bool> g_done;

multithreaded_queue<string> g_frame_functions;

extern SBListener g_listener;

void listener_func() {
  while (!g_done) {
    SBEvent event;
    bool got_event = g_listener.WaitForEvent(1, event);
    if (got_event) {
      if (!event.IsValid())
        throw Exception("event is not valid in listener thread");
      // send process description
      SBProcess process = SBProcess::GetProcessFromEvent(event);
      if (!process.IsValid())
        throw Exception("process is not valid");
      if (SBProcess::GetStateFromEvent(event) != lldb::eStateStopped ||
          SBProcess::GetRestartedFromEvent(event))
        continue; // Only interested in "stopped" events.

      SBStream description;

      for (int i = 0; i < process.GetNumThreads(); ++i) {
        // send each thread description
        SBThread thread = process.GetThreadAtIndex(i);
        // send each frame function name
        uint32_t num_frames = thread.GetNumFrames();
        for (int j = 0; j < num_frames; ++j) {
          const char *function_name =
              thread.GetFrameAtIndex(j).GetSymbol().GetName();
          // The function name is allowed to be null, all we care about here
          // is that some function was found.
          g_frame_functions.push(function_name ? string(function_name)
                                               : string());
        }
      }
    }
  }
}

void check_listener(SBDebugger &dbg) {
  bool got_function_name = false;
  string func_name = g_frame_functions.pop(got_function_name);

  if (got_function_name == false)
    throw Exception("Expected at least one frame function");
}
