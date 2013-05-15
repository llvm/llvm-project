
// LLDB C++ API Test: verify the event description as obtained by calling
// SBEvent::GetCStringFromEvent that is received by an
// SBListener object registered with a process with a breakpoint.

#include <atomic>
#include <iostream>
#include <string>
#include <thread>

#include "lldb-headers.h"

#include "common.h"

using namespace lldb;
using namespace std;

// listener thread control
extern atomic<bool> g_done;

multithreaded_queue<string> g_thread_descriptions;
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
      SBStream description;

      for (int i = 0; i < process.GetNumThreads(); ++i) {
        // send each thread description
        description.Clear();
        SBThread thread = process.GetThreadAtIndex(i);
        thread.GetDescription(description);
        g_thread_descriptions.push(description.GetData());

        // send each frame function name
        uint32_t num_frames = thread.GetNumFrames();
        for(int j = 0; j < num_frames; ++j) {
          const char* function_name = thread.GetFrameAtIndex(j).GetFunction().GetName();
          if (function_name)
            g_frame_functions.push(function_name);
        }
      }
    }
  }
}

void check_listener(SBDebugger &dbg) {
  // check thread description
  bool got_description = false;
  string desc = g_thread_descriptions.pop(5, got_description);
  if (!got_description)
    throw Exception("Expected at least one thread description string");

  // check at least one frame has a function name
  desc = g_frame_functions.pop(5, got_description);
  if (!got_description)
    throw Exception("Expected at least one frame function name string");
}
