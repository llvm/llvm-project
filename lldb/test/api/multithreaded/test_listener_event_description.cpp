
// LLDB C++ API Test: verify the event description that is received by an
// SBListener object registered with a process with a breakpoint.

#include <atomic>
#include <array>
#include <iostream>
#include <string>
#include <thread>

#include "lldb-headers.h"

#include "common.h"

using namespace lldb;
using namespace std;

// listener thread control
extern atomic<bool> g_done; 

multithreaded_queue<string> g_event_descriptions;

extern SBListener g_listener;

void listener_func() {
  while (!g_done) {
    SBEvent event;
    bool got_event = g_listener.WaitForEvent(1, event);
    if (got_event) {
      if (!event.IsValid())
        throw Exception("event is not valid in listener thread");

      SBStream description;
      event.GetDescription(description);
      string str(description.GetData());
      g_event_descriptions.push(str);
    }
  }
}

void check_listener(SBDebugger &dbg) {
  array<string, 2> expected_states = {"running", "stopped"};
  for(string & state : expected_states) {
    bool got_description = false;
    string desc = g_event_descriptions.pop(5, got_description);

    if (!got_description)
      throw Exception("Did not get expected event description");


    if (desc.find("state-changed") == desc.npos)
      throw Exception("Event description incorrect: missing 'state-changed'");

    string state_search_str = "state = " + state;
    if (desc.find(state_search_str) == desc.npos)
      throw Exception("Event description incorrect: expected state "
                      + state
                      + " but desc was "
                      + desc);

    if (desc.find("pid = ") == desc.npos)
      throw Exception("Event description incorrect: missing process pid");
  }
}
