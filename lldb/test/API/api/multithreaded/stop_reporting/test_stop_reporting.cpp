// LLDB C++ API Test: verify that while a stop is being handled, it stays
// private to the thread handling it. That thread sees the process stopped and
// can run expressions, but every other thread sees it running and can neither
// inspect, resume nor halt it.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include <mutex>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"

#include "common.h"

#if !defined(PATH_MAX)
#define PATH_MAX 4096
#endif

using namespace lldb;

// Wait for the next state-changed event, skipping the inferior's stdout and
// stderr events, which also carry the process state.
bool wait_for_state_event(SBListener &listener, SBEvent &event) {
  while (listener.WaitForEvent(10, event))
    if (event.GetType() & SBProcess::eBroadcastBitStateChanged)
      return true;
  return false;
}

bool continue_to_breakpoint(SBDebugger dbg, SBProcess process, SBThread thread,
                            lldb::break_id_t bkpt_id, std::string &reason) {
  // And wait for it to start running and stop again:
  SBEvent event;
  SBListener proc_listener = dbg.GetListener();
  process.Continue();

  if (!wait_for_state_event(proc_listener, event)) {
    reason = "Didn't get an event after Continue";
    return false;
  }

  if (SBProcess::GetStateFromEvent(event) != lldb::eStateRunning) {
    reason = "Didn't get a running event after Continue";
    return false;
  }

  if (!wait_for_state_event(proc_listener, event)) {
    reason = "Didn't get an event after run event";
    return false;
  }

  if (SBProcess::GetStateFromEvent(event) != lldb::eStateStopped) {
    reason = "Didn't get a stop event after run event";
    return false;
  }

  if (!thread.IsValid()) {
    reason = "Traced thread is not valid.";
    return false;
  }

  if (thread.GetStopReason() != lldb::eStopReasonBreakpoint) {
    reason = "Tracked thread didn't stop for a breakpoint reason.";
    return false;
  }

  if (thread.GetStopReasonDataAtIndex(0) != bkpt_id) {
    reason = "Tracked thread didn't stop at the right breakpoint";
    return false;
  }
  return true;
}

bool get_command_output(SBCommandInterpreter &interp, const char *command,
                        std::string &output) {
  SBCommandReturnObject result;
  interp.HandleCommand(command, result);
  if (!result.Succeeded())
    return false;
  const char *result_output = result.GetOutput();
  output = result_output ? result_output : "";
  return true;
}

void test(SBDebugger &dbg, std::vector<std::string> args) {
  // Between a stop event being received and the completion of its
  // PerformAction, the public state of the process stays "running". The
  // PerformAction can call back into the command line or Python, so IT has to
  // see the process stopped. But no other thread can, or it could resume the
  // process and destroy objects that the PerformAction is still using.
  //
  // To test this, the stop runs a breakpoint callback that stalls, calling a
  // function in the target until it is told to stop. Meanwhile another thread
  // repeatedly does things that require the process to be stopped, and checks
  // that they all fail.

  dbg.SetAsync(true);

  const char *exe_name = args.at(0).c_str();
  SBTarget target = dbg.CreateTarget(exe_name);
  if (!target.IsValid())
    throw Exception("invalid target");
  SBFileSpec main_spec("stop_reporting.cpp");

  SBBreakpoint start_bkpt = target.BreakpointCreateBySourceRegex(
      "Set a breakpoint in main", main_spec);
  if (!start_bkpt.IsValid() || start_bkpt.GetNumLocations() == 0)
    throw Exception("invalid breakpoint");

  SBCommandInterpreter interp = dbg.GetCommandInterpreter();
  SBCommandReturnObject result;

  // The test sets LLDB_TEST_SOURCE_DIR to the directory containing this
  // test's support files.
  const char *source_dir = getenv("LLDB_TEST_SOURCE_DIR");
  if (!source_dir)
    throw Exception("LLDB_TEST_SOURCE_DIR is not set");
  SBFileSpec script_spec(source_dir);
  script_spec.AppendPathComponent("stop_reporting_callbacks.py");
  char path[PATH_MAX];
  script_spec.GetPath(path, PATH_MAX);

  std::string import_command("command script import ");
  import_command.append(path);
  interp.HandleCommand(import_command.c_str(), result);
  if (!result.Succeeded())
    throw Exception(std::string("Couldn't import ") + path);

  SBProcess process = target.LaunchSimple(nullptr, nullptr, nullptr);

  if (!process.IsValid())
    throw Exception(std::string("Couldn't launch process."));

  if (process.GetState() != lldb::eStateStopped)
    throw Exception(std::string("Process not stopped after LaunchSimple."));

  SBThread main_thread = process.GetSelectedThread();

  // Continue from the stop point to our initial breakpoint:
  std::string error_str;
  if (!continue_to_breakpoint(dbg, process, main_thread, start_bkpt.GetID(),
                              error_str))
    throw Exception(error_str);

  // Now add our test breakpoint, and then add a callback to it:
  SBBreakpoint breakpoint =
      target.BreakpointCreateBySourceRegex("Set a breakpoint here", main_spec);
  if (!breakpoint.IsValid() || breakpoint.GetNumLocations() == 0)
    throw Exception("invalid breakpoint in bkpt_function");
  SBError error;
  SBStructuredData extra_args;
  error = breakpoint.SetScriptCallbackFunction(
      "stop_reporting_callbacks.slow_breakpoint_callback", extra_args);
  if (error.Fail())
    throw Exception(std::string("Failed adding breakpoint command"));

  // The commands we use to get info in and out of the callback:
  interp.HandleCommand("command script add -f "
                       "stop_reporting_callbacks.handle_signal_var "
                       "handle_signal_var",
                       result);
  if (!result.Succeeded())
    throw Exception(std::string("Couldn't add handle_signal_var command."));
  interp.HandleCommand("command script add -f "
                       "stop_reporting_callbacks.handle_failed_calls "
                       "handle_failed_calls",
                       result);
  if (!result.Succeeded())
    throw Exception(std::string("Couldn't add handle_failed_calls command."));

  // Spawn the thread that waits, using `handle_signal_var`, for the callback
  // to start running, and then checks that the process looks running to it.
  // Finally it flips the variable so the callback can return.
  std::string err_str = "Test thread didn't finish";
  int num_tries = 10;
  std::thread test_running_thread([&interp, &process, &err_str, &num_tries]() {
    // Whatever happens, let the callback return. Otherwise the main thread
    // never gets its stop event back and the test hangs instead of failing.
    struct ReleaseCallback {
      SBCommandInterpreter &interp;
      ~ReleaseCallback() {
        SBCommandReturnObject result;
        interp.HandleCommand("handle_signal_var False", result);
      }
    } release_callback{interp};

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (true) {
      std::string output;
      if (!get_command_output(interp, "handle_signal_var", output)) {
        err_str = "Error calling handle_signal_var";
        return;
      }
      if (output == "True")
        break;
      if (output != "False") {
        err_str = "Unexpected result from handle_signal_var";
        return;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        err_str = "The breakpoint callback never ran";
        return;
      }
      usleep(10);
    }

    // The callback is now running, so everything below must fail.
    if (process.GetState() != lldb::eStateRunning) {
      err_str = "Process didn't look running while the callback ran";
      return;
    }

    for (; num_tries > 0; num_tries--) {
      SBCommandReturnObject expr_result;
      interp.HandleCommand("expr call_me_function(1)", expr_result);
      if (expr_result.Succeeded()) {
        err_str = "Expression succeeded while the callback ran";
        return;
      }
      usleep(10);
    }

    if (process.Continue().Success()) {
      err_str = "Continue succeeded while the callback ran";
      return;
    }

    if (process.Stop().Success()) {
      err_str = "Stop succeeded while the callback ran";
      return;
    }

    err_str.clear();
  });

  // Okay, our test thread is running, so continue the process:
  bool stopped = continue_to_breakpoint(dbg, process, main_thread,
                                        breakpoint.GetID(), error_str);

  test_running_thread.join();
  if (!err_str.empty())
    throw Exception(err_str);
  if (!stopped)
    throw Exception(error_str);

  // We should have gotten 10 tries at the expression, without being
  // interrupted.
  if (num_tries != 0)
    throw Exception(std::string("Didn't get to run 10 times"));

  // And the callback's own expressions must all have worked.
  std::string failed_calls;
  if (!get_command_output(interp, "handle_failed_calls", failed_calls))
    throw Exception(std::string("Error calling handle_failed_calls"));
  if (failed_calls != "0")
    throw Exception("The callback's expressions failed " + failed_calls +
                    " times");

  // Now run to exit - make sure we didn't stall the process.
  breakpoint.SetEnabled(false);
  SBEvent event;
  SBListener proc_listener = dbg.GetListener();
  process.Continue();

  if (!wait_for_state_event(proc_listener, event))
    throw Exception(std::string("Didn't get an event after Continue"));

  if (SBProcess::GetStateFromEvent(event) != lldb::eStateRunning)
    throw Exception(std::string("Didn't get an running event after Continue"));

  if (!wait_for_state_event(proc_listener, event))
    throw Exception(std::string("Didn't get an event after running"));

  StateType final_state = SBProcess::GetStateFromEvent(event);
  if (final_state != lldb::eStateExited)
    throw Exception(std::string("Didn't get an exited event after running, "
                                "got: ") +
                    SBDebugger::StateAsCString(final_state));
}
