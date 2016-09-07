//===-- TestCase.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestCase.h"
#include "Results.h"
#include "Xcode.h"

using namespace lldb_perf;

TestCase::TestCase()
    : m_debugger(), m_target(), m_process(), m_thread(), m_listener(),
      m_verbose(false), m_step(0) {
  SBDebugger::Initialize();
  SBHostOS::ThreadCreated("<lldb-tester.app.main>");
  m_debugger = SBDebugger::Create(false);
  m_listener = m_debugger.GetListener();
  m_listener.StartListeningForEventClass(
      m_debugger, SBProcess::GetBroadcasterClass(),
      SBProcess::eBroadcastBitStateChanged | SBProcess::eBroadcastBitInterrupt);
}

static std::string GetShortOptionString(struct option *long_options) {
  std::string option_string;
  for (int i = 0; long_options[i].name != NULL; ++i) {
    if (long_options[i].flag == NULL) {
      option_string.push_back((char)long_options[i].val);
      switch (long_options[i].has_arg) {
      default:
      case no_argument:
        break;
      case required_argument:
        option_string.push_back(':');
        break;
      case optional_argument:
        option_string.append(2, ':');
        break;
      }
    }
  }
  return option_string;
}

bool TestCase::Setup(int &argc, const char **&argv) {
  bool done = false;

  struct option *long_options = GetLongOptions();

  if (long_options) {
    std::string short_option_string(GetShortOptionString(long_options));

#if __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif
    while (!done) {
      int long_options_index = -1;
      const int short_option = ::getopt_long_only(
          argc, const_cast<char **>(argv), short_option_string.c_str(),
          long_options, &long_options_index);

      switch (short_option) {
      case 0:
        // Already handled
        break;

      case -1:
        done = true;
        break;

      default:
        done = !ParseOption(short_option, optarg);
        break;
      }
    }
    argc -= optind;
    argv += optind;
  }

  return false;
}

bool TestCase::Launch(lldb::SBLaunchInfo &launch_info) {
  lldb::SBError error;
  m_process = m_target.Launch(launch_info, error);
  if (!error.Success())
    fprintf(stderr, "error: %s\n", error.GetCString());
  if (m_process.IsValid())
    return true;
  return false;
}

bool TestCase::Launch(std::initializer_list<const char *> args) {
  std::vector<const char *> args_vect(args);
  args_vect.push_back(NULL);
  lldb::SBLaunchInfo launch_info((const char **)&args_vect[0]);
  return Launch(launch_info);
}

void TestCase::SetVerbose(bool b) { m_verbose = b; }

bool TestCase::GetVerbose() { return m_verbose; }

void TestCase::Loop() {
  while (true) {
    bool call_test_step = false;
    if (m_process.IsValid()) {
      SBEvent evt;
      m_listener.WaitForEvent(UINT32_MAX, evt);
      StateType state = SBProcess::GetStateFromEvent(evt);
      if (m_verbose)
        printf("event = %s\n", SBDebugger::StateAsCString(state));
      if (SBProcess::GetRestartedFromEvent(evt)) {
        if (m_verbose) {
          const uint32_t num_threads = m_process.GetNumThreads();
          for (auto thread_index = 0; thread_index < num_threads;
               thread_index++) {
            SBThread thread(m_process.GetThreadAtIndex(thread_index));
            SBFrame frame(thread.GetFrameAtIndex(0));
            SBStream strm;
            strm.RedirectToFileHandle(stdout, false);
            frame.GetDescription(strm);
          }
          puts("restarted");
        }
        call_test_step = false;
      } else {
        switch (state) {
        case eStateInvalid:
        case eStateDetached:
        case eStateCrashed:
        case eStateUnloaded:
          break;
        case eStateExited:
          return;
        case eStateConnected:
        case eStateAttaching:
        case eStateLaunching:
        case eStateRunning:
        case eStateStepping:
          call_test_step = false;
          break;

        case eStateStopped:
        case eStateSuspended: {
          call_test_step = true;
          bool fatal = false;
          bool selected_thread = false;
          const uint32_t num_threads = m_process.GetNumThreads();
          for (auto thread_index = 0; thread_index < num_threads;
               thread_index++) {
            SBThread thread(m_process.GetThreadAtIndex(thread_index));
            SBFrame frame(thread.GetFrameAtIndex(0));
            SBStream strm;
            strm.RedirectToFileHandle(stdout, false);
            frame.GetDescription(strm);
            bool select_thread = false;
            StopReason stop_reason = thread.GetStopReason();
            if (m_verbose)
              printf("tid = 0x%llx pc = 0x%llx ", thread.GetThreadID(),
                     frame.GetPC());
            switch (stop_reason) {
            case eStopReasonNone:
              if (m_verbose)
                printf("none\n");
              break;

            case eStopReasonTrace:
              select_thread = true;
              if (m_verbose)
                printf("trace\n");
              break;

            case eStopReasonPlanComplete:
              select_thread = true;
              if (m_verbose)
                printf("plan complete\n");
              break;
            case eStopReasonThreadExiting:
              if (m_verbose)
                printf("thread exiting\n");
              break;
            case eStopReasonExec:
              if (m_verbose)
                printf("exec\n");
              break;
            case eStopReasonInvalid:
              if (m_verbose)
                printf("invalid\n");
              break;
            case eStopReasonException:
              select_thread = true;
              if (m_verbose)
                printf("exception\n");
              fatal = true;
              break;
            case eStopReasonBreakpoint:
              select_thread = true;
              if (m_verbose)
                printf("breakpoint id = %lld.%lld\n",
                       thread.GetStopReasonDataAtIndex(0),
                       thread.GetStopReasonDataAtIndex(1));
              break;
            case eStopReasonWatchpoint:
              select_thread = true;
              if (m_verbose)
                printf("watchpoint id = %lld\n",
                       thread.GetStopReasonDataAtIndex(0));
              break;
            case eStopReasonSignal:
              select_thread = true;
              if (m_verbose)
                printf("signal %d\n", (int)thread.GetStopReasonDataAtIndex(0));
              break;
            }
            if (select_thread && !selected_thread) {
              m_thread = thread;
              selected_thread = m_process.SetSelectedThread(thread);
            }
          }
          if (fatal) {
            if (m_verbose)
              Xcode::RunCommand(m_debugger, "bt all", true);
            exit(1);
          }
        } break;
        }
      }
    } else {
      call_test_step = true;
    }

    if (call_test_step) {
    do_the_call:
      if (m_verbose)
        printf("RUNNING STEP %d\n", m_step);
      ActionWanted action;
      TestStep(m_step, action);
      m_step++;
      SBError err;
      switch (action.type) {
      case ActionWanted::Type::eNone:
        // Just exit and wait for the next event
        break;
      case ActionWanted::Type::eContinue:
        err = m_process.Continue();
        break;
      case ActionWanted::Type::eStepOut:
        if (action.thread.IsValid() == false) {
          if (m_verbose) {
            Xcode::RunCommand(m_debugger, "bt all", true);
            printf("error: invalid thread for step out on step %d\n", m_step);
          }
          exit(501);
        }
        m_process.SetSelectedThread(action.thread);
        action.thread.StepOut();
        break;
      case ActionWanted::Type::eStepOver:
        if (action.thread.IsValid() == false) {
          if (m_verbose) {
            Xcode::RunCommand(m_debugger, "bt all", true);
            printf("error: invalid thread for step over %d\n", m_step);
          }
          exit(500);
        }
        m_process.SetSelectedThread(action.thread);
        action.thread.StepOver();
        break;
      case ActionWanted::Type::eRelaunch:
        if (m_process.IsValid()) {
          m_process.Kill();
          m_process.Clear();
        }
        Launch(action.launch_info);
        break;
      case ActionWanted::Type::eKill:
        if (m_verbose)
          printf("kill\n");
        m_process.Kill();
        return;
      case ActionWanted::Type::eCallNext:
        goto do_the_call;
        break;
      }
    }
  }

  if (GetVerbose())
    printf("I am gonna die at step %d\n", m_step);
}

int TestCase::Run(TestCase &test, int argc, const char **argv) {
  if (test.Setup(argc, argv)) {
    test.Loop();
    Results results;
    test.WriteResults(results);
    return RUN_SUCCESS;
  } else
    return RUN_SETUP_ERROR;
}
