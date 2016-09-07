//===-- TestCase.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver__TestCase__
#define __PerfTestDriver__TestCase__

#include "Measurement.h"
#include "lldb/API/LLDB.h"
#include <getopt.h>

namespace lldb_perf {

class Results;

class TestCase {
public:
  TestCase();

  struct ActionWanted {
    enum class Type {
      eStepOver,
      eContinue,
      eStepOut,
      eRelaunch,
      eCallNext,
      eNone,
      eKill
    } type;
    lldb::SBThread thread;
    lldb::SBLaunchInfo launch_info;

    ActionWanted() : type(Type::eContinue), thread(), launch_info(NULL) {}

    void None() {
      type = Type::eNone;
      thread = lldb::SBThread();
    }

    void Continue() {
      type = Type::eContinue;
      thread = lldb::SBThread();
    }

    void StepOver(lldb::SBThread t) {
      type = Type::eStepOver;
      thread = t;
    }

    void StepOut(lldb::SBThread t) {
      type = Type::eStepOut;
      thread = t;
    }

    void Relaunch(lldb::SBLaunchInfo l) {
      type = Type::eRelaunch;
      thread = lldb::SBThread();
      launch_info = l;
    }

    void Kill() {
      type = Type::eKill;
      thread = lldb::SBThread();
    }

    void CallNext() {
      type = Type::eCallNext;
      thread = lldb::SBThread();
    }
  };

  virtual ~TestCase() {}

  virtual bool Setup(int &argc, const char **&argv);

  virtual void TestStep(int counter, ActionWanted &next_action) = 0;

  bool Launch(lldb::SBLaunchInfo &launch_info);

  bool Launch(std::initializer_list<const char *> args = {});

  void Loop();

  void SetVerbose(bool);

  bool GetVerbose();

  virtual void WriteResults(Results &results) = 0;

  template <typename G, typename A>
  Measurement<G, A> CreateMeasurement(A a, const char *name = NULL,
                                      const char *description = NULL) {
    return Measurement<G, A>(a, name, description);
  }

  template <typename A>
  TimeMeasurement<A> CreateTimeMeasurement(A a, const char *name = NULL,
                                           const char *description = NULL) {
    return TimeMeasurement<A>(a, name, description);
  }

  template <typename A>
  MemoryMeasurement<A> CreateMemoryMeasurement(A a, const char *name = NULL,
                                               const char *description = NULL) {
    return MemoryMeasurement<A>(a, name, description);
  }

  static int Run(TestCase &test, int argc, const char **argv);

  virtual bool ParseOption(int short_option, const char *optarg) {
    return false;
  }

  virtual struct option *GetLongOptions() { return NULL; }

  lldb::SBDebugger &GetDebugger() { return m_debugger; }

  lldb::SBTarget &GetTarget() { return m_target; }

  lldb::SBProcess &GetProcess() { return m_process; }

  lldb::SBThread &GetThread() { return m_thread; }

  int GetStep() { return m_step; }

  static const int RUN_SUCCESS = 0;
  static const int RUN_SETUP_ERROR = 100;

protected:
  lldb::SBDebugger m_debugger;
  lldb::SBTarget m_target;
  lldb::SBProcess m_process;
  lldb::SBThread m_thread;
  lldb::SBListener m_listener;
  bool m_verbose;
  int m_step;
};
}

#endif /* defined(__PerfTestDriver__TestCase__) */
