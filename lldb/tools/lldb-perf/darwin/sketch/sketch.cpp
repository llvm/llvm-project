//===-- sketch.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CoreFoundation/CoreFoundation.h>

#include "lldb-perf/lib/Measurement.h"
#include "lldb-perf/lib/Metric.h"
#include "lldb-perf/lib/TestCase.h"
#include "lldb-perf/lib/Timer.h"
#include "lldb-perf/lib/Xcode.h"

#include <fstream>
#include <getopt.h>
#include <iostream>
#include <unistd.h>

using namespace lldb_perf;

static struct option g_long_options[] = {
    {"verbose", no_argument, NULL, 'v'},
    {"sketch", required_argument, NULL, 'c'},
    {"foobar", required_argument, NULL, 'f'},
    {"out-file", required_argument, NULL, 'o'},
    {NULL, 0, NULL, 0}};

class SketchTest : public TestCase {
public:
  SketchTest()
      : m_fetch_frames_measurement(
            [this]() -> void {
              Xcode::FetchFrames(GetProcess(), false, false);
            },
            "fetch-frames",
            "time to dump backtrace for every frame in every thread"),
        m_file_line_bp_measurement(
            [this](const char *file, uint32_t line) -> void {
              Xcode::CreateFileLineBreakpoint(GetTarget(), file, line);
            },
            "file-line-bkpt", "time to set a breakpoint given a file and line"),
        m_fetch_modules_measurement(
            [this]() -> void { Xcode::FetchModules(GetTarget()); },
            "fetch-modules", "time to get info for all modules in the process"),
        m_fetch_vars_measurement(
            [this](int depth) -> void {
              SBProcess process(GetProcess());
              auto threads_count = process.GetNumThreads();
              for (size_t thread_num = 0; thread_num < threads_count;
                   thread_num++) {
                SBThread thread(process.GetThreadAtIndex(thread_num));
                SBFrame frame(thread.GetFrameAtIndex(0));
                Xcode::FetchVariables(frame, depth, GetVerbose());
              }
            },
            "fetch-vars",
            "time to dump variables for the topmost frame in every thread"),
        m_run_expr_measurement(
            [this](SBFrame frame, const char *expr) -> void {
              SBValue value(
                  frame.EvaluateExpression(expr, lldb::eDynamicCanRunTarget));
              Xcode::FetchVariable(value, 0, GetVerbose());
            },
            "run-expr",
            "time to evaluate an expression and display the result") {
    m_app_path.clear();
    m_out_path.clear();
    m_doc_path.clear();
    m_print_help = false;
  }

  virtual ~SketchTest() {}

  virtual bool ParseOption(int short_option, const char *optarg) {
    switch (short_option) {
    case 0:
      return false;

    case -1:
      return false;

    case '?':
    case 'h':
      m_print_help = true;
      break;

    case 'v':
      SetVerbose(true);
      break;

    case 'c': {
      SBFileSpec file(optarg);
      if (file.Exists())
        SetExecutablePath(optarg);
      else
        fprintf(stderr, "error: file specified in --sketch (-c) option doesn't "
                        "exist: '%s'\n",
                optarg);
    } break;

    case 'f': {
      SBFileSpec file(optarg);
      if (file.Exists())
        SetDocumentPath(optarg);
      else
        fprintf(stderr, "error: file specified in --foobar (-f) option doesn't "
                        "exist: '%s'\n",
                optarg);
    } break;

    case 'o':
      SetResultFilePath(optarg);
      break;

    default:
      m_print_help = true;
      fprintf(stderr, "error: unrecognized option %c\n", short_option);
      break;
    }
    return true;
  }

  virtual struct option *GetLongOptions() { return g_long_options; }

  virtual bool Setup(int &argc, const char **&argv) {
    TestCase::Setup(argc, argv);
    bool error = false;

    if (GetExecutablePath() == NULL) {
      // --sketch is mandatory
      error = true;
      fprintf(stderr, "error: the '--sketch=PATH' option is mandatory\n");
    }

    if (GetDocumentPath() == NULL) {
      // --foobar is mandatory
      error = true;
      fprintf(stderr, "error: the '--foobar=PATH' option is mandatory\n");
    }

    if (error || GetPrintHelp()) {
      puts(R"(
                 NAME
                 lldb_perf_sketch -- a tool that measures LLDB peformance while debugging sketch.
                 
                 SYNOPSIS
                 lldb_perf_sketch --sketch=PATH --foobar=PATH [--out-file=PATH --verbose]
                 
                 DESCRIPTION
                 Runs a set of static timing and memory tasks against sketch and outputs results
                 to a plist file.
                 )");
    }

    if (error) {
      exit(1);
    }
    lldb::SBLaunchInfo launch_info = GetLaunchInfo();
    m_target = m_debugger.CreateTarget(m_app_path.c_str());
    m_file_line_bp_measurement("SKTDocument.m", 245);
    m_file_line_bp_measurement("SKTDocument.m", 283);
    m_file_line_bp_measurement("SKTText.m", 326);
    return Launch(launch_info);
  }

  lldb::SBLaunchInfo GetLaunchInfo() {
    const char *file_arg = m_doc_path.c_str();
    const char *persist_arg = "-ApplePersistenceIgnoreState";
    const char *persist_skip = "YES";
    const char *empty = nullptr;
    const char *args[] = {file_arg, persist_arg, persist_skip, empty};
    return SBLaunchInfo(args);
  }

  void DoTest() {
    m_fetch_frames_measurement();
    m_fetch_modules_measurement();
    m_fetch_vars_measurement(1);
  }

  virtual void TestStep(int counter, ActionWanted &next_action) {
    static int launch = 1;
    switch (counter % 10) {
    case 0: {
      DoTest();
      if (counter == 0)
        m_file_line_bp_measurement("SKTDocument.m", 254);
      next_action.Continue();
    } break;

    case 1: {
      DoTest();
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0), "properties");
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0),
                             "[properties description]");
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0), "typeName");
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0), "data");
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0), "[data description]");
      next_action.Continue();
    } break;

    case 2: {
      DoTest();
      next_action.Continue();
    } break;

    case 3: {
      DoTest();
      next_action.StepOver(m_thread);
    } break;

    case 4: {
      DoTest();
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0), "layoutManager");
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0), "contents");
      next_action.StepOver(m_thread);
    } break;

    case 5: {
      DoTest();
      next_action.StepOver(m_thread);
    } break;

    case 6: {
      DoTest();
      next_action.StepOver(m_thread);
    } break;

    case 7: {
      DoTest();
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0), "@\"an NSString\"");
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0),
                             "[(id)@\"an NSString\" description]");
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0), "@[@1,@2,@3]");
      next_action.StepOut(m_thread);
    } break;

    case 8: {
      DoTest();
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0),
                             "[graphics description]");
      m_run_expr_measurement(m_thread.GetFrameAtIndex(0),
                             "[selectionIndexes description]");
      m_run_expr_measurement(
          m_thread.GetFrameAtIndex(0),
          "(BOOL)NSIntersectsRect(rect, graphicDrawingBounds)");
    }
      next_action.CallNext();
      break;
    case 9:
      if (++launch < 10)
        next_action.Relaunch(GetLaunchInfo());
      else
        next_action.Kill();
      break;

    default: { next_action.Kill(); } break;
    }
  }

  virtual void WriteResults(Results &results) {
    m_fetch_frames_measurement.WriteAverageAndStandardDeviation(results);
    m_file_line_bp_measurement.WriteAverageAndStandardDeviation(results);
    m_fetch_modules_measurement.WriteAverageAndStandardDeviation(results);
    m_fetch_vars_measurement.WriteAverageAndStandardDeviation(results);
    m_run_expr_measurement.WriteAverageAndStandardDeviation(results);
    results.Write(GetResultFilePath());
  }

  void SetExecutablePath(const char *str) {
    if (str)
      m_app_path.assign(str);
  }

  const char *GetExecutablePath() {
    if (m_app_path.empty())
      return NULL;
    return m_app_path.c_str();
  }

  void SetDocumentPath(const char *str) {
    if (str)
      m_doc_path.assign(str);
  }

  const char *GetDocumentPath() {
    if (m_doc_path.empty())
      return NULL;
    return m_doc_path.c_str();
  }

  void SetResultFilePath(const char *str) {
    if (str)
      m_out_path.assign(str);
  }

  const char *GetResultFilePath() {
    if (m_out_path.empty())
      return "/dev/stdout";
    return m_out_path.c_str();
  }

  bool GetPrintHelp() { return m_print_help; }

private:
  Measurement<lldb_perf::TimeGauge, std::function<void()>>
      m_fetch_frames_measurement;
  Measurement<lldb_perf::TimeGauge, std::function<void(const char *, uint32_t)>>
      m_file_line_bp_measurement;
  Measurement<lldb_perf::TimeGauge, std::function<void()>>
      m_fetch_modules_measurement;
  Measurement<lldb_perf::TimeGauge, std::function<void(int)>>
      m_fetch_vars_measurement;
  Measurement<lldb_perf::TimeGauge, std::function<void(SBFrame, const char *)>>
      m_run_expr_measurement;

  std::string m_app_path;
  std::string m_doc_path;
  std::string m_out_path;
  bool m_print_help;
};

int main(int argc, const char *argv[]) {
  SketchTest test;
  return TestCase::Run(test, argc, argv);
}
