// This is part of a temporary patch, to collect start-up time pieces in LLDB.
#include "lldb/Utility/CarolinesTimers.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <string>

#include <stdio.h>


namespace lldb_private {

std::string event_name[(int)CarolineTimerEvent::eCarolineLastItem+1] = {
  "Start Frame Var", "End Frame Var", "Start ExprEval",
  "End ExprEval", "Other"
};

static double CalculateTimeDiff(timespec start, timespec end) {
  double start_time;
  double end_time;
  double diff = 0.0;

  start_time = start.tv_sec + ((double)start.tv_nsec / 1000000000);
  end_time = end.tv_sec + ((double)end.tv_nsec / 1000000000);
  diff =  end_time - start_time;

  return diff;
}

static void CarolineGenerateTimeReport(timespec *time_stamps,
                                       const std::string& expr) {

  double FrameVar = CalculateTimeDiff(time_stamps[eCarolineStartFrameVar],
                                       time_stamps[eCarolineEndFrameVar]);

  double ExprEval = CalculateTimeDiff(time_stamps[eCarolineStartExprEval],
                                       time_stamps[eCarolineEndExprEval]);

  std::filesystem::path file_path =
      "/usr/local/google/home/cmtice/dil-report.txt";
  if (std::filesystem::exists(file_path)) {
    FILE *f = fopen("/usr/local/google/home/cmtice/dil-report.txt", "a");
    if (f) {
      fprintf(f, "%s,%f,%f\n", expr.c_str(), FrameVar, ExprEval);
      fclose(f);
    }
  } else {
    FILE *f = fopen("/usr/local/google/home/cmtice/dil-report.txt", "w");
    if (f) {
      fprintf(f, "expr,DIL_time,ExprEval_time\n");
      fprintf(f, "%s,%f,%f\n", expr.c_str(), FrameVar, ExprEval);
      fclose(f);
    }
  }
}

void CarolineTimeStamp(CarolineTimerEvent event_kind,
                       const std::string &in_expr,
                       timespec *timespec_ptr)
{
  timespec ts;
  static timespec time_stamps[eCarolineLastItem+1];
  static std::string expr;
  static bool done = false;

  // if (event_kind == eCarolineStartFrameVar && timespec_ptr) {
  //ts.tv_sec = timespec_ptr->tv_sec;
  //ts.tv_nsec = timespec_ptr->tv_nsec;
    //} else {
    timespec_get(&ts, TIME_UTC);
    //}
  time_stamps[event_kind] = ts;

  bool all_ok = true;
  if (expr.empty()) {
    if (event_kind == eCarolineStartFrameVar)
      expr = std::move(in_expr);
    else
      all_ok = false;
  } else if (expr != in_expr)
    all_ok = false;

  assert(all_ok && "Something is wrong in Carolines Timers!");

  if (event_kind == eCarolineEndExprEval && !done) {
    CarolineGenerateTimeReport(time_stamps, expr);
    //done = true;
    expr.clear();
  }
}

} // namespace lldb_private
