#ifndef BENCHMARK_LOG_H_
#define BENCHMARK_LOG_H_

#include <iostream>
#include <ostream>

namespace benchmark {
namespace internal {

typedef std::basic_ostream<char>&(EndLType)(std::basic_ostream<char>&);

class LogType {
  friend LogType& GetNullLogInstance();
  friend LogType& GetErrorLogInstance();

  // FIXME: Add locking to output.
  template <class Tp>
  friend LogType& operator<<(LogType&, Tp const&);
  friend LogType& operator<<(LogType&, EndLType*);

 private:
  LogType(std::ostream* out) : out_(out) {}
  std::ostream* out_;

  // NOTE: we could use BENCHMARK_DISALLOW_COPY_AND_ASSIGN but we shouldn't have
  // a dependency on benchmark.h from here.
  LogType(const LogType&) = delete;
  LogType& operator=(const LogType&) = delete;
};

template <class Tp>
LogType& operator<<(LogType& log, Tp const& value) {
  if (log.out_) {
    *log.out_ << value;
  }
  return log;
}

inline LogType& operator<<(LogType& log, EndLType* m) {
  if (log.out_) {
    *log.out_ << m;
  }
  return log;
}

inline int& LogLevel() {
  static int log_level = 0;
  return log_level;
}

inline LogType& GetNullLogInstance() {
  static LogType null_log(static_cast<std::ostream*>(nullptr));
  return null_log;
}

inline LogType& GetErrorLogInstance() {
  static LogType error_log(&std::clog);
  return error_log;
}

inline LogType& GetLogInstanceForLevel(int level) {
  if (level <= LogLevel()) {
    return GetErrorLogInstance();
  }
  return GetNullLogInstance();
}

}  // end namespace internal
}  // end namespace benchmark

// clang-format off
#define BM_VLOG(x)                                                               \
  (::benchmark::internal::GetLogInstanceForLevel(x) << "-- LOG(" << x << "):" \
                                                                         " ")
// clang-format on
#endif
