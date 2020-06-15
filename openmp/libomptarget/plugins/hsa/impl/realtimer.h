/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/

#ifndef SRC_RUNTIME_INCLUDE_REALTIMER_H_
#define SRC_RUNTIME_INCLUDE_REALTIMER_H_

#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <string>
#include "rt.h"
#ifdef DEBUG
#define USE_PROFILE
#endif

namespace core {

/// \class RealTimer RealTimer.h "core/RealTimer.h"
///
/// \brief Compute elapsed_ time.
class RealTimer {
 public:
  /// Constructor.  If desc is provides, it will be output along with
  /// the RealTimer state when the RealTimer object is destroyed.
  explicit RealTimer(const std::string& desc = "");
  ~RealTimer();

  /// Start the timer.
  void start();

  /// Stop the timer.
  void stop();

  /// Reset all counters to 0.
  void reset();

  /// Return the total time during which the timer was running.
  double elapsed() const;

  /// Number of times stop() is called.
  int count() const;

  void pause();
  void resume();

  /// print the current state of the counter to the stream.
  void print(std::ostream& o) const;

  /// print the current state of the counter to the stream with custom
  /// description
  /// string
  void bufPrint(std::ostream& o, const std::string& str) const;

  /// Return the system time.
  double currentTime() const;

  /// Return true if the timer is running (started), otherwise return false
  /// (stopped).
  bool is_running();

 private:
  std::string desc_;
  int time_offset_;
  double start_time_;
  double elapsed_;
  bool is_running_;
  int count_;
  int profile_mode_;
};

/// Timer stream insertion operator
inline std::ostream& operator<<(std::ostream& os, const RealTimer& t) {
  t.print(os);
  return os;
}

inline RealTimer::RealTimer(const std::string& desc)
    : desc_(desc),
      time_offset_(0),
      start_time_(0),
      elapsed_(0.0),
      is_running_(false),
      count_(0) {
  time_offset_ = static_cast<int>(currentTime());
  profile_mode_ = core::Runtime::getInstance().getProfileMode();
}

inline RealTimer::~RealTimer() {
#if 0
  if (desc_ != "")
    std::cout << "Timer " << desc_ << std::endl;
#endif
}

inline void RealTimer::start() {
#ifdef USE_PROFILE
  if (profile_mode_) {
    static std::string functionName("RealTimer::start()");
    if (is_running_ == true) {
      std::cout << functionName << ": Warning: Timer " << desc_
                << " has already been started." << std::endl;
    } else {
      start_time_ = currentTime();
      is_running_ = true;
    }
  }
#endif
}

inline void RealTimer::stop() {
#ifdef USE_PROFILE
  if (profile_mode_) {
    static std::string functionName("RealTimer::stop()");
    if (is_running_ == false) {
      std::cout << functionName << ": Warning: Timer " << desc_
                << " has already been stopped." << std::endl;
    } else {
      elapsed_ += currentTime() - start_time_;
      is_running_ = false;
      count_++;
    }
  }
#endif
}

inline bool RealTimer::is_running() { return is_running_; }

inline void RealTimer::reset() {
  elapsed_ = 0.0;
  start_time_ = 0;
  count_ = 0;
  time_offset_ = 0;
  time_offset_ = static_cast<int>(currentTime());
}

inline double RealTimer::elapsed() const {
  static std::string functionName("inline double Timer::elapsed() const");
  if (is_running_ == true) {
    std::cout << functionName << ": Warning: Timer " << desc_
              << " is still running." << std::endl;
    return elapsed_ + currentTime() - start_time_;
  }
  return elapsed_;
}

inline int RealTimer::count() const { return count_; }

inline double RealTimer::currentTime() const {
#if 1
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<double>(ts.tv_sec - time_offset_) +
         static_cast<double>(ts.tv_nsec) * 1e-9;
#else
  timeval tv;
  gettimeofday(&tv, NULL);
  return static_cast<double>(tv.tv_sec - time_offset_) +
         static_cast<double>(tv.tv_usec) * 1e-6;
#endif
}

inline void RealTimer::print(std::ostream& o) const {
#ifdef USE_PROFILE
  if (profile_mode_) {
    o << desc_ << ": " << elapsed_ * 1000 << " msecs " << count_ << " times";
    if (count_ > 1) o << " " << (elapsed_ / count_) * 1000 << " msecs each\n";
  }
#endif
}

inline void RealTimer::bufPrint(std::ostream& o, const std::string& str) const {
#ifdef USE_PROFILE
  if (profile_mode_) {
    o << str << ": " << elapsed_ * 1000 << " msecs " << count_ << " times";
    if (count_ > 1) o << " " << (elapsed_ / count_) * 1000 << " msecs each\n";
  }
#endif
}
}  // namespace core
#endif  //  SRC_RUNTIME_INCLUDE_REALTIMER_H_
