#ifndef LLDB_TEST_API_COMMON_H
#define LLDB_TEST_API_COMMON_H

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>

#include <unistd.h>

/// Simple exception class with a message
struct Exception : public std::exception
{
  std::string s;
  Exception(std::string ss) : s(ss) {}
  virtual ~Exception() throw () { }
  const char* what() const throw() { return s.c_str(); }
};

/// Throws an Exception with the given message if 'condition' is false.
inline void expect(bool condition, const std::string &message) {
  if (!condition)
    throw Exception(message);
}

/// Throws an Exception describing 'what' if 'actual' is null or doesn't equal
/// the string 'expected'.
inline void expect_string(const char *actual, const char *expected,
                          const std::string &what) {
  if (!actual)
    throw Exception(what + ": expected '" + expected + "' but got no string");
  if (std::strcmp(actual, expected) != 0)
    throw Exception(what + ": expected '" + expected + "' but got '" + actual +
                    "'");
}

/// Throws an Exception describing 'what' if 'actual' doesn't equal 'expected'.
inline void expect_int(int64_t actual, int64_t expected,
                       const std::string &what) {
  if (actual != expected)
    throw Exception(what + ": expected " + std::to_string(expected) +
                    " but got " + std::to_string(actual));
}

// Synchronized data structure for listener to send events through
template<typename T>
class multithreaded_queue {
  std::condition_variable m_condition;
  std::mutex m_mutex;
  std::queue<T> m_data;
  bool m_notified;

public:

  void push(T e) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_data.push(e);
    m_notified = true;
    m_condition.notify_all();
  }

  T pop(int timeout_seconds, bool &success) {
    int count = 0;
    while (count < timeout_seconds) {
      std::unique_lock<std::mutex> lock(m_mutex);
      if (!m_data.empty()) {
        m_notified = false;
        T ret = m_data.front();
        m_data.pop();
        success = true;
        return ret;
      } else if (!m_notified)
        m_condition.wait_for(lock, std::chrono::seconds(1));
      count ++;
    }
    success = false;
    return T();
  }
};

/// Allocates a char buffer with the current working directory
inline char* get_working_dir() {
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) ||       \
    defined(__OpenBSD__)
    return getwd(0);
#else
    return get_current_dir_name();
#endif
}

#endif // LLDB_TEST_API_COMMON_H
