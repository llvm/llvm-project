#ifndef LLDB_TEST_API_COMMON_H
#define LLDB_TEST_API_COMMON_H

#include <condition_variable>
#include <chrono>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <queue>

#include <unistd.h>

/// Simple exception class with a message
struct Exception : public std::exception
{
  std::string s;
  Exception(std::string ss) : s(ss) {}
  virtual ~Exception() throw () { }
  const char* what() const throw() { return s.c_str(); }
};

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
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)
    return getwd(0);
#else
    return get_current_dir_name();
#endif
}

#endif // LLDB_TEST_API_COMMON_H
