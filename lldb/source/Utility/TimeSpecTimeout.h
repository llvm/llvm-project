//===--------------------- TimeSpecTimeout.h --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_TimeSpecTimeout_h_
#define utility_TimeSpecTimeout_h_

#include "lldb/Host/TimeValue.h"

namespace lldb_private {

class TimeSpecTimeout {
public:
  TimeSpecTimeout() : m_infinite(false) {
    m_timespec.tv_sec = 0;
    m_timespec.tv_nsec = 0;
  }
  ~TimeSpecTimeout() {}

  //----------------------------------------------------------------------
  /// Sets the timespec pointer correctly given a timeout relative to the
  /// current time. This function should be called immediately prior to
  /// calling the function you will use this timeout with since time can
  /// elapse between when this function is called and when the timeout is
  /// used.
  ///
  /// @param[in] timeout_usec
  ///     The timeout in micro seconds. If timeout_usec is UINT32_MAX, the
  ///     timeout should be set to INFINITE. Otherwise the current time is
  ///     filled into the timespec and \a timeout_usec is added to the
  ///     current time.
  ///
  /// @return
  ///     If the timeout is INFINITE, then return NULL, otherwise return
  ///     a pointer to the timespec with the appropriate timeout value.
  //----------------------------------------------------------------------
  const struct timespec *
  SetAbsoluteTimeoutMircoSeconds32(uint32_t timeout_usec);

  //----------------------------------------------------------------------
  /// Sets the timespec pointer correctly given a relative time in micro
  /// seconds.
  ///
  /// @param[in] timeout_usec
  ///     The timeout in micro seconds. If timeout_usec is UINT32_MAX, the
  ///     timeout should be set to INFINITE. Otherwise \a timeout_usec
  ///     is correctly placed into the timespec.
  ///
  /// @return
  ///     If the timeout is INFINITE, then return NULL, otherwise return
  ///     a pointer to the timespec with the appropriate timeout value.
  //----------------------------------------------------------------------
  const struct timespec *
  SetRelativeTimeoutMircoSeconds32(uint32_t timeout_usec);

  //----------------------------------------------------------------------
  /// Gets the timespec pointer that is appropriate for the timeout
  /// specified. This function should only be used after a call to
  /// SetRelativeTimeoutXXX() functions.
  ///
  /// @return
  ///     If the timeout is INFINITE, then return NULL, otherwise return
  ///     a pointer to the timespec with the appropriate timeout value.
  //----------------------------------------------------------------------
  const struct timespec *GetTimeSpecPtr() const {
    if (m_infinite)
      return nullptr;
    return &m_timespec;
  }

protected:
  struct timespec m_timespec;
  bool m_infinite;
};

} // namespace lldb_private

#endif // #ifndef utility_TimeSpecTimeout_h_
