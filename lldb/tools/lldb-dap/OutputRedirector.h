//===-- OutputRedirector.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#ifndef LLDB_TOOLS_LLDB_DAP_OUTPUT_REDIRECTOR_H
#define LLDB_TOOLS_LLDB_DAP_OUTPUT_REDIRECTOR_H

#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <functional>

namespace lldb_dap {
class Log;
class OutputRedirector {
public:
  static int kInvalidDescriptor;

  /// Creates a writable file descriptor that will invoke the given callback
  /// when data is written, dispatched from the provided main loop.
  ///
  /// \param[in] loop
  ///     The main loop used to poll the read end of the redirection pipe.
  ///
  /// \param[in] file_override
  ///     If non-null, redirects this file's descriptor to the pipe so writes
  ///     to it are captured.
  ///
  /// \param[in] callback
  ///     Invoked on the main loop thread with each chunk of data read from
  ///     the pipe.
  ///
  /// \param[in] log
  ///     Used to report read errors from the redirection pipe.
  ///
  /// \return
  ///     \a Error::success if the redirection was set up correctly, or an error
  ///     otherwise.
  llvm::Error RedirectTo(lldb_private::MainLoopBase &loop,
                         std::FILE *file_override,
                         std::function<void(llvm::StringRef)> callback,
                         Log &log);

  llvm::Expected<int> GetWriteFileDescriptor();

  ~OutputRedirector() { Stop(); }

  OutputRedirector();
  OutputRedirector(const OutputRedirector &) = delete;
  OutputRedirector &operator=(const OutputRedirector &) = delete;

private:
  void Stop();

  int m_fd;
  int m_original_fd;
  int m_restore_fd;
  lldb::IOObjectSP m_read_obj;
  lldb_private::MainLoop::ReadHandleUP m_read_handle;
};

} // namespace lldb_dap

#endif // LLDB_TOOLS_LLDB_DAP_OUTPUT_REDIRECTOR_H
