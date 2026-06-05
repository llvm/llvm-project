//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBLLDB_PLUGINS_PROCESS_WINDOWS_COMMON_IO_HANDLER_PROCESS_STDIO_WINDOWS_H_
#define LIBLLDB_PLUGINS_PROCESS_WINDOWS_COMMON_IO_HANDLER_PROCESS_STDIO_WINDOWS_H_

#include "lldb/Core/IOHandler.h"
#include "lldb/Host/File.h"
#include "lldb/Target/Process.h"

using HANDLE = void *;

using namespace lldb_private;

class IOHandlerProcessSTDIOWindows : public IOHandler {
public:
  IOHandlerProcessSTDIOWindows(Process *process);

  ~IOHandlerProcessSTDIOWindows() override;

  void SetIsRunning(bool running);

  /// Peek the console for input. If it has any, drain the pipe until text input
  /// is found or the pipe is empty.
  ///
  /// \param hStdin
  ///     The handle to the standard input's pipe.
  ///
  /// \return
  ///     true if the pipe has text input.
  llvm::Expected<bool> ConsoleHasTextInput(const HANDLE hStdin);

  void Run() override;

  void Cancel() override;

  bool Interrupt() override;

  void GotEOF() override {}

private:
  enum ControlOp : char {
    eControlOpQuit = 'q',
    eControlOpInterrupt = 'i',
    eControlOpNone = 0,
  };

  Process *m_process;
  /// Read from this file (usually actual STDIN for LLDB)
  NativeFile m_read_file;
  HANDLE m_interrupt_event =
      reinterpret_cast<HANDLE>(static_cast<intptr_t>(-1));
  std::atomic<ControlOp> m_pending_op{eControlOpNone};
  std::mutex m_mutex;
  bool m_is_running = false;
};

#endif // LIBLLDB_PLUGINS_PROCESS_WINDOWS_COMMON_IO_HANDLER_PROCESS_STDIO_WINDOWS_H_
