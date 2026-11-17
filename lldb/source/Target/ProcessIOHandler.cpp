//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ProcessIOHandler.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Host/Terminal.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/SelectHelper.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"

using namespace lldb_private;

IOHandlerProcessSTDIO::IOHandlerProcessSTDIO(Process *process, int write_fd)
    : IOHandler(process->GetTarget().GetDebugger(), IOHandler::Type::ProcessIO),
      m_process(process),
      m_read_file(GetInputFD(), File::eOpenOptionReadOnly, false),
      m_write_file(write_fd, File::eOpenOptionWriteOnly, false) {
  m_pipe.CreateNew();
}

void IOHandlerProcessSTDIO::SetIsRunning(bool running) {
  std::lock_guard<std::mutex> guard(m_mutex);
  SetIsDone(!running);
  m_is_running = running;
}

// Each IOHandler gets to run until it is done. It should read data from the
// "in" and place output into "out" and "err and return when done.
void IOHandlerProcessSTDIO::Run() {
  if (!m_read_file.IsValid() || !m_write_file.IsValid() || !m_pipe.CanRead() ||
      !m_pipe.CanWrite()) {
    SetIsDone(true);
    return;
  }

  SetIsDone(false);
  const int read_fd = m_read_file.GetDescriptor();
  Terminal terminal(read_fd);
  TerminalState terminal_state(terminal, false);
  // FIXME: error handling?
  llvm::consumeError(terminal.SetCanonical(false));
  llvm::consumeError(terminal.SetEcho(false));
// FD_ZERO, FD_SET are not supported on windows
#ifndef _WIN32
  const int pipe_read_fd = m_pipe.GetReadFileDescriptor();
  SetIsRunning(true);
  while (true) {
    {
      std::lock_guard<std::mutex> guard(m_mutex);
      if (GetIsDone())
        break;
    }

    SelectHelper select_helper;
    select_helper.FDSetRead(read_fd);
    select_helper.FDSetRead(pipe_read_fd);
    Status error = select_helper.Select();

    if (error.Fail())
      break;

    char ch = 0;
    size_t n;
    if (select_helper.FDIsSetRead(read_fd)) {
      n = 1;
      if (m_read_file.Read(&ch, n).Success() && n == 1) {
        if (m_write_file.Write(&ch, n).Fail() || n != 1)
          break;
      } else
        break;
    }

    if (select_helper.FDIsSetRead(pipe_read_fd)) {
      // Consume the interrupt byte
      if (llvm::Expected<size_t> bytes_read = m_pipe.Read(&ch, 1)) {
        if (ch == 'q')
          break;
        if (ch == 'i')
          if (StateIsRunningState(m_process->GetState()))
            m_process->SendAsyncInterrupt();
      } else {
        LLDB_LOG_ERROR(GetLog(LLDBLog::Process), bytes_read.takeError(),
                       "Pipe read failed: {0}");
      }
    }
  }
  SetIsRunning(false);
#endif
}

void IOHandlerProcessSTDIO::Cancel() {
  std::lock_guard<std::mutex> guard(m_mutex);
  SetIsDone(true);
  // Only write to our pipe to cancel if we are in
  // IOHandlerProcessSTDIO::Run(). We can end up with a python command that
  // is being run from the command interpreter:
  //
  // (lldb) step_process_thousands_of_times
  //
  // In this case the command interpreter will be in the middle of handling
  // the command and if the process pushes and pops the IOHandler thousands
  // of times, we can end up writing to m_pipe without ever consuming the
  // bytes from the pipe in IOHandlerProcessSTDIO::Run() and end up
  // deadlocking when the pipe gets fed up and blocks until data is consumed.
  if (m_is_running) {
    char ch = 'q'; // Send 'q' for quit
    if (llvm::Error err = m_pipe.Write(&ch, 1).takeError()) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Process), std::move(err),
                     "Pipe write failed: {0}");
    }
  }
}

bool IOHandlerProcessSTDIO::Interrupt() {
  // Do only things that are safe to do in an interrupt context (like in a
  // SIGINT handler), like write 1 byte to a file descriptor. This will
  // interrupt the IOHandlerProcessSTDIO::Run() and we can look at the byte
  // that was written to the pipe and then call
  // m_process->SendAsyncInterrupt() from a much safer location in code.
  if (m_active) {
    char ch = 'i'; // Send 'i' for interrupt
    return !errorToBool(m_pipe.Write(&ch, 1).takeError());
  } else {
    // This IOHandler might be pushed on the stack, but not being run
    // currently so do the right thing if we aren't actively watching for
    // STDIN by sending the interrupt to the process. Otherwise the write to
    // the pipe above would do nothing. This can happen when the command
    // interpreter is running and gets a "expression ...". It will be on the
    // IOHandler thread and sending the input is complete to the delegate
    // which will cause the expression to run, which will push the process IO
    // handler, but not run it.

    if (StateIsRunningState(m_process->GetState())) {
      m_process->SendAsyncInterrupt();
      return true;
    }
  }
  return false;
}

#ifdef _WIN32

#include "lldb/Host/windows/windows.h"

IOHandlerProcessSTDIOWindows::IOHandlerProcessSTDIOWindows(Process *process)
    : IOHandler(process->GetTarget().GetDebugger(), IOHandler::Type::ProcessIO),
      m_process(process),
      m_read_file(GetInputFD(), File::eOpenOptionReadOnly, false),
      m_interrupt_event(
          CreateEvent(/*lpEventAttributes=*/nullptr, /*bManualReset=*/FALSE,
                      /*bInitialState=*/FALSE, /*lpName=*/nullptr)) {}

IOHandlerProcessSTDIOWindows::~IOHandlerProcessSTDIOWindows() {
  if (m_interrupt_event != INVALID_HANDLE_VALUE)
    ::CloseHandle(m_interrupt_event);
}

void IOHandlerProcessSTDIOWindows::SetIsRunning(bool running) {
  std::lock_guard<std::mutex> guard(m_mutex);
  SetIsDone(!running);
  m_is_running = running;
}

llvm::Expected<bool>
IOHandlerProcessSTDIOWindows::ConsoleHasTextInput(const HANDLE hStdin) {
  // Check if there are already characters buffered. Pressing enter counts as
  // 2 characters '\r\n' and only one of them is a keyDown event.
  DWORD bytesAvailable = 0;
  if (PeekNamedPipe(hStdin, nullptr, 0, nullptr, &bytesAvailable, nullptr)) {
    if (bytesAvailable > 0)
      return true;
  }

  while (true) {
    INPUT_RECORD inputRecord;
    DWORD numRead = 0;
    if (!PeekConsoleInput(hStdin, &inputRecord, 1, &numRead))
      return llvm::createStringError("failed to peek standard input");

    if (numRead == 0)
      return false;

    if (inputRecord.EventType == KEY_EVENT &&
        inputRecord.Event.KeyEvent.bKeyDown &&
        inputRecord.Event.KeyEvent.uChar.AsciiChar != 0)
      return true;

    if (!ReadConsoleInput(hStdin, &inputRecord, 1, &numRead))
      return llvm::createStringError("failed to read standard input");
  }
}

void IOHandlerProcessSTDIOWindows::Run() {
  if (!m_read_file.IsValid()) {
    SetIsDone(true);
    return;
  }

  SetIsDone(false);
  SetIsRunning(true);

  HANDLE hStdin = m_read_file.GetWaitableHandle();
  HANDLE waitHandles[2] = {hStdin, m_interrupt_event};

  DWORD consoleMode;
  bool isConsole = GetConsoleMode(hStdin, &consoleMode) != 0;
  // With ENABLE_LINE_INPUT, ReadFile returns only when a carriage return is
  // read. This will block lldb in ReadFile until the user hits enter. Save
  // the previous console mode to restore it later and remove
  // ENABLE_LINE_INPUT.
  DWORD oldConsoleMode = consoleMode;
  SetConsoleMode(hStdin, consoleMode & ~ENABLE_LINE_INPUT & ~ENABLE_ECHO_INPUT);

  while (true) {
    {
      std::lock_guard<std::mutex> guard(m_mutex);
      if (GetIsDone())
        goto exit_loop;
    }

    DWORD result = WaitForMultipleObjects(2, waitHandles, FALSE, INFINITE);
    switch (result) {
    case WAIT_FAILED:
      goto exit_loop;
    case WAIT_OBJECT_0: {
      if (isConsole) {
        auto hasInputOrErr = ConsoleHasTextInput(hStdin);
        if (!hasInputOrErr) {
          Log *log = GetLog(LLDBLog::Process);
          LLDB_LOG_ERROR(log, hasInputOrErr.takeError(),
                         "failed to process debuggee's IO: {0}");
          goto exit_loop;
        }

        // If no text input is ready, go back to waiting.
        if (!*hasInputOrErr)
          continue;
      }

      char ch = 0;
      DWORD read = 0;
      if (!ReadFile(hStdin, &ch, 1, &read, nullptr) || read != 1)
        goto exit_loop;

      Status err;
      m_process->PutSTDIN(&ch, 1, err);
      if (err.Fail())
        goto exit_loop;
      break;
    }
    case WAIT_OBJECT_0 + 1: {
      ControlOp op = m_pending_op.exchange(eControlOpNone);
      if (op == eControlOpQuit)
        goto exit_loop;
      if (op == eControlOpInterrupt &&
          StateIsRunningState(m_process->GetState()))
        m_process->SendAsyncInterrupt();
      break;
    }
    default:
      goto exit_loop;
    }
  }

exit_loop:;
  SetIsRunning(false);
  SetIsDone(true);
  SetConsoleMode(hStdin, oldConsoleMode);
}

void IOHandlerProcessSTDIOWindows::Cancel() {
  std::lock_guard<std::mutex> guard(m_mutex);
  SetIsDone(true);
  if (m_is_running) {
    m_pending_op.store(eControlOpQuit);
    ::SetEvent(m_interrupt_event);
  }
}

bool IOHandlerProcessSTDIOWindows::Interrupt() {
  if (m_active) {
    m_pending_op.store(eControlOpInterrupt);
    ::SetEvent(m_interrupt_event);
    return true;
  }
  if (StateIsRunningState(m_process->GetState())) {
    m_process->SendAsyncInterrupt();
    return true;
  }
  return false;
}

#endif // _WIN32
