//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IOHandlerProcessSTDIOWindows.h"

#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"

using namespace lldb_private;

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

/// Peek the console for input. If it has any, drain the pipe until text input
/// is found or the pipe is empty.
///
/// \param hStdin
///     The handle to the standard input's pipe.
///
/// \return
///     true if the pipe has text input.
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
