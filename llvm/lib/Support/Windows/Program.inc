//===- Win32/Program.cpp - Win32 Program Implementation ------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Program class.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"
#include <cstdio>
#include <malloc.h>
#include <io.h>
#include <fcntl.h>

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

namespace {
  struct Win32ProcessInfo {
    HANDLE hProcess;
    DWORD  dwProcessId;
  };
}

namespace llvm {
using namespace sys;

Program::Program() : Data_(0) {}

Program::~Program() {
  if (Data_) {
    Win32ProcessInfo* wpi = reinterpret_cast<Win32ProcessInfo*>(Data_);
    CloseHandle(wpi->hProcess);
    delete wpi;
    Data_ = 0;
  }
}

unsigned Program::GetPid() const {
  Win32ProcessInfo* wpi = reinterpret_cast<Win32ProcessInfo*>(Data_);
  return wpi->dwProcessId;
}

// This function just uses the PATH environment variable to find the program.
Path
Program::FindProgramByName(const std::string& progName) {

  // Check some degenerate cases
  if (progName.length() == 0) // no program
    return Path();
  Path temp;
  if (!temp.set(progName)) // invalid name
    return Path();
  // Return paths with slashes verbatim.
  if (progName.find('\\') != std::string::npos ||
      progName.find('/') != std::string::npos)
    return temp;

  // At this point, the file name is valid and does not contain slashes.
  // Let Windows search for it.
  char buffer[MAX_PATH];
  char *dummy = NULL;
  DWORD len = SearchPath(NULL, progName.c_str(), ".exe", MAX_PATH,
                         buffer, &dummy);

  // See if it wasn't found.
  if (len == 0)
    return Path();

  // See if we got the entire path.
  if (len < MAX_PATH)
    return Path(buffer);

  // Buffer was too small; grow and retry.
  while (true) {
    char *b = reinterpret_cast<char *>(_alloca(len+1));
    DWORD len2 = SearchPath(NULL, progName.c_str(), ".exe", len+1, b, &dummy);

    // It is unlikely the search failed, but it's always possible some file
    // was added or removed since the last search, so be paranoid...
    if (len2 == 0)
      return Path();
    else if (len2 <= len)
      return Path(b);

    len = len2;
  }
}

static HANDLE RedirectIO(const Path *path, int fd, std::string* ErrMsg) {
  HANDLE h;
  if (path == 0) {
    DuplicateHandle(GetCurrentProcess(), (HANDLE)_get_osfhandle(fd),
                    GetCurrentProcess(), &h,
                    0, TRUE, DUPLICATE_SAME_ACCESS);
    return h;
  }

  const char *fname;
  if (path->isEmpty())
    fname = "NUL";
  else
    fname = path->c_str();

  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = 0;
  sa.bInheritHandle = TRUE;

  h = CreateFile(fname, fd ? GENERIC_WRITE : GENERIC_READ, FILE_SHARE_READ,
                 &sa, fd == 0 ? OPEN_EXISTING : CREATE_ALWAYS,
                 FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) {
    MakeErrMsg(ErrMsg, std::string(fname) + ": Can't open file for " +
        (fd ? "input: " : "output: "));
  }

  return h;
}

/// ArgNeedsQuotes - Check whether argument needs to be quoted when calling
/// CreateProcess.
static bool ArgNeedsQuotes(const char *Str) {
  return Str[0] == '\0' || strpbrk(Str, "\t \"&\'()*<>\\`^|") != 0;
}


/// ArgLenWithQuotes - Check whether argument needs to be quoted when calling
/// CreateProcess and returns length of quoted arg with escaped quotes
static unsigned int ArgLenWithQuotes(const char *Str) {
  unsigned int len = ArgNeedsQuotes(Str) ? 2 : 0;

  while (*Str != '\0') {
    if (*Str == '\"')
      ++len;

    ++len;
    ++Str;
  }

  return len;
}


bool
Program::Execute(const Path& path,
                 const char** args,
                 const char** envp,
                 const Path** redirects,
                 unsigned memoryLimit,
                 std::string* ErrMsg) {
  if (Data_) {
    Win32ProcessInfo* wpi = reinterpret_cast<Win32ProcessInfo*>(Data_);
    CloseHandle(wpi->hProcess);
    delete wpi;
    Data_ = 0;
  }

  if (!path.canExecute()) {
    if (ErrMsg)
      *ErrMsg = "program not executable";
    return false;
  }

  // Windows wants a command line, not an array of args, to pass to the new
  // process.  We have to concatenate them all, while quoting the args that
  // have embedded spaces (or are empty).

  // First, determine the length of the command line.
  unsigned len = 0;
  for (unsigned i = 0; args[i]; i++) {
    len += ArgLenWithQuotes(args[i]) + 1;
  }

  // Now build the command line.
  char *command = reinterpret_cast<char *>(_alloca(len+1));
  char *p = command;

  for (unsigned i = 0; args[i]; i++) {
    const char *arg = args[i];

    bool needsQuoting = ArgNeedsQuotes(arg);
    if (needsQuoting)
      *p++ = '"';

    while (*arg != '\0') {
      if (*arg == '\"')
        *p++ = '\\';

      *p++ = *arg++;
    }

    if (needsQuoting)
      *p++ = '"';
    *p++ = ' ';
  }

  *p = 0;

  // The pointer to the environment block for the new process.
  char *envblock = 0;

  if (envp) {
    // An environment block consists of a null-terminated block of
    // null-terminated strings. Convert the array of environment variables to
    // an environment block by concatenating them.

    // First, determine the length of the environment block.
    len = 0;
    for (unsigned i = 0; envp[i]; i++)
      len += strlen(envp[i]) + 1;

    // Now build the environment block.
    envblock = reinterpret_cast<char *>(_alloca(len+1));
    p = envblock;

    for (unsigned i = 0; envp[i]; i++) {
      const char *ev = envp[i];
      size_t len = strlen(ev) + 1;
      memcpy(p, ev, len);
      p += len;
    }

    *p = 0;
  }

  // Create a child process.
  STARTUPINFO si;
  memset(&si, 0, sizeof(si));
  si.cb = sizeof(si);
  si.hStdInput = INVALID_HANDLE_VALUE;
  si.hStdOutput = INVALID_HANDLE_VALUE;
  si.hStdError = INVALID_HANDLE_VALUE;

  if (redirects) {
    si.dwFlags = STARTF_USESTDHANDLES;

    si.hStdInput = RedirectIO(redirects[0], 0, ErrMsg);
    if (si.hStdInput == INVALID_HANDLE_VALUE) {
      MakeErrMsg(ErrMsg, "can't redirect stdin");
      return false;
    }
    si.hStdOutput = RedirectIO(redirects[1], 1, ErrMsg);
    if (si.hStdOutput == INVALID_HANDLE_VALUE) {
      CloseHandle(si.hStdInput);
      MakeErrMsg(ErrMsg, "can't redirect stdout");
      return false;
    }
    if (redirects[1] && redirects[2] && *(redirects[1]) == *(redirects[2])) {
      // If stdout and stderr should go to the same place, redirect stderr
      // to the handle already open for stdout.
      DuplicateHandle(GetCurrentProcess(), si.hStdOutput,
                      GetCurrentProcess(), &si.hStdError,
                      0, TRUE, DUPLICATE_SAME_ACCESS);
    } else {
      // Just redirect stderr
      si.hStdError = RedirectIO(redirects[2], 2, ErrMsg);
      if (si.hStdError == INVALID_HANDLE_VALUE) {
        CloseHandle(si.hStdInput);
        CloseHandle(si.hStdOutput);
        MakeErrMsg(ErrMsg, "can't redirect stderr");
        return false;
      }
    }
  }

  PROCESS_INFORMATION pi;
  memset(&pi, 0, sizeof(pi));

  fflush(stdout);
  fflush(stderr);
  BOOL rc = CreateProcess(path.c_str(), command, NULL, NULL, TRUE, 0,
                          envblock, NULL, &si, &pi);
  DWORD err = GetLastError();

  // Regardless of whether the process got created or not, we are done with
  // the handles we created for it to inherit.
  CloseHandle(si.hStdInput);
  CloseHandle(si.hStdOutput);
  CloseHandle(si.hStdError);

  // Now return an error if the process didn't get created.
  if (!rc) {
    SetLastError(err);
    MakeErrMsg(ErrMsg, std::string("Couldn't execute program '") +
               path.str() + "'");
    return false;
  }
  Win32ProcessInfo* wpi = new Win32ProcessInfo;
  wpi->hProcess = pi.hProcess;
  wpi->dwProcessId = pi.dwProcessId;
  Data_ = wpi;

  // Make sure these get closed no matter what.
  ScopedCommonHandle hThread(pi.hThread);

  // Assign the process to a job if a memory limit is defined.
  ScopedJobHandle hJob;
  if (memoryLimit != 0) {
    hJob = CreateJobObject(0, 0);
    bool success = false;
    if (hJob) {
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli;
      memset(&jeli, 0, sizeof(jeli));
      jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
      jeli.ProcessMemoryLimit = uintptr_t(memoryLimit) * 1048576;
      if (SetInformationJobObject(hJob, JobObjectExtendedLimitInformation,
                                  &jeli, sizeof(jeli))) {
        if (AssignProcessToJobObject(hJob, pi.hProcess))
          success = true;
      }
    }
    if (!success) {
      SetLastError(GetLastError());
      MakeErrMsg(ErrMsg, std::string("Unable to set memory limit"));
      TerminateProcess(pi.hProcess, 1);
      WaitForSingleObject(pi.hProcess, INFINITE);
      return false;
    }
  }

  return true;
}

int
Program::Wait(const Path &path,
              unsigned secondsToWait,
              std::string* ErrMsg) {
  if (Data_ == 0) {
    MakeErrMsg(ErrMsg, "Process not started!");
    return -1;
  }

  Win32ProcessInfo* wpi = reinterpret_cast<Win32ProcessInfo*>(Data_);
  HANDLE hProcess = wpi->hProcess;

  // Wait for the process to terminate.
  DWORD millisecondsToWait = INFINITE;
  if (secondsToWait > 0)
    millisecondsToWait = secondsToWait * 1000;

  if (WaitForSingleObject(hProcess, millisecondsToWait) == WAIT_TIMEOUT) {
    if (!TerminateProcess(hProcess, 1)) {
      MakeErrMsg(ErrMsg, "Failed to terminate timed-out program.");
      // -2 indicates a crash or timeout as opposed to failure to execute.
      return -2;
    }
    WaitForSingleObject(hProcess, INFINITE);
  }

  // Get its exit status.
  DWORD status;
  BOOL rc = GetExitCodeProcess(hProcess, &status);
  DWORD err = GetLastError();

  if (!rc) {
    SetLastError(err);
    MakeErrMsg(ErrMsg, "Failed getting status for program.");
    // -2 indicates a crash or timeout as opposed to failure to execute.
    return -2;
  }

  if (!status)
    return 0;

  // Pass 10(Warning) and 11(Error) to the callee as negative value.
  if ((status & 0xBFFF0000U) == 0x80000000U)
    return (int)status;

  if (status & 0xFF)
    return status & 0x7FFFFFFF;

  return 1;
}

bool
Program::Kill(std::string* ErrMsg) {
  if (Data_ == 0) {
    MakeErrMsg(ErrMsg, "Process not started!");
    return true;
  }

  Win32ProcessInfo* wpi = reinterpret_cast<Win32ProcessInfo*>(Data_);
  HANDLE hProcess = wpi->hProcess;
  if (TerminateProcess(hProcess, 1) == 0) {
    MakeErrMsg(ErrMsg, "The process couldn't be killed!");
    return true;
  }

  return false;
}

error_code Program::ChangeStdinToBinary(){
  int result = _setmode( _fileno(stdin), _O_BINARY );
  if (result == -1)
    return error_code(errno, generic_category());
  return make_error_code(errc::success);
}

error_code Program::ChangeStdoutToBinary(){
  int result = _setmode( _fileno(stdout), _O_BINARY );
  if (result == -1)
    return error_code(errno, generic_category());
  return make_error_code(errc::success);
}

error_code Program::ChangeStderrToBinary(){
  int result = _setmode( _fileno(stderr), _O_BINARY );
  if (result == -1)
    return error_code(errno, generic_category());
  return make_error_code(errc::success);
}

}
