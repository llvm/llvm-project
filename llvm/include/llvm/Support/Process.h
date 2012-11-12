//===- llvm/Support/Process.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Process class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_PROCESS_H
#define LLVM_SYSTEM_PROCESS_H

#include "llvm/Support/TimeValue.h"

namespace llvm {
namespace sys {

  /// This class provides an abstraction for getting information about the
  /// currently executing process.
  /// @since 1.4
  /// @brief An abstraction for operating system processes.
  class Process {
    /// @name Accessors
    /// @{
    public:
      /// This static function will return the operating system's virtual memory
      /// page size.
      /// @returns The number of bytes in a virtual memory page.
      /// @brief Get the virtual memory page size
      static unsigned GetPageSize();

      /// This static function will return the total amount of memory allocated
      /// by the process. This only counts the memory allocated via the malloc,
      /// calloc and realloc functions and includes any "free" holes in the
      /// allocated space.
      /// @brief Return process memory usage.
      static size_t GetMallocUsage();

      /// This static function will return the total memory usage of the
      /// process. This includes code, data, stack and mapped pages usage. Notei
      /// that the value returned here is not necessarily the Running Set Size,
      /// it is the total virtual memory usage, regardless of mapped state of
      /// that memory.
      static size_t GetTotalMemoryUsage();

      /// This static function will set \p user_time to the amount of CPU time
      /// spent in user (non-kernel) mode and \p sys_time to the amount of CPU
      /// time spent in system (kernel) mode.  If the operating system does not
      /// support collection of these metrics, a zero TimeValue will be for both
      /// values.
      static void GetTimeUsage(
        TimeValue& elapsed,
          ///< Returns the TimeValue::now() giving current time
        TimeValue& user_time,
          ///< Returns the current amount of user time for the process
        TimeValue& sys_time
          ///< Returns the current amount of system time for the process
      );

      /// This static function will return the process' current user id number.
      /// Not all operating systems support this feature. Where it is not
      /// supported, the function should return 65536 as the value.
      static int GetCurrentUserId();

      /// This static function will return the process' current group id number.
      /// Not all operating systems support this feature. Where it is not
      /// supported, the function should return 65536 as the value.
      static int GetCurrentGroupId();

      /// This function makes the necessary calls to the operating system to
      /// prevent core files or any other kind of large memory dumps that can
      /// occur when a program fails.
      /// @brief Prevent core file generation.
      static void PreventCoreFiles();

      /// This function determines if the standard input is connected directly
      /// to a user's input (keyboard probably), rather than coming from a file
      /// or pipe.
      static bool StandardInIsUserInput();

      /// This function determines if the standard output is connected to a
      /// "tty" or "console" window. That is, the output would be displayed to
      /// the user rather than being put on a pipe or stored in a file.
      static bool StandardOutIsDisplayed();

      /// This function determines if the standard error is connected to a
      /// "tty" or "console" window. That is, the output would be displayed to
      /// the user rather than being put on a pipe or stored in a file.
      static bool StandardErrIsDisplayed();

      /// This function determines if the given file descriptor is connected to
      /// a "tty" or "console" window. That is, the output would be displayed to
      /// the user rather than being put on a pipe or stored in a file.
      static bool FileDescriptorIsDisplayed(int fd);

      /// This function determines if the given file descriptor is displayd and
      /// supports colors.
      static bool FileDescriptorHasColors(int fd);

      /// This function determines the number of columns in the window
      /// if standard output is connected to a "tty" or "console"
      /// window. If standard output is not connected to a tty or
      /// console, or if the number of columns cannot be determined,
      /// this routine returns zero.
      static unsigned StandardOutColumns();

      /// This function determines the number of columns in the window
      /// if standard error is connected to a "tty" or "console"
      /// window. If standard error is not connected to a tty or
      /// console, or if the number of columns cannot be determined,
      /// this routine returns zero.
      static unsigned StandardErrColumns();

      /// This function determines whether the terminal connected to standard
      /// output supports colors. If standard output is not connected to a
      /// terminal, this function returns false.
      static bool StandardOutHasColors();

      /// This function determines whether the terminal connected to standard
      /// error supports colors. If standard error is not connected to a
      /// terminal, this function returns false.
      static bool StandardErrHasColors();

      /// Whether changing colors requires the output to be flushed.
      /// This is needed on systems that don't support escape sequences for
      /// changing colors.
      static bool ColorNeedsFlush();

      /// This function returns the colorcode escape sequences.
      /// If ColorNeedsFlush() is true then this function will change the colors
      /// and return an empty escape sequence. In that case it is the
      /// responsibility of the client to flush the output stream prior to
      /// calling this function.
      static const char *OutputColor(char c, bool bold, bool bg);

      /// Same as OutputColor, but only enables the bold attribute.
      static const char *OutputBold(bool bg);

      /// This function returns the escape sequence to reverse forground and
      /// background colors.
      static const char *OutputReverse();

      /// Resets the terminals colors, or returns an escape sequence to do so.
      static const char *ResetColor();

      /// Get the result of a process wide random number generator. The
      /// generator will be automatically seeded in non-deterministic fashion.
      static unsigned GetRandomNumber();
    /// @}
  };
}
}

#endif
