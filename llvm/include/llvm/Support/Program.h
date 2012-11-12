//===- llvm/Support/Program.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Program class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_PROGRAM_H
#define LLVM_SYSTEM_PROGRAM_H

#include "llvm/Support/Path.h"

namespace llvm {
class error_code;
namespace sys {

  // TODO: Add operations to communicate with the process, redirect its I/O,
  // etc.

  /// This class provides an abstraction for programs that are executable by the
  /// operating system. It provides a platform generic way to find executable
  /// programs from the path and to execute them in various ways. The sys::Path
  /// class is used to specify the location of the Program.
  /// @since 1.4
  /// @brief An abstraction for finding and executing programs.
  class Program {
    /// Opaque handle for target specific data.
    void *Data_;

    // Noncopyable.
    Program(const Program& other) LLVM_DELETED_FUNCTION;
    Program& operator=(const Program& other) LLVM_DELETED_FUNCTION;

    /// @name Methods
    /// @{
  public:

    Program();
    ~Program();

    /// Return process ID of this program.
    unsigned GetPid() const;

    /// This function executes the program using the \p arguments provided.  The
    /// invoked program will inherit the stdin, stdout, and stderr file
    /// descriptors, the environment and other configuration settings of the
    /// invoking program. If Path::executable() does not return true when this
    /// function is called then a std::string is thrown.
    /// @returns false in case of error, true otherwise.
    /// @see FindProgramByName
    /// @brief Executes the program with the given set of \p args.
    bool Execute
    ( const Path& path,  ///< sys::Path object providing the path of the
      ///< program to be executed. It is presumed this is the result of
      ///< the FindProgramByName method.
      const char** args, ///< A vector of strings that are passed to the
      ///< program.  The first element should be the name of the program.
      ///< The list *must* be terminated by a null char* entry.
      const char ** env = 0, ///< An optional vector of strings to use for
      ///< the program's environment. If not provided, the current program's
      ///< environment will be used.
      const sys::Path** redirects = 0, ///< An optional array of pointers to
      ///< Paths. If the array is null, no redirection is done. The array
      ///< should have a size of at least three. If the pointer in the array
      ///< are not null, then the inferior process's stdin(0), stdout(1),
      ///< and stderr(2) will be redirected to the corresponding Paths.
      ///< When an empty Path is passed in, the corresponding file
      ///< descriptor will be disconnected (ie, /dev/null'd) in a portable
      ///< way.
      unsigned memoryLimit = 0, ///< If non-zero, this specifies max. amount
      ///< of memory can be allocated by process. If memory usage will be
      ///< higher limit, the child is killed and this call returns. If zero
      ///< - no memory limit.
      std::string* ErrMsg = 0 ///< If non-zero, provides a pointer to a string
      ///< instance in which error messages will be returned. If the string
      ///< is non-empty upon return an error occurred while invoking the
      ///< program.
      );

    /// This function waits for the program to exit. This function will block
    /// the current program until the invoked program exits.
    /// @returns an integer result code indicating the status of the program.
    /// A zero or positive value indicates the result code of the program.
    /// -1 indicates failure to execute
    /// -2 indicates a crash during execution or timeout
    /// @see Execute
    /// @brief Waits for the program to exit.
    int Wait
    ( const Path& path, ///< The path to the child process executable.
      unsigned secondsToWait, ///< If non-zero, this specifies the amount
      ///< of time to wait for the child process to exit. If the time
      ///< expires, the child is killed and this call returns. If zero,
      ///< this function will wait until the child finishes or forever if
      ///< it doesn't.
      std::string* ErrMsg ///< If non-zero, provides a pointer to a string
      ///< instance in which error messages will be returned. If the string
      ///< is non-empty upon return an error occurred while waiting.
      );

    /// This function terminates the program.
    /// @returns true if an error occurred.
    /// @see Execute
    /// @brief Terminates the program.
    bool Kill
    ( std::string* ErrMsg = 0 ///< If non-zero, provides a pointer to a string
      ///< instance in which error messages will be returned. If the string
      ///< is non-empty upon return an error occurred while killing the
      ///< program.
      );

    /// This static constructor (factory) will attempt to locate a program in
    /// the operating system's file system using some pre-determined set of
    /// locations to search (e.g. the PATH on Unix). Paths with slashes are
    /// returned unmodified.
    /// @returns A Path object initialized to the path of the program or a
    /// Path object that is empty (invalid) if the program could not be found.
    /// @brief Construct a Program by finding it by name.
    static Path FindProgramByName(const std::string& name);

    // These methods change the specified standard stream (stdin, stdout, or
    // stderr) to binary mode. They return errc::success if the specified stream
    // was changed. Otherwise a platform dependent error is returned.
    static error_code ChangeStdinToBinary();
    static error_code ChangeStdoutToBinary();
    static error_code ChangeStderrToBinary();

    /// A convenience function equivalent to Program prg; prg.Execute(..);
    /// prg.Wait(..);
    /// @see Execute, Wait
    static int ExecuteAndWait(const Path& path,
                              const char** args,
                              const char ** env = 0,
                              const sys::Path** redirects = 0,
                              unsigned secondsToWait = 0,
                              unsigned memoryLimit = 0,
                              std::string* ErrMsg = 0);

    /// A convenience function equivalent to Program prg; prg.Execute(..);
    /// @see Execute
    static void ExecuteNoWait(const Path& path,
                              const char** args,
                              const char ** env = 0,
                              const sys::Path** redirects = 0,
                              unsigned memoryLimit = 0,
                              std::string* ErrMsg = 0);

    /// @}

  };
}
}

#endif
