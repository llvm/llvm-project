//===- Debugger.h - LLVM debugger library interface -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM source-level debugger library interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGGER_DEBUGGER_H
#define LLVM_DEBUGGER_DEBUGGER_H

#include <string>
#include <vector>

namespace llvm {
  class Module;
  class InferiorProcess;

  /// Debugger class - This class implements the LLVM source-level debugger.
  /// This allows clients to handle the user IO processing without having to
  /// worry about how the debugger itself works.
  ///
  class Debugger {
    // State the debugger needs when starting and stopping the program.
    std::vector<std::string> ProgramArguments;

    // The environment to run the program with.  This should eventually be
    // changed to vector of strings when we allow the user to edit the
    // environment.
    const char * const *Environment;

    // Program - The currently loaded program, or null if none is loaded.
    Module *Program;

    // Process - The currently executing inferior process.
    InferiorProcess *Process;

    Debugger(const Debugger &);         // DO NOT IMPLEMENT
    void operator=(const Debugger &);   // DO NOT IMPLEMENT
  public:
    Debugger();
    ~Debugger();

    //===------------------------------------------------------------------===//
    // Methods for manipulating and inspecting the execution environment.
    //

    /// initializeEnvironment - Specify the environment the program should run
    /// with.  This is used to initialize the environment of the program to the
    /// environment of the debugger.
    void initializeEnvironment(const char *const *envp) {
      Environment = envp;
    }

    /// setWorkingDirectory - Specify the working directory for the program to
    /// be started from.
    void setWorkingDirectory(const std::string &Dir) {
      // FIXME: implement
    }

    template<typename It>
    void setProgramArguments(It I, It E) {
      ProgramArguments.assign(I, E);
    }
    unsigned getNumProgramArguments() const {
      return static_cast<unsigned>(ProgramArguments.size());
    }
    const std::string &getProgramArgument(unsigned i) const {
      return ProgramArguments[i];
    }


    //===------------------------------------------------------------------===//
    // Methods for manipulating and inspecting the program currently loaded.
    //

    /// isProgramLoaded - Return true if there is a program currently loaded.
    ///
    bool isProgramLoaded() const { return Program != 0; }

    /// getProgram - Return the LLVM module corresponding to the program.
    ///
    Module *getProgram() const { return Program; }

    /// getProgramPath - Get the path of the currently loaded program, or an
    /// empty string if none is loaded.
    std::string getProgramPath() const;

    /// loadProgram - If a program is currently loaded, unload it.  Then search
    /// the PATH for the specified program, loading it when found.  If the
    /// specified program cannot be found, an exception is thrown to indicate
    /// the error.
    void loadProgram(const std::string &Path);

    /// unloadProgram - If a program is running, kill it, then unload all traces
    /// of the current program.  If no program is loaded, this method silently
    /// succeeds.
    void unloadProgram();

    //===------------------------------------------------------------------===//
    // Methods for manipulating and inspecting the program currently running.
    //
    // If the program is running, and the debugger is active, then we know that
    // the program has stopped.  This being the case, we can inspect the
    // program, ask it for its source location, set breakpoints, etc.
    //

    /// isProgramRunning - Return true if a program is loaded and has a
    /// currently active instance.
    bool isProgramRunning() const { return Process != 0; }

    /// getRunningProcess - If there is no program running, throw an exception.
    /// Otherwise return the running process so that it can be inspected by the
    /// debugger.
    const InferiorProcess &getRunningProcess() const {
      if (Process == 0) throw "No process running.";
      return *Process;
    }

    /// createProgram - Create an instance of the currently loaded program,
    /// killing off any existing one.  This creates the program and stops it at
    /// the first possible moment.  If there is no program loaded or if there is
    /// a problem starting the program, this method throws an exception.
    void createProgram();

    /// killProgram - If the program is currently executing, kill off the
    /// process and free up any state related to the currently running program.
    /// If there is no program currently running, this just silently succeeds.
    /// If something horrible happens when killing the program, an exception
    /// gets thrown.
    void killProgram();


    //===------------------------------------------------------------------===//
    // Methods for continuing execution.  These methods continue the execution
    // of the program by some amount.  If the program is successfully stopped,
    // execution returns, otherwise an exception is thrown.
    //
    // NOTE: These methods should always be used in preference to directly
    // accessing the Dbg object, because these will delete the Process object if
    // the process unexpectedly dies.
    //

    /// stepProgram - Implement the 'step' command, continuing execution until
    /// the next possible stop point.
    void stepProgram();

    /// nextProgram - Implement the 'next' command, continuing execution until
    /// the next possible stop point that is in the current function.
    void nextProgram();

    /// finishProgram - Implement the 'finish' command, continuing execution
    /// until the specified frame ID returns.
    void finishProgram(void *Frame);

    /// contProgram - Implement the 'cont' command, continuing execution until
    /// the next breakpoint is encountered.
    void contProgram();
  };

  class NonErrorException {
    std::string Message;
  public:
    NonErrorException(const std::string &M) : Message(M) {}
    const std::string &getMessage() const { return Message; }
  };

} // end namespace llvm

#endif
