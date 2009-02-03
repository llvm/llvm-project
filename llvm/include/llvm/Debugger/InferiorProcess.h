//===- InferiorProcess.h - Represent the program being debugged -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferiorProcess class, which is used to represent,
// inspect, and manipulate a process under the control of the LLVM debugger.
//
// This is an abstract class which should allow various different types of
// implementations.  Initially we implement a unix specific debugger backend
// that does not require code generator support, but we could eventually use
// code generator support with ptrace, support windows based targets, supported
// remote targets, etc.
//
// If the inferior process unexpectedly dies, an attempt to communicate with it
// will cause an InferiorProcessDead exception to be thrown, indicating the exit
// code of the process.  When this occurs, no methods on the InferiorProcess
// class should be called except for the destructor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGGER_INFERIORPROCESS_H
#define LLVM_DEBUGGER_INFERIORPROCESS_H

#include <string>
#include <vector>

namespace llvm {
  class Module;
  class GlobalVariable;

  /// InferiorProcessDead exception - This class is thrown by methods that
  /// communicate with the interior process if the process unexpectedly exits or
  /// dies.  The instance variable indicates what the exit code of the process
  /// was, or -1 if unknown.
  class InferiorProcessDead {
    int ExitCode;
  public:
    InferiorProcessDead(int EC) : ExitCode(EC) {}
    int getExitCode() const { return ExitCode; }
  };

  /// InferiorProcess class - This class represents the process being debugged
  /// by the debugger.  Objects of this class should not be stack allocated,
  /// because the destructor can throw exceptions.
  ///
  class InferiorProcess {
    Module *M;
  protected:
    InferiorProcess(Module *m) : M(m) {}
  public:
    /// create - Create an inferior process of the specified module, and
    /// stop it at the first opportunity.  If there is a problem starting the
    /// program (for example, it has no main), throw an exception.
    static InferiorProcess *create(Module *M,
                                   const std::vector<std::string> &Arguments,
                                   const char * const *envp);

    // InferiorProcess destructor - Kill the current process.  If something
    // terrible happens, we throw an exception from the destructor.
    virtual ~InferiorProcess() {}

    //===------------------------------------------------------------------===//
    // Status methods - These methods return information about the currently
    // stopped process.
    //

    /// getStatus - Return a status message that is specific to the current type
    /// of inferior process that is created.  This can return things like the
    /// PID of the inferior or other potentially interesting things.
    virtual std::string getStatus() const {
      return "";
    }

    //===------------------------------------------------------------------===//
    // Methods for inspecting the call stack.
    //

    /// getPreviousFrame - Given the descriptor for the current stack frame,
    /// return the descriptor for the caller frame.  This returns null when it
    /// runs out of frames.  If Frame is null, the initial frame should be
    /// returned.
    virtual void *getPreviousFrame(void *Frame) const = 0;

    /// getSubprogramDesc - Return the subprogram descriptor for the current
    /// stack frame.
    virtual const GlobalVariable *getSubprogramDesc(void *Frame) const = 0;

    /// getFrameLocation - This method returns the source location where each
    /// stack frame is stopped.
    virtual void getFrameLocation(void *Frame, unsigned &LineNo,
                                  unsigned &ColNo,
                                  const GlobalVariable *&SourceDesc) const = 0;

    //===------------------------------------------------------------------===//
    // Methods for manipulating breakpoints.
    //

    /// addBreakpoint - This method adds a breakpoint at the specified line,
    /// column, and source file, and returns a unique identifier for it.
    ///
    /// It is up to the debugger to determine whether or not there is actually a
    /// stop-point that corresponds with the specified location.
    virtual unsigned addBreakpoint(unsigned LineNo, unsigned ColNo,
                                   const GlobalVariable *SourceDesc) = 0;

    /// removeBreakpoint - This deletes the breakpoint with the specified ID
    /// number.
    virtual void removeBreakpoint(unsigned ID) = 0;


    //===------------------------------------------------------------------===//
    // Execution methods - These methods cause the program to continue execution
    // by some amount.  If the program successfully stops, this returns.
    // Otherwise, if the program unexpectedly terminates, an InferiorProcessDead
    // exception is thrown.
    //

    /// stepProgram - Implement the 'step' command, continuing execution until
    /// the next possible stop point.
    virtual void stepProgram() = 0;

    /// finishProgram - Implement the 'finish' command, continuing execution
    /// until the current function returns.
    virtual void finishProgram(void *Frame) = 0;

    /// contProgram - Implement the 'cont' command, continuing execution until
    /// a breakpoint is encountered.
    virtual void contProgram() = 0;
  };
}  // end namespace llvm

#endif
