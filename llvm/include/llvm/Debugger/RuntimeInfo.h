//===- RuntimeInfo.h - Information about running program --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes that capture various pieces of information about
// the currently executing, but stopped, program.  One instance of this object
// is created every time a program is stopped, and destroyed every time it
// starts running again.  This object's main goal is to make access to runtime
// information easy and efficient, by caching information as requested.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGGER_RUNTIMEINFO_H
#define LLVM_DEBUGGER_RUNTIMEINFO_H

#include <vector>
#include <cassert>

namespace llvm {
  class ProgramInfo;
  class RuntimeInfo;
  class InferiorProcess;
  class GlobalVariable;
  class SourceFileInfo;

  /// StackFrame - One instance of this structure is created for each stack
  /// frame that is active in the program.
  ///
  class StackFrame {
    RuntimeInfo &RI;
    void *FrameID;
    const GlobalVariable *FunctionDesc;

    /// LineNo, ColNo, FileInfo - This information indicates WHERE in the source
    /// code for the program the stack frame is located.
    unsigned LineNo, ColNo;
    const SourceFileInfo *SourceInfo;
  public:
    StackFrame(RuntimeInfo &RI, void *ParentFrameID);

    StackFrame &operator=(const StackFrame &RHS) {
      FrameID = RHS.FrameID;
      FunctionDesc = RHS.FunctionDesc;
      return *this;
    }

    /// getFrameID - return the low-level opaque frame ID of this stack frame.
    ///
    void *getFrameID() const { return FrameID; }

    /// getFunctionDesc - Return the descriptor for the function that contains
    /// this stack frame, or null if it is unknown.
    ///
    const GlobalVariable *getFunctionDesc();

    /// getSourceLocation - Return the source location that this stack frame is
    /// sitting at.
    void getSourceLocation(unsigned &LineNo, unsigned &ColNo,
                           const SourceFileInfo *&SourceInfo);
  };


  /// RuntimeInfo - This class collects information about the currently running
  /// process.  It is created whenever the program stops execution for the
  /// debugger, and destroyed whenver execution continues.
  class RuntimeInfo {
    /// ProgInfo - This object contains static information about the program.
    ///
    ProgramInfo *ProgInfo;

    /// IP - This object contains information about the actual inferior process
    /// that we are communicating with and aggregating information from.
    const InferiorProcess &IP;

    /// CallStack - This caches information about the current stack trace of the
    /// program.  This is lazily computed as needed.
    std::vector<StackFrame> CallStack;

    /// CurrentFrame - The user can traverse the stack frame with the
    /// up/down/frame family of commands.  This index indicates the current
    /// stack frame.
    unsigned CurrentFrame;

  public:
    RuntimeInfo(ProgramInfo *PI, const InferiorProcess &ip)
      : ProgInfo(PI), IP(ip), CurrentFrame(0) {
      // Make sure that the top of stack has been materialized.  If this throws
      // an exception, something is seriously wrong and the RuntimeInfo object
      // would be unusable anyway.
      getStackFrame(0);
    }

    ProgramInfo &getProgramInfo() { return *ProgInfo; }
    const InferiorProcess &getInferiorProcess() const { return IP; }

    //===------------------------------------------------------------------===//
    // Methods for inspecting the call stack of the program.
    //

    /// getStackFrame - Materialize the specified stack frame and return it.  If
    /// the specified ID is off of the bottom of the stack, throw an exception
    /// indicating the problem.
    StackFrame &getStackFrame(unsigned ID) {
      if (ID >= CallStack.size())
        materializeFrame(ID);
      return CallStack[ID];
    }

    /// getCurrentFrame - Return the current stack frame object that the user is
    /// inspecting.
    StackFrame &getCurrentFrame() {
      assert(CallStack.size() > CurrentFrame &&
             "Must have materialized frame before making it current!");
      return CallStack[CurrentFrame];
    }

    /// getCurrentFrameIdx - Return the current frame the user is inspecting.
    ///
    unsigned getCurrentFrameIdx() const { return CurrentFrame; }

    /// setCurrentFrameIdx - Set the current frame index to the specified value.
    /// Note that the specified frame must have been materialized with
    /// getStackFrame before it can be made current.
    void setCurrentFrameIdx(unsigned Idx) {
      assert(Idx < CallStack.size() &&
             "Must materialize frame before making it current!");
      CurrentFrame = Idx;
    }
  private:
    /// materializeFrame - Create and process all frames up to and including the
    /// specified frame number.  This throws an exception if the specified frame
    /// ID is nonexistant.
    void materializeFrame(unsigned ID);
  };
}

#endif
