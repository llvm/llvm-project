//===-- RuntimeInfo.cpp - Compute and cache info about running program ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RuntimeInfo and related classes, by querying and
// cachine information from the running inferior process.
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/InferiorProcess.h"
#include "llvm/Debugger/ProgramInfo.h"
#include "llvm/Debugger/RuntimeInfo.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// StackFrame class implementation

StackFrame::StackFrame(RuntimeInfo &ri, void *ParentFrameID)
  : RI(ri), SourceInfo(0) {
  FrameID = RI.getInferiorProcess().getPreviousFrame(ParentFrameID);
  if (FrameID == 0) throw "Stack frame does not exist!";

  // Compute lazily as needed.
  FunctionDesc = 0;
}

const GlobalVariable *StackFrame::getFunctionDesc() {
  if (FunctionDesc == 0)
    FunctionDesc = RI.getInferiorProcess().getSubprogramDesc(FrameID);
  return FunctionDesc;
}

/// getSourceLocation - Return the source location that this stack frame is
/// sitting at.
void StackFrame::getSourceLocation(unsigned &lineNo, unsigned &colNo,
                                   const SourceFileInfo *&sourceInfo) {
  if (SourceInfo == 0) {
    const GlobalVariable *SourceDesc = 0;
    RI.getInferiorProcess().getFrameLocation(FrameID, LineNo,ColNo, SourceDesc);
    SourceInfo = &RI.getProgramInfo().getSourceFile(SourceDesc);
  }

  lineNo = LineNo;
  colNo = ColNo;
  sourceInfo = SourceInfo;
}

//===----------------------------------------------------------------------===//
// RuntimeInfo class implementation

/// materializeFrame - Create and process all frames up to and including the
/// specified frame number.  This throws an exception if the specified frame
/// ID is nonexistant.
void RuntimeInfo::materializeFrame(unsigned ID) {
  assert(ID >= CallStack.size() && "no need to materialize this frame!");
  void *CurFrame = 0;
  if (!CallStack.empty())
    CurFrame = CallStack.back().getFrameID();

  while (CallStack.size() <= ID) {
    CallStack.push_back(StackFrame(*this, CurFrame));
    CurFrame = CallStack.back().getFrameID();
  }
}
