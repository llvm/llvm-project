//===- SourceMgr.h - Manager for Source Buffers & Diagnostics ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SMDiagnostic and SourceMgr classes.  This
// provides a simple substrate for diagnostics, #include handling, and other low
// level things for simple parsers.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SOURCEMGR_H
#define SUPPORT_SOURCEMGR_H

#include "llvm/Support/SMLoc.h"

#include <string>
#include <vector>
#include <cassert>

namespace llvm {
  class MemoryBuffer;
  class SourceMgr;
  class SMDiagnostic;
  class Twine;
  class raw_ostream;

/// SourceMgr - This owns the files read by a parser, handles include stacks,
/// and handles diagnostic wrangling.
class SourceMgr {
public:
  /// DiagHandlerTy - Clients that want to handle their own diagnostics in a
  /// custom way can register a function pointer+context as a diagnostic
  /// handler.  It gets called each time PrintMessage is invoked.
  typedef void (*DiagHandlerTy)(const SMDiagnostic&, void *Context);
private:
  struct SrcBuffer {
    /// Buffer - The memory buffer for the file.
    MemoryBuffer *Buffer;

    /// IncludeLoc - This is the location of the parent include, or null if at
    /// the top level.
    SMLoc IncludeLoc;
  };

  /// Buffers - This is all of the buffers that we are reading from.
  std::vector<SrcBuffer> Buffers;

  // IncludeDirectories - This is the list of directories we should search for
  // include files in.
  std::vector<std::string> IncludeDirectories;

  /// LineNoCache - This is a cache for line number queries, its implementation
  /// is really private to SourceMgr.cpp.
  mutable void *LineNoCache;

  DiagHandlerTy DiagHandler;
  void *DiagContext;
  
  SourceMgr(const SourceMgr&);    // DO NOT IMPLEMENT
  void operator=(const SourceMgr&); // DO NOT IMPLEMENT
public:
  SourceMgr() : LineNoCache(0), DiagHandler(0), DiagContext(0) {}
  ~SourceMgr();

  void setIncludeDirs(const std::vector<std::string> &Dirs) {
    IncludeDirectories = Dirs;
  }

  /// setDiagHandler - Specify a diagnostic handler to be invoked every time
  /// PrintMessage is called. Ctx is passed into the handler when it is invoked.
  void setDiagHandler(DiagHandlerTy DH, void *Ctx = 0) {
    DiagHandler = DH;
    DiagContext = Ctx;
  }

  const SrcBuffer &getBufferInfo(unsigned i) const {
    assert(i < Buffers.size() && "Invalid Buffer ID!");
    return Buffers[i];
  }

  const MemoryBuffer *getMemoryBuffer(unsigned i) const {
    assert(i < Buffers.size() && "Invalid Buffer ID!");
    return Buffers[i].Buffer;
  }

  SMLoc getParentIncludeLoc(unsigned i) const {
    assert(i < Buffers.size() && "Invalid Buffer ID!");
    return Buffers[i].IncludeLoc;
  }

  /// AddNewSourceBuffer - Add a new source buffer to this source manager.  This
  /// takes ownership of the memory buffer.
  unsigned AddNewSourceBuffer(MemoryBuffer *F, SMLoc IncludeLoc) {
    SrcBuffer NB;
    NB.Buffer = F;
    NB.IncludeLoc = IncludeLoc;
    Buffers.push_back(NB);
    return Buffers.size()-1;
  }

  /// AddIncludeFile - Search for a file with the specified name in the current
  /// directory or in one of the IncludeDirs.  If no file is found, this returns
  /// ~0, otherwise it returns the buffer ID of the stacked file.
  /// The full path to the included file can be found in IncludedFile.
  unsigned AddIncludeFile(const std::string &Filename, SMLoc IncludeLoc,
                          std::string &IncludedFile);

  /// FindBufferContainingLoc - Return the ID of the buffer containing the
  /// specified location, returning -1 if not found.
  int FindBufferContainingLoc(SMLoc Loc) const;

  /// FindLineNumber - Find the line number for the specified location in the
  /// specified file.  This is not a fast method.
  unsigned FindLineNumber(SMLoc Loc, int BufferID = -1) const;

  /// PrintMessage - Emit a message about the specified location with the
  /// specified string.
  ///
  /// @param Type - If non-null, the kind of message (e.g., "error") which is
  /// prefixed to the message.
  /// @param ShowLine - Should the diagnostic show the source line.
  void PrintMessage(SMLoc Loc, const Twine &Msg, const char *Type,
                    bool ShowLine = true) const;


  /// GetMessage - Return an SMDiagnostic at the specified location with the
  /// specified string.
  ///
  /// @param Type - If non-null, the kind of message (e.g., "error") which is
  /// prefixed to the message.
  /// @param ShowLine - Should the diagnostic show the source line.
  SMDiagnostic GetMessage(SMLoc Loc,
                          const Twine &Msg, const char *Type,
                          bool ShowLine = true) const;

  /// PrintIncludeStack - Prints the names of included files and the line of the
  /// file they were included from.  A diagnostic handler can use this before
  /// printing its custom formatted message.
  ///
  /// @param IncludeLoc - The line of the include.
  /// @param OS the raw_ostream to print on.
  void PrintIncludeStack(SMLoc IncludeLoc, raw_ostream &OS) const;
};


/// SMDiagnostic - Instances of this class encapsulate one diagnostic report,
/// allowing printing to a raw_ostream as a caret diagnostic.
class SMDiagnostic {
  const SourceMgr *SM;
  SMLoc Loc;
  std::string Filename;
  int LineNo, ColumnNo;
  std::string Message, LineContents;
  unsigned ShowLine : 1;

public:
  // Null diagnostic.
  SMDiagnostic() : SM(0), LineNo(0), ColumnNo(0), ShowLine(0) {}
  // Diagnostic with no location (e.g. file not found, command line arg error).
  SMDiagnostic(const std::string &filename, const std::string &Msg)
    : SM(0), Filename(filename), LineNo(-1), ColumnNo(-1),
      Message(Msg), ShowLine(false) {}
  
  // Diagnostic with a location.
  SMDiagnostic(const SourceMgr &sm, SMLoc L, const std::string &FN,
               int Line, int Col,
               const std::string &Msg, const std::string &LineStr,
               bool showline = true)
    : SM(&sm), Loc(L), Filename(FN), LineNo(Line), ColumnNo(Col), Message(Msg),
      LineContents(LineStr), ShowLine(showline) {}

  const SourceMgr *getSourceMgr() const { return SM; }
  SMLoc getLoc() const { return Loc; }
  const std::string &getFilename() const { return Filename; }
  int getLineNo() const { return LineNo; }
  int getColumnNo() const { return ColumnNo; }
  const std::string &getMessage() const { return Message; }
  const std::string &getLineContents() const { return LineContents; }
  bool getShowLine() const { return ShowLine; }
  
  void Print(const char *ProgName, raw_ostream &S) const;
};

}  // end llvm namespace

#endif
