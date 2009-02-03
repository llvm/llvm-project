//===- SourceFile.h - Class to represent a source code file -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SourceFile class which is used to represent a single
// file of source code in the program, caching data from the file to make access
// efficient.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGGER_SOURCEFILE_H
#define LLVM_DEBUGGER_SOURCEFILE_H

#include "llvm/System/Path.h"
#include "llvm/ADT/OwningPtr.h"
#include <vector>

namespace llvm {
  class GlobalVariable;
  class MemoryBuffer;

  class SourceFile {
    /// Filename - This is the full path of the file that is loaded.
    ///
    sys::Path Filename;

    /// Descriptor - The debugging descriptor for this source file.  If there
    /// are multiple descriptors for the same file, this is just the first one
    /// encountered.
    ///
    const GlobalVariable *Descriptor;

    /// This is the memory mapping for the file so we can gain access to it.
    OwningPtr<MemoryBuffer> File;

    /// LineOffset - This vector contains a mapping from source line numbers to
    /// their offsets in the file.  This data is computed lazily, the first time
    /// it is asked for.  If there are zero elements allocated in this vector,
    /// then it has not yet been computed.
    mutable std::vector<unsigned> LineOffset;

  public:
    /// SourceFile constructor - Read in the specified source file if it exists,
    /// but do not build the LineOffsets table until it is requested.  This will
    /// NOT throw an exception if the file is not found, if there is an error
    /// reading it, or if the user cancels the operation.  Instead, it will just
    /// be an empty source file.
    SourceFile(const std::string &fn, const GlobalVariable *Desc);
    
    ~SourceFile();

    /// getDescriptor - Return the debugging decriptor for this source file.
    ///
    const GlobalVariable *getDescriptor() const { return Descriptor; }

    /// getFilename - Return the fully resolved path that this file was loaded
    /// from.
    const std::string &getFilename() const { return Filename.toString(); }

    /// getSourceLine - Given a line number, return the start and end of the
    /// line in the file.  If the line number is invalid, or if the file could
    /// not be loaded, null pointers are returned for the start and end of the
    /// file.  Note that line numbers start with 0, not 1.  This also strips off
    /// any newlines from the end of the line, to ease formatting of the text.
    void getSourceLine(unsigned LineNo, const char *&LineStart,
                       const char *&LineEnd) const;

    /// getNumLines - Return the number of lines the source file contains.
    ///
    unsigned getNumLines() const {
      if (LineOffset.empty()) calculateLineOffsets();
      return static_cast<unsigned>(LineOffset.size());
    }

  private:
    /// calculateLineOffsets - Compute the LineOffset vector for the current
    /// file.
    void calculateLineOffsets() const;
  };
} // end namespace llvm

#endif
