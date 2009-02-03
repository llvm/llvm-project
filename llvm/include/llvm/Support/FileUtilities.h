//===- llvm/Support/FileUtilities.h - File System Utilities -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a family of utility functions which are useful for doing
// various things with files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILEUTILITIES_H
#define LLVM_SUPPORT_FILEUTILITIES_H

#include "llvm/System/Path.h"

namespace llvm {

  /// DiffFilesWithTolerance - Compare the two files specified, returning 0 if
  /// the files match, 1 if they are different, and 2 if there is a file error.
  /// This function allows you to specify an absolete and relative FP error that
  /// is allowed to exist.  If you specify a string to fill in for the error
  /// option, it will set the string to an error message if an error occurs, or
  /// if the files are different.
  ///
  int DiffFilesWithTolerance(const sys::PathWithStatus &FileA, 
                             const sys::PathWithStatus &FileB,
                             double AbsTol, double RelTol,
                             std::string *Error = 0);


  /// FileRemover - This class is a simple object meant to be stack allocated.
  /// If an exception is thrown from a region, the object removes the filename
  /// specified (if deleteIt is true).
  ///
  class FileRemover {
    sys::Path Filename;
    bool DeleteIt;
  public:
    explicit FileRemover(const sys::Path &filename, bool deleteIt = true)
      : Filename(filename), DeleteIt(deleteIt) {}

    ~FileRemover() {
      if (DeleteIt) {
        // Ignore problems deleting the file.
        Filename.eraseFromDisk();
      }
    }

    /// releaseFile - Take ownership of the file away from the FileRemover so it
    /// will not be removed when the object is destroyed.
    void releaseFile() { DeleteIt = false; }
  };
} // End llvm namespace

#endif
