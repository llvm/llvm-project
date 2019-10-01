//===- llvm/Support/FileUtilities.h - File System Utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a family of utility functions which are useful for doing
// various things with files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILEUTILITIES_H
#define LLVM_SUPPORT_FILEUTILITIES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {

  /// DiffFilesWithTolerance - Compare the two files specified, returning 0 if
  /// the files match, 1 if they are different, and 2 if there is a file error.
  /// This function allows you to specify an absolute and relative FP error that
  /// is allowed to exist.  If you specify a string to fill in for the error
  /// option, it will set the string to an error message if an error occurs, or
  /// if the files are different.
  ///
  int DiffFilesWithTolerance(StringRef FileA,
                             StringRef FileB,
                             double AbsTol, double RelTol,
                             std::string *Error = nullptr);


  /// FileRemover - This class is a simple object meant to be stack allocated.
  /// If an exception is thrown from a region, the object removes the filename
  /// specified (if deleteIt is true).
  ///
  class FileRemover {
    SmallString<128> Filename;
    bool DeleteIt;
  public:
    FileRemover() : DeleteIt(false) {}

    explicit FileRemover(const Twine& filename, bool deleteIt = true)
      : DeleteIt(deleteIt) {
      filename.toVector(Filename);
    }

    ~FileRemover() {
      if (DeleteIt) {
        // Ignore problems deleting the file.
        sys::fs::remove(Filename);
      }
    }

    /// setFile - Give ownership of the file to the FileRemover so it will
    /// be removed when the object is destroyed.  If the FileRemover already
    /// had ownership of a file, remove it first.
    void setFile(const Twine& filename, bool deleteIt = true) {
      if (DeleteIt) {
        // Ignore problems deleting the file.
        sys::fs::remove(Filename);
      }

      Filename.clear();
      filename.toVector(Filename);
      DeleteIt = deleteIt;
    }

    /// releaseFile - Take ownership of the file away from the FileRemover so it
    /// will not be removed when the object is destroyed.
    void releaseFile() { DeleteIt = false; }
  };

  enum class atomic_write_error {
    failed_to_create_uniq_file = 0,
    output_stream_error,
    failed_to_rename_temp_file
  };

  class AtomicFileWriteError : public llvm::ErrorInfo<AtomicFileWriteError> {
  public:
    AtomicFileWriteError(atomic_write_error Error) : Error(Error) {}

    void log(raw_ostream &OS) const override;

    const atomic_write_error Error;
    static char ID;

  private:
    // Users are not expected to use error_code.
    std::error_code convertToErrorCode() const override {
      return llvm::inconvertibleErrorCode();
    }
  };

  // atomic_write_error + whatever the Writer can return

  /// Creates a unique file with name according to the given \p TempPathModel,
  /// writes content of \p Buffer to the file and renames it to \p FinalPath.
  ///
  /// \returns \c AtomicFileWriteError in case of error.
  llvm::Error writeFileAtomically(StringRef TempPathModel, StringRef FinalPath,
                                  StringRef Buffer);

  llvm::Error
  writeFileAtomically(StringRef TempPathModel, StringRef FinalPath,
                      std::function<llvm::Error(llvm::raw_ostream &)> Writer);
} // End llvm namespace

#endif
