//===------------------- FileManager.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the FileManager code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "FileManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace advisor {

Expected<std::string> FileManager::createTempDir(llvm::StringRef Prefix) {
  SmallString<128> TempDirPath;
  if (std::error_code Ec = sys::fs::createUniqueDirectory(Prefix, TempDirPath))
    return createStringError(Ec, "Failed to create unique temporary directory");
  return std::string(TempDirPath.str());
}

Error FileManager::copyDirectory(StringRef Source, StringRef Dest) {
  // Normalize source path (preserve root directories)
  SmallString<128> SourceNorm = Source;
  if (SourceNorm.size() > 1 && sys::path::is_separator(SourceNorm.back()))
    SourceNorm.pop_back();

  std::error_code Ec;
  sys::fs::recursive_directory_iterator Iter(Source, Ec), End;
  if (Ec)
    return createFileError(Source, Ec, "Failed to iterate directory");

  // Functional handlers with minimal state capture
  const auto HandleEntry = [Dest](const sys::fs::directory_entry &Entry,
                                  StringRef SourceNorm) -> Error {
    const auto CurrentPath = Entry.path();
    StringRef CurrentPathRef(CurrentPath);
    SmallString<128> DestPath = Dest;

    // Validate and compute relative path in single pass
    const auto [valid, relative] = [&]() -> std::pair<bool, StringRef> {
      if (!CurrentPathRef.starts_with(SourceNorm))
        return std::make_pair(false, StringRef{});

      auto Rel = CurrentPathRef.drop_front(SourceNorm.size());
      if (!Rel.empty() && sys::path::is_separator(Rel.front()))
        Rel = Rel.drop_front(1);
      return std::make_pair(true, Rel);
    }();

    if (!valid)
      return createStringError(
          std::errc::invalid_argument,
          "Path '{}' not contained in source directory '{}'",
          CurrentPath.c_str(), SourceNorm.str().c_str());

    sys::path::append(DestPath, relative);
    auto StatusOrErr = Entry.status();
    if (!StatusOrErr)
      return createFileError(CurrentPath, StatusOrErr.getError(),
                             "Failed to stat path");
    const auto Status = *StatusOrErr;

    // Unified directory creation for both files and directories
    const auto EnsureParent = [&]() -> Error {
      const auto Parent = sys::path::parent_path(DestPath);
      if (!Parent.empty()) {
        std::error_code CreateEc = sys::fs::create_directories(Parent);
        if (CreateEc)
          return createFileError(Parent, CreateEc,
                                 "Failed to create parent directory");
      }
      return Error::success();
    };

    // Type-specific handling with early returns
    if (Status.type() == sys::fs::file_type::directory_file) {
      if (std::error_code Ec = sys::fs::create_directories(DestPath))
        return createFileError(DestPath, Ec, "Failed to create directory");
      return Error::success();
    }

    if (Error Err = EnsureParent())
      return Err;
    if (std::error_code Ec = sys::fs::copy_file(CurrentPath, DestPath))
      return createFileError(CurrentPath, Ec, "Failed to copy file to '%s'",
                             DestPath.c_str());

    return Error::success();
  };

  // Declarative iteration with error propagation
  while (Iter != End && !Ec) {
    if (Error Err = HandleEntry(*Iter, SourceNorm))
      return Err;
    Iter.increment(Ec);
  }

  return Ec ? createFileError(Source, Ec, "Directory iteration failed")
            : Error::success();
}

Error FileManager::removeDirectory(llvm::StringRef Path) {
  if (!sys::fs::exists(Path))
    return Error::success();

  std::error_code EC;
  SmallVector<std::string, 8> Dirs;
  for (sys::fs::recursive_directory_iterator I(Path, EC), E; I != E && !EC;
       I.increment(EC)) {
    if (I->type() == sys::fs::file_type::directory_file) {
      Dirs.push_back(I->path());
    } else {
      if (auto E = sys::fs::remove(I->path()))
        return createStringError(E, "Failed to remove file: " + I->path());
    }
  }

  if (EC)
    return createStringError(EC, "Error iterating directory " + Path);

  for (const auto &Dir : llvm::reverse(Dirs)) {
    if (auto E = sys::fs::remove(Dir))
      return createStringError(E, "Failed to remove directory: " + Dir);
  }

  if (auto E = sys::fs::remove(Path))
    return createStringError(E,
                             "Failed to remove top-level directory: " + Path);

  return Error::success();
}

SmallVector<std::string, 8> FileManager::findFiles(llvm::StringRef Directory,
                                                   llvm::StringRef Pattern) {
  SmallVector<std::string, 8> Files;
  std::error_code EC;
  for (sys::fs::recursive_directory_iterator I(Directory, EC), E; I != E && !EC;
       I.increment(EC)) {
    if (I->type() != sys::fs::file_type::directory_file) {
      StringRef Filename = sys::path::filename(I->path());
      if (Filename.find(Pattern) != StringRef::npos)
        Files.push_back(I->path());
    }
  }
  return Files;
}

SmallVector<std::string, 8> FileManager::findFilesByExtension(
    llvm::StringRef Directory, const SmallVector<std::string, 8> &Extensions) {
  SmallVector<std::string, 8> Files;
  std::error_code EC;
  for (sys::fs::recursive_directory_iterator I(Directory, EC), E; I != E && !EC;
       I.increment(EC)) {
    if (I->type() != sys::fs::file_type::directory_file) {
      StringRef Filepath = I->path();
      for (const auto &Ext : Extensions) {
        if (Filepath.ends_with(Ext)) {
          Files.push_back(Filepath.str());
          break;
        }
      }
    }
  }
  return Files;
}

Error FileManager::moveFile(llvm::StringRef Source, llvm::StringRef Dest) {
  if (Source == Dest)
    return Error::success();

  llvm::StringRef Parent = sys::path::parent_path(Dest);
  if (!Parent.empty()) {
    if (sys::fs::create_directories(Parent))
      return createStringError(
          std::make_error_code(std::errc::io_error),
          "Failed to create parent directory for destination: " + Dest);
  }

  if (sys::fs::rename(Source, Dest)) {
    // If rename fails, try copy and remove
    if (sys::fs::copy_file(Source, Dest)) {
      return createStringError(std::make_error_code(std::errc::io_error),
                               "Failed to move file (copy failed): " + Source);
    }
    if (sys::fs::remove(Source)) {
      return createStringError(std::make_error_code(std::errc::io_error),
                               "Failed to move file (source removal failed): " +
                                   Source);
    }
  }

  return Error::success();
}

Error FileManager::copyFile(llvm::StringRef Source, llvm::StringRef Dest) {
  if (Source == Dest)
    return Error::success();

  llvm::StringRef Parent = sys::path::parent_path(Dest);
  if (!Parent.empty()) {
    if (sys::fs::create_directories(Parent)) {
      return createStringError(
          std::make_error_code(std::errc::io_error),
          "Failed to create parent directory for destination: " + Dest);
    }
  }

  if (sys::fs::copy_file(Source, Dest))
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Failed to copy file: " + Source);

  return Error::success();
}

Expected<size_t> FileManager::getFileSize(llvm::StringRef Path) {
  sys::fs::file_status Status;
  if (auto EC = sys::fs::status(Path, Status))
    return createStringError(EC, "File not found: " + Path);

  return Status.getSize();
}

} // namespace advisor
} // namespace llvm
