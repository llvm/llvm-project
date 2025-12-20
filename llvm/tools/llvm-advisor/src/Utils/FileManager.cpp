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

Expected<std::string> FileManager::createTempDir(llvm::StringRef prefix) {
  SmallString<128> tempDirPath;
  if (std::error_code ec =
          sys::fs::createUniqueDirectory(prefix, tempDirPath))
    return createStringError(ec, "Failed to create unique temporary directory");
  return std::string(tempDirPath.str());
}

Error FileManager::copyDirectory(StringRef source, StringRef dest) {
  // Normalize source path (preserve root directories)
  SmallString<128> sourceNorm = source;
  if (sourceNorm.size() > 1 && sys::path::is_separator(sourceNorm.back())) {
    sourceNorm.pop_back();
  }

  std::error_code ec;
  sys::fs::recursive_directory_iterator iter(source, ec), end;
  if (ec) return createFileError(source, ec, "Failed to iterate directory");

  // Functional handlers with minimal state capture
  const auto handleEntry = [dest](const sys::fs::directory_entry& entry,
                                 StringRef sourceNorm) -> Error {
    const auto currentPath = entry.path();
    StringRef currentPathRef(currentPath);
    SmallString<128> destPath = dest;

    // Validate and compute relative path in single pass
    const auto [valid, relative] = [&]() -> std::pair<bool, StringRef> {
      if (!currentPathRef.starts_with(sourceNorm))
        return std::make_pair(false, StringRef{});

      auto rel = currentPathRef.drop_front(sourceNorm.size());
      if (!rel.empty() && sys::path::is_separator(rel.front()))
        rel = rel.drop_front(1);
      return std::make_pair(true, rel);
    }();

    if (!valid)
      return createStringError(
          std::errc::invalid_argument,
          "Path '{}' not contained in source directory '{}'",
          currentPath.c_str(), sourceNorm.str().c_str());


    sys::path::append(destPath, relative);
    auto statusOrErr = entry.status();
    if (!statusOrErr)
      return createFileError(currentPath, statusOrErr.getError(),
                             "Failed to stat path");
    const auto status = *statusOrErr;

    // Unified directory creation for both files and directories
    const auto ensureParent = [&]() -> Error {
      const auto parent = sys::path::parent_path(destPath);
      if (!parent.empty()) {
        std::error_code createEc = sys::fs::create_directories(parent);
        if (createEc)
          return createFileError(parent, createEc, "Failed to create parent directory");
      }
      return Error::success();
    };

    // Type-specific handling with early returns
    if (status.type() == sys::fs::file_type::directory_file) {
      if (std::error_code ec = sys::fs::create_directories(destPath))
        return createFileError(destPath, ec, "Failed to create directory");
      return Error::success();
    }

    if (Error err = ensureParent()) return err;
    if (std::error_code ec = sys::fs::copy_file(currentPath, destPath))
      return createFileError(
          currentPath, ec, "Failed to copy file to '%s'", destPath.c_str());

    return Error::success();
  };

  // Declarative iteration with error propagation
  while (iter != end && !ec) {
    if (Error err = handleEntry(*iter, sourceNorm)) return err;
    iter.increment(ec);
  }

  return ec ? createFileError(source, ec, "Directory iteration failed")
            : Error::success();
}

Error FileManager::removeDirectory(llvm::StringRef path) {
  if (!sys::fs::exists(path)) return Error::success();

  std::error_code EC;
  SmallVector<std::string, 8> Dirs;
  for (sys::fs::recursive_directory_iterator I(path, EC), E; I != E && !EC;
       I.increment(EC)) {
    if (I->type() == sys::fs::file_type::directory_file) {
      Dirs.push_back(I->path());
    } else {
      if (auto E = sys::fs::remove(I->path())) {
        return createStringError(E, "Failed to remove file: " + I->path());
      }
    }
  }

  if (EC) return createStringError(EC, "Error iterating directory " + path);

  for (const auto &Dir : llvm::reverse(Dirs)) {
    if (auto E = sys::fs::remove(Dir))
      return createStringError(E, "Failed to remove directory: " + Dir);
  }

  if (auto E = sys::fs::remove(path))
    return createStringError(E,
                             "Failed to remove top-level directory: " + path);

  return Error::success();
}

SmallVector<std::string, 8> FileManager::findFiles(llvm::StringRef directory,
                                                   llvm::StringRef pattern) {
  SmallVector<std::string, 8> files;
  std::error_code EC;
  for (sys::fs::recursive_directory_iterator I(directory, EC), E; I != E && !EC;
       I.increment(EC)) {
    if (I->type() != sys::fs::file_type::directory_file) {
      StringRef filename = sys::path::filename(I->path());
      if (filename.find(pattern) != StringRef::npos)
        files.push_back(I->path());
    }
  }
  return files;
}

SmallVector<std::string, 8> FileManager::findFilesByExtension(
    llvm::StringRef directory, const SmallVector<std::string, 8> &extensions) {
  SmallVector<std::string, 8> files;
  std::error_code EC;
  for (sys::fs::recursive_directory_iterator I(directory, EC), E; I != E && !EC;
       I.increment(EC)) {
    if (I->type() != sys::fs::file_type::directory_file) {
      StringRef filepath = I->path();
      for (const auto &ext : extensions) {
        if (filepath.ends_with(ext)) {
          files.push_back(filepath.str());
          break;
        }
      }
    }
  }
  return files;
}

Error FileManager::moveFile(llvm::StringRef source, llvm::StringRef dest) {
  if (source == dest) return Error::success();

  if (sys::fs::create_directories(sys::path::parent_path(dest)))
    return createStringError(
        std::make_error_code(std::errc::io_error),
        "Failed to create parent directory for destination: " + dest);

  if (sys::fs::rename(source, dest)) {
    // If rename fails, try copy and remove
    if (sys::fs::copy_file(source, dest)) {
      return createStringError(std::make_error_code(std::errc::io_error),
                               "Failed to move file (copy failed): " + source);
    }
    if (sys::fs::remove(source)) {
      return createStringError(std::make_error_code(std::errc::io_error),
                               "Failed to move file (source removal failed): " +
                                   source);
    }
  }

  return Error::success();
}

Error FileManager::copyFile(llvm::StringRef source, llvm::StringRef dest) {
  if (source == dest) return Error::success();

  if (sys::fs::create_directories(sys::path::parent_path(dest))) {
    return createStringError(
        std::make_error_code(std::errc::io_error),
        "Failed to create parent directory for destination: " + dest);
  }

  if (sys::fs::copy_file(source, dest))
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Failed to copy file: " + source);

  return Error::success();
}

Expected<size_t> FileManager::getFileSize(llvm::StringRef path) {
  sys::fs::file_status status;
  if (auto EC = sys::fs::status(path, status)) {
    return createStringError(EC, "File not found: " + path);
  }

  return status.getSize();
}

} // namespace advisor
} // namespace llvm
