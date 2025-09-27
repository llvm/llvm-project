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
          sys::fs::createUniqueDirectory(prefix, tempDirPath)) {
    return createStringError(ec, "Failed to create unique temporary directory");
  }
  return tempDirPath.str().str();
}

Error FileManager::copyDirectory(llvm::StringRef source, llvm::StringRef dest) {
  std::error_code EC;

  SmallString<128> sourcePathNorm(source);
  // Remove trailing slash manually if present
  if (sourcePathNorm.ends_with("/") && sourcePathNorm.size() > 1) {
    sourcePathNorm.pop_back();
  }

  for (sys::fs::recursive_directory_iterator I(source, EC), E; I != E && !EC;
       I.increment(EC)) {
    StringRef currentPath = I->path();
    SmallString<128> destPath(dest);

    StringRef relativePath = currentPath;
    if (!relativePath.consume_front(sourcePathNorm)) {
      return createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Path '" + currentPath.str() + "' not in source dir '" + source +
              "'");
    }
    // Remove leading slash manually if present
    if (relativePath.starts_with("/")) {
      relativePath = relativePath.drop_front(1);
    }

    sys::path::append(destPath, relativePath);

    if (sys::fs::is_directory(currentPath)) {
      if (sys::fs::create_directories(destPath)) {
        return createStringError(std::make_error_code(std::errc::io_error),
                                 "Failed to create directory: " +
                                     destPath.str().str());
      }
    } else {
      if (sys::fs::create_directories(sys::path::parent_path(destPath))) {
        return createStringError(std::make_error_code(std::errc::io_error),
                                 "Failed to create parent directory for: " +
                                     destPath.str().str());
      }
      if (sys::fs::copy_file(currentPath, destPath)) {
        return createStringError(std::make_error_code(std::errc::io_error),
                                 "Failed to copy file: " + currentPath.str());
      }
    }
  }

  if (EC) {
    return createStringError(EC, "Failed to iterate directory: " + source);
  }

  return Error::success();
}

Error FileManager::removeDirectory(llvm::StringRef path) {
  if (!sys::fs::exists(path)) {
    return Error::success();
  }

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

  if (EC) {
    return createStringError(EC, "Error iterating directory " + path);
  }

  for (const auto &Dir : llvm::reverse(Dirs)) {
    if (auto E = sys::fs::remove(Dir)) {
      return createStringError(E, "Failed to remove directory: " + Dir);
    }
  }

  if (auto E = sys::fs::remove(path)) {
    return createStringError(E,
                             "Failed to remove top-level directory: " + path);
  }

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
      if (filename.find(pattern) != StringRef::npos) {
        files.push_back(I->path());
      }
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
  if (source == dest) {
    return Error::success();
  }

  if (sys::fs::create_directories(sys::path::parent_path(dest))) {
    return createStringError(
        std::make_error_code(std::errc::io_error),
        "Failed to create parent directory for destination: " + dest);
  }

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
  if (source == dest) {
    return Error::success();
  }

  if (sys::fs::create_directories(sys::path::parent_path(dest))) {
    return createStringError(
        std::make_error_code(std::errc::io_error),
        "Failed to create parent directory for destination: " + dest);
  }

  if (sys::fs::copy_file(source, dest)) {
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Failed to copy file: " + source);
  }

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
