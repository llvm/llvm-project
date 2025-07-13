#include "FileManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <system_error>

namespace llvm {
namespace advisor {

Expected<std::string> FileManager::createTempDir(const std::string &prefix) {
  SmallString<128> tempDirPath;
  if (std::error_code ec =
          sys::fs::createUniqueDirectory(prefix, tempDirPath)) {
    return createStringError(ec, "Failed to create unique temporary directory");
  }
  return std::string(tempDirPath.str());
}

Error FileManager::copyDirectory(const std::string &source,
                                 const std::string &dest) {
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

Error FileManager::removeDirectory(const std::string &path) {
  if (!sys::fs::exists(path)) {
    return Error::success();
  }

  std::error_code EC;
  std::vector<std::string> Dirs;
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

std::vector<std::string> FileManager::findFiles(const std::string &directory,
                                                const std::string &pattern) {
  std::vector<std::string> files;
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

std::vector<std::string>
FileManager::findFilesByExtension(const std::string &directory,
                                  const std::vector<std::string> &extensions) {
  std::vector<std::string> files;
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

Error FileManager::moveFile(const std::string &source,
                            const std::string &dest) {
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

Error FileManager::copyFile(const std::string &source,
                            const std::string &dest) {
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

Expected<size_t> FileManager::getFileSize(const std::string &path) {
  sys::fs::file_status status;
  if (auto EC = sys::fs::status(path, status)) {
    return createStringError(EC, "File not found: " + path);
  }

  return status.getSize();
}

} // namespace advisor
} // namespace llvm