// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: keith.ray@gmail.com (Keith Ray)

#include <gtest/internal/gtest-filepath.h>
#include <gtest/internal/gtest-port.h>

#include <stdlib.h>

#ifdef _WIN32_WCE
#include <windows.h>
#elif defined(GTEST_OS_WINDOWS)
#include <direct.h>
#include <io.h>
#include <sys/stat.h>
#elif defined(GTEST_OS_SYMBIAN)
// Symbian OpenC has PATH_MAX in sys/syslimits.h
#include <sys/syslimits.h>
#include <unistd.h>
#else
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#endif  // _WIN32_WCE or _WIN32

#ifdef GTEST_OS_WINDOWS
#define GTEST_PATH_MAX_ _MAX_PATH
#elif defined(PATH_MAX)
#define GTEST_PATH_MAX_ PATH_MAX
#elif defined(_XOPEN_PATH_MAX)
#define GTEST_PATH_MAX_ _XOPEN_PATH_MAX
#else
#define GTEST_PATH_MAX_ _POSIX_PATH_MAX
#endif  // GTEST_OS_WINDOWS

#include <gtest/internal/gtest-string.h>

namespace testing {
namespace internal {

#ifdef GTEST_OS_WINDOWS
const char kPathSeparator = '\\';
const char kPathSeparatorString[] = "\\";
#ifdef _WIN32_WCE
// Windows CE doesn't have a current directory. You should not use
// the current directory in tests on Windows CE, but this at least
// provides a reasonable fallback.
const char kCurrentDirectoryString[] = "\\";
// Windows CE doesn't define INVALID_FILE_ATTRIBUTES
const DWORD kInvalidFileAttributes = 0xffffffff;
#else
const char kCurrentDirectoryString[] = ".\\";
#endif  // _WIN32_WCE
#else
const char kPathSeparator = '/';
const char kPathSeparatorString[] = "/";
const char kCurrentDirectoryString[] = "./";
#endif  // GTEST_OS_WINDOWS

// Returns the current working directory, or "" if unsuccessful.
FilePath FilePath::GetCurrentDir() {
#ifdef _WIN32_WCE
// Windows CE doesn't have a current directory, so we just return
// something reasonable.
  return FilePath(kCurrentDirectoryString);
#elif defined(GTEST_OS_WINDOWS)
  char cwd[GTEST_PATH_MAX_ + 1] = {};
  return FilePath(_getcwd(cwd, sizeof(cwd)) == NULL ? "" : cwd);
#else
  char cwd[GTEST_PATH_MAX_ + 1] = {};
  return FilePath(getcwd(cwd, sizeof(cwd)) == NULL ? "" : cwd);
#endif
}

// Returns a copy of the FilePath with the case-insensitive extension removed.
// Example: FilePath("dir/file.exe").RemoveExtension("EXE") returns
// FilePath("dir/file"). If a case-insensitive extension is not
// found, returns a copy of the original FilePath.
FilePath FilePath::RemoveExtension(const char* extension) const {
  String dot_extension(String::Format(".%s", extension));
  if (pathname_.EndsWithCaseInsensitive(dot_extension.c_str())) {
    return FilePath(String(pathname_.c_str(), pathname_.GetLength() - 4));
  }
  return *this;
}

// Returns a copy of the FilePath with the directory part removed.
// Example: FilePath("path/to/file").RemoveDirectoryName() returns
// FilePath("file"). If there is no directory part ("just_a_file"), it returns
// the FilePath unmodified. If there is no file part ("just_a_dir/") it
// returns an empty FilePath ("").
// On Windows platform, '\' is the path separator, otherwise it is '/'.
FilePath FilePath::RemoveDirectoryName() const {
  const char* const last_sep = strrchr(c_str(), kPathSeparator);
  return last_sep ? FilePath(String(last_sep + 1)) : *this;
}

// RemoveFileName returns the directory path with the filename removed.
// Example: FilePath("path/to/file").RemoveFileName() returns "path/to/".
// If the FilePath is "a_file" or "/a_file", RemoveFileName returns
// FilePath("./") or, on Windows, FilePath(".\\"). If the filepath does
// not have a file, like "just/a/dir/", it returns the FilePath unmodified.
// On Windows platform, '\' is the path separator, otherwise it is '/'.
FilePath FilePath::RemoveFileName() const {
  const char* const last_sep = strrchr(c_str(), kPathSeparator);
  return FilePath(last_sep ? String(c_str(), last_sep + 1 - c_str())
                           : String(kCurrentDirectoryString));
}

// Helper functions for naming files in a directory for xml output.

// Given directory = "dir", base_name = "test", number = 0,
// extension = "xml", returns "dir/test.xml". If number is greater
// than zero (e.g., 12), returns "dir/test_12.xml".
// On Windows platform, uses \ as the separator rather than /.
FilePath FilePath::MakeFileName(const FilePath& directory,
                                const FilePath& base_name,
                                int number,
                                const char* extension) {
  FilePath dir(directory.RemoveTrailingPathSeparator());
  if (number == 0) {
    return FilePath(String::Format("%s%c%s.%s", dir.c_str(), kPathSeparator,
                                   base_name.c_str(), extension));
  }
  return FilePath(String::Format("%s%c%s_%d.%s", dir.c_str(), kPathSeparator,
                                 base_name.c_str(), number, extension));
}

// Returns true if pathname describes something findable in the file-system,
// either a file, directory, or whatever.
bool FilePath::FileOrDirectoryExists() const {
#ifdef GTEST_OS_WINDOWS
#ifdef _WIN32_WCE
  LPCWSTR unicode = String::AnsiToUtf16(pathname_.c_str());
  const DWORD attributes = GetFileAttributes(unicode);
  delete [] unicode;
  return attributes != kInvalidFileAttributes;
#else
  struct _stat file_stat = {};
  return _stat(pathname_.c_str(), &file_stat) == 0;
#endif  // _WIN32_WCE
#else
  struct stat file_stat = {};
  return stat(pathname_.c_str(), &file_stat) == 0;
#endif  // GTEST_OS_WINDOWS
}

// Returns true if pathname describes a directory in the file-system
// that exists.
bool FilePath::DirectoryExists() const {
  bool result = false;
#ifdef GTEST_OS_WINDOWS
  // Don't strip off trailing separator if path is a root directory on
  // Windows (like "C:\\").
  const FilePath& path(IsRootDirectory() ? *this :
                                           RemoveTrailingPathSeparator());
#ifdef _WIN32_WCE
  LPCWSTR unicode = String::AnsiToUtf16(path.c_str());
  const DWORD attributes = GetFileAttributes(unicode);
  delete [] unicode;
  if ((attributes != kInvalidFileAttributes) &&
      (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
    result = true;
  }
#else
  struct _stat file_stat = {};
  result = _stat(path.c_str(), &file_stat) == 0 &&
      (_S_IFDIR & file_stat.st_mode) != 0;
#endif  // _WIN32_WCE
#else
  struct stat file_stat = {};
  result = stat(pathname_.c_str(), &file_stat) == 0 &&
      S_ISDIR(file_stat.st_mode);
#endif  // GTEST_OS_WINDOWS
  return result;
}

// Returns true if pathname describes a root directory. (Windows has one
// root directory per disk drive.)
bool FilePath::IsRootDirectory() const {
#ifdef GTEST_OS_WINDOWS
  const char* const name = pathname_.c_str();
  return pathname_.GetLength() == 3 &&
     ((name[0] >= 'a' && name[0] <= 'z') ||
      (name[0] >= 'A' && name[0] <= 'Z')) &&
     name[1] == ':' &&
     name[2] == kPathSeparator;
#else
  return pathname_ == kPathSeparatorString;
#endif
}

// Returns a pathname for a file that does not currently exist. The pathname
// will be directory/base_name.extension or
// directory/base_name_<number>.extension if directory/base_name.extension
// already exists. The number will be incremented until a pathname is found
// that does not already exist.
// Examples: 'dir/foo_test.xml' or 'dir/foo_test_1.xml'.
// There could be a race condition if two or more processes are calling this
// function at the same time -- they could both pick the same filename.
FilePath FilePath::GenerateUniqueFileName(const FilePath& directory,
                                          const FilePath& base_name,
                                          const char* extension) {
  FilePath full_pathname;
  int number = 0;
  do {
    full_pathname.Set(MakeFileName(directory, base_name, number++, extension));
  } while (full_pathname.FileOrDirectoryExists());
  return full_pathname;
}

// Returns true if FilePath ends with a path separator, which indicates that
// it is intended to represent a directory. Returns false otherwise.
// This does NOT check that a directory (or file) actually exists.
bool FilePath::IsDirectory() const {
  return pathname_.EndsWith(kPathSeparatorString);
}

// Create directories so that path exists. Returns true if successful or if
// the directories already exist; returns false if unable to create directories
// for any reason.
bool FilePath::CreateDirectoriesRecursively() const {
  if (!this->IsDirectory()) {
    return false;
  }

  if (pathname_.GetLength() == 0 || this->DirectoryExists()) {
    return true;
  }

  const FilePath parent(this->RemoveTrailingPathSeparator().RemoveFileName());
  return parent.CreateDirectoriesRecursively() && this->CreateFolder();
}

// Create the directory so that path exists. Returns true if successful or
// if the directory already exists; returns false if unable to create the
// directory for any reason, including if the parent directory does not
// exist. Not named "CreateDirectory" because that's a macro on Windows.
bool FilePath::CreateFolder() const {
#ifdef GTEST_OS_WINDOWS
#ifdef _WIN32_WCE
  FilePath removed_sep(this->RemoveTrailingPathSeparator());
  LPCWSTR unicode = String::AnsiToUtf16(removed_sep.c_str());
  int result = CreateDirectory(unicode, NULL) ? 0 : -1;
  delete [] unicode;
#else
  int result = _mkdir(pathname_.c_str());
#endif  // !WIN32_WCE
#else
  int result = mkdir(pathname_.c_str(), 0777);
#endif  // _WIN32
  if (result == -1) {
    return this->DirectoryExists();  // An error is OK if the directory exists.
  }
  return true;  // No error.
}

// If input name has a trailing separator character, remove it and return the
// name, otherwise return the name string unmodified.
// On Windows platform, uses \ as the separator, other platforms use /.
FilePath FilePath::RemoveTrailingPathSeparator() const {
  return pathname_.EndsWith(kPathSeparatorString)
      ? FilePath(String(pathname_.c_str(), pathname_.GetLength() - 1))
      : *this;
}

// Normalize removes any redundant separators that might be in the pathname.
// For example, "bar///foo" becomes "bar/foo". Does not eliminate other
// redundancies that might be in a pathname involving "." or "..".
void FilePath::Normalize() {
  if (pathname_.c_str() == NULL) {
    pathname_ = "";
    return;
  }
  const char* src = pathname_.c_str();
  char* const dest = new char[pathname_.GetLength() + 1];
  char* dest_ptr = dest;
  memset(dest_ptr, 0, pathname_.GetLength() + 1);

  while (*src != '\0') {
    *dest_ptr++ = *src;
    if (*src != kPathSeparator)
      src++;
    else
      while (*src == kPathSeparator)
        src++;
  }
  *dest_ptr = '\0';
  pathname_ = dest;
  delete[] dest;
}

}  // namespace internal
}  // namespace testing
