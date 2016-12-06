//===- FuzzerIOWindows.cpp - IO utils for Windows. ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// IO functions implementation for Windows.
//===----------------------------------------------------------------------===//

#include "FuzzerDefs.h"
#if LIBFUZZER_WINDOWS
#include "FuzzerExtFunctions.h"
#include "FuzzerIO.h"
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <io.h>
#include <iterator>
#include <sys/types.h>
#include <sys/stat.h>
#include <windows.h>

namespace fuzzer {

static bool IsFile(const std::string &Path, const DWORD &FileAttributes) {

  if (FileAttributes & FILE_ATTRIBUTE_NORMAL)
    return true;

  if (FileAttributes & FILE_ATTRIBUTE_DIRECTORY)
    return false;

  HANDLE FileHandle(
      CreateFileA(Path.c_str(), 0, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                  FILE_FLAG_BACKUP_SEMANTICS, 0));

  if (FileHandle == INVALID_HANDLE_VALUE) {
    Printf("CreateFileA() failed for \"%s\" (Error code: %lu).\n", Path.c_str(),
        GetLastError());
    return false;
  }

  DWORD FileType = GetFileType(FileHandle);

  if (FileType == FILE_TYPE_UNKNOWN) {
    Printf("GetFileType() failed for \"%s\" (Error code: %lu).\n", Path.c_str(),
        GetLastError());
    CloseHandle(FileHandle);
    return false;
  }

  if (FileType != FILE_TYPE_DISK) {
    CloseHandle(FileHandle);
    return false;
  }

  CloseHandle(FileHandle);
  return true;
}

bool IsFile(const std::string &Path) {
  DWORD Att = GetFileAttributesA(Path.c_str());

  if (Att == INVALID_FILE_ATTRIBUTES) {
    Printf("GetFileAttributesA() failed for \"%s\" (Error code: %lu).\n",
        Path.c_str(), GetLastError());
    return false;
  }

  return IsFile(Path, Att);
}

void ListFilesInDirRecursive(const std::string &Dir, long *Epoch,
                             std::vector<std::string> *V, bool TopDir) {
  auto E = GetEpoch(Dir);
  if (Epoch)
    if (E && *Epoch >= E) return;

  std::string Path(Dir);
  assert(!Path.empty());
  if (Path.back() != '\\')
      Path.push_back('\\');
  Path.push_back('*');

  // Get the first directory entry.
  WIN32_FIND_DATAA FindInfo;
  HANDLE FindHandle(FindFirstFileA(Path.c_str(), &FindInfo));
  if (FindHandle == INVALID_HANDLE_VALUE)
  {
    Printf("No file found in: %s.\n", Dir.c_str());
    return;
  }

  do {
    std::string FileName = DirPlusFile(Dir, FindInfo.cFileName);

    if (FindInfo.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
      size_t FilenameLen = strlen(FindInfo.cFileName);
      if ((FilenameLen == 1 && FindInfo.cFileName[0] == '.') ||
          (FilenameLen == 2 && FindInfo.cFileName[0] == '.' &&
                               FindInfo.cFileName[1] == '.'))
        continue;

      ListFilesInDirRecursive(FileName, Epoch, V, false);
    }
    else if (IsFile(FileName, FindInfo.dwFileAttributes))
      V->push_back(FileName);
  } while (FindNextFileA(FindHandle, &FindInfo));

  DWORD LastError = GetLastError();
  if (LastError != ERROR_NO_MORE_FILES)
    Printf("FindNextFileA failed (Error code: %lu).\n", LastError);

  FindClose(FindHandle);

  if (Epoch && TopDir)
    *Epoch = E;
}

char GetSeparator() {
  return '\\';
}

FILE* OpenFile(int Fd, const char* Mode) {
  return _fdopen(Fd, Mode);
}

int CloseFile(int Fd) {
  return _close(Fd);
}

int DuplicateFile(int Fd) {
  return _dup(Fd);
}

void DeleteFile(const std::string &Path) {
  _unlink(Path.c_str());
}

std::string DirName(const std::string &FileName) {
  assert(0 && "Unimplemented");
}

}  // namespace fuzzer
#endif // LIBFUZZER_WINDOWS
