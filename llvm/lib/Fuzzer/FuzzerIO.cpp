//===- FuzzerIO.cpp - IO utils. -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// IO functions.
//===----------------------------------------------------------------------===//
#include "FuzzerInternal.h"
#include <iterator>
#include <fstream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdio>

namespace fuzzer {

bool IsFile(const std::string &Path) {
  struct stat St;
  if (stat(Path.c_str(), &St))
    return false;
  return S_ISREG(St.st_mode);
}

static long GetEpoch(const std::string &Path) {
  struct stat St;
  if (stat(Path.c_str(), &St))
    return 0;  // Can't stat, be conservative.
  return St.st_mtime;
}

static std::vector<std::string> ListFilesInDir(const std::string &Dir,
                                               long *Epoch) {
  std::vector<std::string> V;
  if (Epoch) {
    auto E = GetEpoch(Dir);
    if (E && *Epoch >= E) return V;
    *Epoch = E;
  }
  DIR *D = opendir(Dir.c_str());
  if (!D) {
    Printf("No such directory: %s; exiting\n", Dir.c_str());
    exit(1);
  }
  while (auto E = readdir(D)) {
    if (E->d_type == DT_REG || E->d_type == DT_LNK)
      V.push_back(E->d_name);
  }
  closedir(D);
  return V;
}

Unit FileToVector(const std::string &Path) {
  std::ifstream T(Path);
  if (!T) {
    Printf("No such directory: %s; exiting\n", Path.c_str());
    exit(1);
  }
  return Unit((std::istreambuf_iterator<char>(T)),
              std::istreambuf_iterator<char>());
}

std::string FileToString(const std::string &Path) {
  std::ifstream T(Path);
  return std::string((std::istreambuf_iterator<char>(T)),
                     std::istreambuf_iterator<char>());
}

void CopyFileToErr(const std::string &Path) {
  Printf("%s", FileToString(Path).c_str());
}

void WriteToFile(const Unit &U, const std::string &Path) {
  // Use raw C interface because this function may be called from a sig handler.
  FILE *Out = fopen(Path.c_str(), "w");
  if (!Out) return;
  fwrite(U.data(), sizeof(U[0]), U.size(), Out);
  fclose(Out);
}

void ReadDirToVectorOfUnits(const char *Path, std::vector<Unit> *V,
                            long *Epoch) {
  long E = Epoch ? *Epoch : 0;
  for (auto &X : ListFilesInDir(Path, Epoch)) {
    auto FilePath = DirPlusFile(Path, X);
    if (Epoch && GetEpoch(FilePath) < E) continue;
    V->push_back(FileToVector(FilePath));
  }
}

std::string DirPlusFile(const std::string &DirPath,
                        const std::string &FileName) {
  return DirPath + "/" + FileName;
}

void Printf(const char *Fmt, ...) {
  va_list ap;
  va_start(ap, Fmt);
  vfprintf(stderr, Fmt, ap);
  va_end(ap);
}

}  // namespace fuzzer
