//===---------------- AutoLoadDylibUtils.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/AutoLoadDylibUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace orc {

#if defined(LLVM_ON_UNIX)
const char *const kEnvDelim = ":";
#elif defined(_WIN32)
const char *const kEnvDelim = ";";
#else
#error "Unknown platform (environmental delimiter)"
#endif

#if defined(LLVM_ON_UNIX)
bool Popen(const std::string &Cmd, llvm::SmallVectorImpl<char> &Buf, bool RdE) {
  if (FILE *PF = ::popen(RdE ? (Cmd + " 2>&1").c_str() : Cmd.c_str(), "r")) {
    Buf.resize(0);
    const size_t Chunk = Buf.capacity_in_bytes();
    while (true) {
      const size_t Len = Buf.size();
      Buf.resize(Len + Chunk);
      const size_t R = ::fread(&Buf[Len], sizeof(char), Chunk, PF);
      if (R < Chunk) {
        Buf.resize(Len + R);
        break;
      }
    }
    ::pclose(PF);
    return !Buf.empty();
  }
  return false;
}
#endif

std::string NormalizePath(const std::string &Path) {

  llvm::SmallString<256> Buffer;
  std::error_code EC = llvm::sys::fs::real_path(Path, Buffer, true);
  if (EC)
    return std::string();
  return std::string(Buffer.str());
}

static void LogNonExistantDirectory(StringRef Path) {
#define DEBUG_TYPE "LogNonExistantDirectory"
  LLVM_DEBUG(dbgs() << "  ignoring nonexistent directory \"" << Path << "\"\n");
#undef DEBUG_TYPE
}

bool SplitPaths(StringRef PathStr, SmallVectorImpl<StringRef> &Paths,
                SplitMode Mode, StringRef Delim, bool Verbose) {
#define DEBUG_TYPE "SplitPths"

  assert(Delim.size() && "Splitting without a delimiter");

#if defined(_WIN32)
  // Support using a ':' delimiter on Windows.
  const bool WindowsColon = Delim.equals(":");
#endif

  bool AllExisted = true;
  for (std::pair<StringRef, StringRef> Split = PathStr.split(Delim);
       !Split.second.empty(); Split = PathStr.split(Delim)) {

    if (!Split.first.empty()) {
      bool Exists = sys::fs::is_directory(Split.first);

#if defined(_WIN32)
      // Because drive letters will have a colon we have to make sure the split
      // occurs at a colon not followed by a path separator.
      if (!Exists && WindowsColon && Split.first.size() == 1) {
        // Both clang and cl.exe support '\' and '/' path separators.
        if (Split.second.front() == '\\' || Split.second.front() == '/') {
          const std::pair<StringRef, StringRef> Tmp = Split.second.split(Delim);
          // Split.first = 'C', but we want 'C:', so Tmp.first.size()+2
          Split.first = StringRef(Split.first.data(), Tmp.first.size() + 2);
          Split.second = Tmp.second;
          Exists = sys::fs::is_directory(Split.first);
        }
      }
#endif

      AllExisted = AllExisted && Exists;

      if (!Exists) {
        if (Mode == SplitMode::FailNonExistant) {
          if (Verbose) {
            // Exiting early, but still log all non-existant paths that we have
            LogNonExistantDirectory(Split.first);
            while (!Split.second.empty()) {
              Split = PathStr.split(Delim);
              if (sys::fs::is_directory(Split.first)) {
                LLVM_DEBUG(dbgs() << "  ignoring directory that exists \""
                                  << Split.first << "\"\n");
              } else
                LogNonExistantDirectory(Split.first);
              Split = Split.second.split(Delim);
            }
            if (!sys::fs::is_directory(Split.first))
              LogNonExistantDirectory(Split.first);
          }
          return false;
        } else if (Mode == SplitMode::AllowNonExistant)
          Paths.push_back(Split.first);
        else if (Verbose)
          LogNonExistantDirectory(Split.first);
      } else
        Paths.push_back(Split.first);
    }

    PathStr = Split.second;
  }

  // Trim trailing sep in case of A:B:C:D:
  if (!PathStr.empty() && PathStr.ends_with(Delim))
    PathStr = PathStr.substr(0, PathStr.size() - Delim.size());

  if (!PathStr.empty()) {
    if (!sys::fs::is_directory(PathStr)) {
      AllExisted = false;
      if (Mode == SplitMode::AllowNonExistant)
        Paths.push_back(PathStr);
      else if (Verbose)
        LogNonExistantDirectory(PathStr);
    } else
      Paths.push_back(PathStr);
  }

  return AllExisted;

#undef DEBUG_TYPE
}

bool GetSystemLibraryPaths(llvm::SmallVectorImpl<std::string> &Paths) {
#if defined(__APPLE__) || defined(__CYGWIN__)
  Paths.push_back("/usr/local/lib/");
  Paths.push_back("/usr/X11R6/lib/");
  Paths.push_back("/usr/lib/");
  Paths.push_back("/lib/");

#ifndef __APPLE__
  Paths.push_back("/lib/x86_64-linux-gnu/");
  Paths.push_back("/usr/local/lib64/");
  Paths.push_back("/usr/lib64/");
  Paths.push_back("/lib64/");
#endif
#elif defined(LLVM_ON_UNIX)
  llvm::SmallString<1024> Buf;
  Popen("LD_DEBUG=libs LD_PRELOAD=DOESNOTEXIST ls", Buf, true);
  const llvm::StringRef Result = Buf.str();

  const std::size_t NPos = std::string::npos;
  const std::size_t LD = Result.find("(LD_LIBRARY_PATH)");
  std::size_t From = Result.find("search path=", LD == NPos ? 0 : LD);
  if (From != NPos) {
    std::size_t To = Result.find("(system search path)", From);
    if (To != NPos) {
      From += 12;
      while (To > From && isspace(Result[To - 1]))
        --To;
      std::string SysPath = Result.substr(From, To - From).str();
      SysPath.erase(std::remove_if(SysPath.begin(), SysPath.end(), ::isspace),
                    SysPath.end());

      llvm::SmallVector<llvm::StringRef, 10> CurPaths;
      SplitPaths(SysPath, CurPaths);
      for (const auto &Path : CurPaths)
        Paths.push_back(Path.str());
    }
  }
#endif
  return true;
}

} // namespace orc
} // namespace llvm