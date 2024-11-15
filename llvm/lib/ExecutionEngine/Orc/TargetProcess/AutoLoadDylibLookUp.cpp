//===---------------- AutoLoadDylibLookup.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/AutoLoadDylibLookup.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#if defined(_WIN32)
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Support/Endian.h"
#endif

#include <fstream>
#include <sys/stat.h>
#include <system_error>

static bool GetSystemLibraryPaths(llvm::SmallVectorImpl<std::string> &Paths) {
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

static std::string NormalizePath(const std::string &Path) {

  llvm::SmallString<256> Buffer;
  std::error_code EC = llvm::sys::fs::real_path(Path, Buffer, true);
  if (EC)
    return std::string();
  return std::string(Buffer.str());
}

namespace llvm {
namespace orc {

void LogNonExistantDirectory(StringRef Path) {
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

AutoLoadDynamicLibraryLookup ::AutoLoadDynamicLibraryLookup() {
  const SmallVector<const char *, 10> kSysLibraryEnv = {
    "LD_LIBRARY_PATH",
#if __APPLE__
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
  /*
  "DYLD_VERSIONED_LIBRARY_PATH",
  "DYLD_FRAMEWORK_PATH",
  "DYLD_FALLBACK_FRAMEWORK_PATH",
  "DYLD_VERSIONED_FRAMEWORK_PATH",
  */
#elif defined(_WIN32)
    "PATH",
#endif
  };

  // Behaviour is to not add paths that don't exist...In an interpreted env
  // does this make sense? Path could pop into existance at any time.
  for (const char *Var : kSysLibraryEnv) {
    if (const char *Env = ::getenv(Var)) {
      SmallVector<StringRef, 10> CurPaths;
      SplitPaths(Env, CurPaths, SplitMode::PruneNonExistant, kEnvDelim);
      for (const auto &Path : CurPaths)
        addSearchPath(Path);
    }
  }

  // $CWD is the last user path searched.
  addSearchPath(".");

  SmallVector<std::string, 64> SysPaths;
  GetSystemLibraryPaths(SysPaths);

  for (const std::string &P : SysPaths)
    addSearchPath(P, /*IsUser*/ false);
}
///\returns substitution of pattern in the front of original with replacement
/// Example: substFront("@rpath/abc", "@rpath/", "/tmp") -> "/tmp/abc"
static std::string substFront(StringRef original, StringRef pattern,
                              StringRef replacement) {
  if (!original.starts_with_insensitive(pattern))
    return original.str();
  SmallString<512> result(replacement);
  result.append(original.drop_front(pattern.size()));
  return result.str().str();
}

///\returns substitution of all known linker variables in \c original
static std::string substAll(StringRef original, StringRef libLoader) {

  // Handle substitutions (MacOS):
  // @rpath - This function does not substitute @rpath, becouse
  //          this variable is already handled by lookupLibrary where
  //          @rpath is replaced with all paths from RPATH one by one.
  // @executable_path - Main program path.
  // @loader_path - Loader library (or main program) path.
  //
  // Handle substitutions (Linux):
  // https://man7.org/linux/man-pages/man8/ld.so.8.html
  // $origin - Loader library (or main program) path.
  // $lib - lib lib64
  // $platform - x86_64 AT_PLATFORM

  std::string result;
#ifdef __APPLE__
  SmallString<512> mainExecutablePath(
      llvm::sys::fs::getMainExecutable(nullptr, nullptr));
  llvm::sys::path::remove_filename(mainExecutablePath);
  SmallString<512> loaderPath;
  if (libLoader.empty()) {
    loaderPath = mainExecutablePath;
  } else {
    loaderPath = libLoader.str();
    llvm::sys::path::remove_filename(loaderPath);
  }

  result = substFront(original, "@executable_path", mainExecutablePath);
  result = substFront(result, "@loader_path", loaderPath);
  return result;
#else
  SmallString<512> loaderPath;
  if (libLoader.empty()) {
    loaderPath = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  } else {
    loaderPath = libLoader.str();
  }
  llvm::sys::path::remove_filename(loaderPath);

  result = substFront(original, "$origin", loaderPath);
  // result = substFront(result, "$lib", true?"lib":"lib64");
  // result = substFront(result, "$platform", "x86_64");
  return result;
#endif
}

std::string AutoLoadDynamicLibraryLookup::lookupLibInPaths(
    StringRef libStem, SmallVector<llvm::StringRef, 2> RPath /*={}*/,
    SmallVector<llvm::StringRef, 2> RunPath /*={}*/,
    StringRef libLoader /*=""*/) const {
#define DEBUG_TYPE "Dyld::lookupLibInPaths"

  LLVM_DEBUG(dbgs() << "Dyld::lookupLibInPaths" << libStem.str()
                    << ", ..., libLoader=" << libLoader << "\n");

  // Lookup priority is: RPATH, LD_LIBRARY_PATH/SearchPaths, RUNPATH
  // See: https://en.wikipedia.org/wiki/Rpath
  // See: https://amir.rachum.com/blog/2016/09/17/shared-libraries/

  LLVM_DEBUG({
    dbgs() << "Dyld::lookupLibInPaths: \n";
    dbgs() << ":: RPATH\n";
    for (auto Info : RPath) {
      dbgs() << ":::: " << Info.str() << "\n";
    }
    dbgs() << ":: SearchPaths (LD_LIBRARY_PATH, etc...)\n";
    for (auto Info : getSearchPaths()) {
      dbgs() << ":::: " << Info.Path
             << ", user=" << (Info.IsUser ? "true" : "false") << "\n";
    }
    dbgs() << ":: RUNPATH\n";
    for (auto Info : RunPath) {
      dbgs() << ":::: " << Info.str() << "\n";
    }
  });

  SmallString<512> ThisPath;
  // RPATH
  for (auto Info : RPath) {
    ThisPath = substAll(Info, libLoader);
    llvm::sys::path::append(ThisPath, libStem);
    // to absolute path?
    LLVM_DEBUG(dbgs() << "## Try: " << ThisPath);
    if (isSharedLibrary(ThisPath.str())) {
      LLVM_DEBUG(dbgs() << " ... Found (in RPATH)!\n");
      return ThisPath.str().str();
    }
  }
  // SearchPaths
  for (const SearchPathInfo &Info : SearchPaths) {
    ThisPath = Info.Path;
    llvm::sys::path::append(ThisPath, libStem);
    // to absolute path?
    LLVM_DEBUG(dbgs() << "## Try: " << ThisPath);
    if (isSharedLibrary(ThisPath.str())) {
      LLVM_DEBUG(dbgs() << " ... Found (in SearchPaths)!\n");
      return ThisPath.str().str();
    }
  }
  // RUNPATH
  for (auto Info : RunPath) {
    ThisPath = substAll(Info, libLoader);
    llvm::sys::path::append(ThisPath, libStem);
    // to absolute path?
    LLVM_DEBUG(dbgs() << "## Try: " << ThisPath);
    if (isSharedLibrary(ThisPath.str())) {
      LLVM_DEBUG(dbgs() << " ... Found (in RUNPATH)!\n");
      return ThisPath.str().str();
    }
  }

  LLVM_DEBUG(dbgs() << "## NotFound!!!\n");

  return "";

#undef DEBUG_TYPE
}

std::string AutoLoadDynamicLibraryLookup::lookupLibMaybeAddExt(
    StringRef libStem, SmallVector<llvm::StringRef, 2> RPath /*={}*/,
    SmallVector<llvm::StringRef, 2> RunPath /*={}*/,
    StringRef libLoader /*=""*/) const {
#define DEBUG_TYPE "Dyld::lookupLibMaybeAddExt:"

  using namespace llvm::sys;

  LLVM_DEBUG(dbgs() << "Dyld::lookupLibMaybeAddExt: " << libStem.str()
                    << ", ..., libLoader=" << libLoader << "\n");

  std::string foundDyLib = lookupLibInPaths(libStem, RPath, RunPath, libLoader);

  if (foundDyLib.empty()) {
    // Add DyLib extension:
    SmallString<512> filenameWithExt(libStem);
#if defined(LLVM_ON_UNIX)
#ifdef __APPLE__
    SmallString<512>::iterator IStemEnd = filenameWithExt.end() - 1;
#endif
    static const char *DyLibExt = ".so";
#elif defined(_WIN32)
    static const char *DyLibExt = ".dll";
#else
#error "Unsupported platform."
#endif
    filenameWithExt += DyLibExt;
    foundDyLib = lookupLibInPaths(filenameWithExt, RPath, RunPath, libLoader);
#ifdef __APPLE__
    if (foundDyLib.empty()) {
      filenameWithExt.erase(IStemEnd + 1, filenameWithExt.end());
      filenameWithExt += ".dylib";
      foundDyLib = lookupLibInPaths(filenameWithExt, RPath, RunPath, libLoader);
    }
#endif
  }

  if (foundDyLib.empty())
    return std::string();

  // get canonical path name and check if already loaded
  const std::string Path = NormalizePath(foundDyLib);
  if (Path.empty()) {
    LLVM_DEBUG(
        dbgs() << "AutoLoadDynamicLibraryLookup::lookupLibMaybeAddExt(): "
               << "error getting real (canonical) path of library "
               << foundDyLib << '\n');
    return foundDyLib;
  }
  return Path;

#undef DEBUG_TYPE
}

std::string AutoLoadDynamicLibraryLookup::normalizePath(StringRef path) {
#define DEBUG_TYPE "Dyld::normalizePath:"
  // Make the path canonical if the file exists.
  const std::string Path = path.str();
  struct stat buffer;
  if (::stat(Path.c_str(), &buffer) != 0)
    return std::string();

  const std::string NPath = NormalizePath(Path);
  if (NPath.empty())
    LLVM_DEBUG(dbgs() << "Could not normalize: '" << Path << "'");
  return NPath;
#undef DEBUG_TYPE
}

std::string RPathToStr2(SmallVector<StringRef, 2> V) {
  std::string result;
  for (auto item : V)
    result += item.str() + ",";
  if (!result.empty())
    result.pop_back();
  return result;
}

std::string AutoLoadDynamicLibraryLookup::lookupLibrary(
    StringRef libStem, SmallVector<llvm::StringRef, 2> RPath /*={}*/,
    SmallVector<llvm::StringRef, 2> RunPath /*={}*/,
    StringRef libLoader /*=""*/, bool variateLibStem /*=true*/) const {
#define DEBUG_TYPE "Dyld::lookupLibrary:"
  LLVM_DEBUG(dbgs() << "Dyld::lookupLibrary: " << libStem.str() << ", "
                    << RPathToStr2(RPath) << ", " << RPathToStr2(RunPath)
                    << ", " << libLoader.str() << "\n");
  if (libStem.empty())
    return std::string();

  // If it is an absolute path, don't try iterate over the paths.
  if (llvm::sys::path::is_absolute(libStem)) {
    if (isSharedLibrary(libStem))
      return normalizePath(libStem);

    LLVM_DEBUG(dbgs() << "Dyld::lookupLibrary: '" << libStem.str() << "'"
                      << "is not a shared library\n");
    return std::string();
  }

  // Subst all known linker variables ($origin, @rpath, etc.)
#ifdef __APPLE__
  // On MacOS @rpath is preplaced by all paths in RPATH one by one.
  if (libStem.starts_with_insensitive("@rpath")) {
    for (auto &P : RPath) {
      std::string result = substFront(libStem, "@rpath", P);
      if (isSharedLibrary(result))
        return normalizePath(result);
    }
  } else {
#endif
    std::string result = substAll(libStem, libLoader);
    if (isSharedLibrary(result))
      return normalizePath(result);
#ifdef __APPLE__
  }
#endif

  // Expand libStem with paths, extensions, etc.
  std::string foundName;
  if (variateLibStem) {
    foundName = lookupLibMaybeAddExt(libStem, RPath, RunPath, libLoader);
    if (foundName.empty()) {
      StringRef libStemName = llvm::sys::path::filename(libStem);
      if (!libStemName.starts_with("lib")) {
        // try with "lib" prefix:
        foundName = lookupLibMaybeAddExt(
            libStem.str().insert(libStem.size() - libStemName.size(), "lib"),
            RPath, RunPath, libLoader);
      }
    }
  } else {
    foundName = lookupLibInPaths(libStem, RPath, RunPath, libLoader);
  }

  if (!foundName.empty())
    return NormalizePath(foundName);

  return std::string();
#undef DEBUG_TYPE
}

void AutoLoadDynamicLibraryLookup::addLoadedLib(StringRef lib) {
  LoadedLibraries.insert(lib);
}

bool AutoLoadDynamicLibraryLookup::isLibraryLoaded(StringRef fullPath) const {
  std::string canonPath = normalizePath(fullPath);
  if (LoadedLibraries.find(canonPath) != LoadedLibraries.end())
    return true;
  return false;
}

void AutoLoadDynamicLibraryLookup::dump(
    llvm::raw_ostream *S /*= nullptr*/) const {
  llvm::raw_ostream &OS = S ? *S : llvm::outs();

  // FIXME: print in a stable order the contents of SearchPaths
  for (const auto &Info : getSearchPaths()) {
    if (!Info.IsUser)
      OS << "[system] ";
    OS << Info.Path.c_str() << "\n";
  }
}

#if defined(_WIN32)
static bool IsDLL(llvm::StringRef headers) {
  using namespace llvm::support::endian;

  uint32_t headeroffset = read32le(headers.data() + 0x3c);
  auto peheader = headers.substr(headeroffset, 24);
  if (peheader.size() != 24) {
    return false;
  }
  // Read Characteristics from the coff header
  uint32_t characteristics = read16le(peheader.data() + 22);
  return (characteristics & llvm::COFF::IMAGE_FILE_DLL) != 0;
}
#endif

bool AutoLoadDynamicLibraryLookup::isSharedLibrary(StringRef libFullPath,
                                                   bool *exists /*=0*/) {
  using namespace llvm;

  auto filetype = sys::fs::get_file_type(libFullPath, /*Follow*/ true);
  if (filetype != sys::fs::file_type::regular_file) {
    if (exists) {
      // get_file_type returns status_error also in case of file_not_found.
      *exists = filetype != sys::fs::file_type::status_error;
    }
    return false;
  }

  // Do not use the identify_magic overload taking a path: It will open the
  // file and then mmap its contents, possibly causing bus errors when another
  // process truncates the file while we are trying to read it. Instead just
  // read the first 1024 bytes, which should be enough for identify_magic to
  // do its work.
  // TODO: Fix the code upstream and consider going back to calling the
  // convenience function after a future LLVM upgrade.
  std::string path = libFullPath.str();
  std::ifstream in(path, std::ios::binary);
  char header[1024] = {0};
  in.read(header, sizeof(header));
  if (in.fail()) {
    if (exists)
      *exists = false;
    return false;
  }

  StringRef headerStr(header, in.gcount());
  file_magic Magic = identify_magic(headerStr);

  bool result =
#ifdef __APPLE__
      (Magic == file_magic::macho_fixed_virtual_memory_shared_lib ||
       Magic == file_magic::macho_dynamically_linked_shared_lib ||
       Magic == file_magic::macho_dynamically_linked_shared_lib_stub ||
       Magic == file_magic::macho_universal_binary)
#elif defined(LLVM_ON_UNIX)
#ifdef __CYGWIN__
      (Magic == file_magic::pecoff_executable)
#else
      (Magic == file_magic::elf_shared_object)
#endif
#elif defined(_WIN32)
      // We should only include dll libraries without including executables,
      // object code and others...
      (Magic == file_magic::pecoff_executable && IsDLL(headerStr))
#else
#error "Unsupported platform."
#endif
      ;

  return result;
}

} // end namespace orc
} // end namespace llvm
