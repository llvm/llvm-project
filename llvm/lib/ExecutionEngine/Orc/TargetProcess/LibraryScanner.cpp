//===- LibraryScanner.cpp - Provide Library Scanning Implementation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/LibraryScanner.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/LibraryResolver.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#ifdef LLVM_ON_UNIX
#include <sys/stat.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

#ifdef __APPLE__
#include <sys/stat.h>
#undef LC_LOAD_DYLIB
#undef LC_RPATH
#endif // __APPLE__

#define DEBUG_TYPE "orc-scanner"

namespace llvm::orc {

void handleError(Error Err, StringRef context = "") {
  consumeError(handleErrors(std::move(Err), [&](const ErrorInfoBase &EIB) {
    dbgs() << "LLVM Error";
    if (!context.empty())
      dbgs() << " [" << context << "]";
    dbgs() << ": " << EIB.message() << "\n";
  }));
}

bool ObjectFileLoader::isArchitectureCompatible(const object::ObjectFile &Obj) {
  Triple HostTriple(sys::getDefaultTargetTriple());
  Triple ObjTriple = Obj.makeTriple();

  LLVM_DEBUG({
    dbgs() << "Host triple: " << HostTriple.str()
           << ", Object triple: " << ObjTriple.str() << "\n";
  });

  if (ObjTriple.getArch() != Triple::UnknownArch &&
      HostTriple.getArch() != ObjTriple.getArch())
    return false;

  if (ObjTriple.getOS() != Triple::UnknownOS &&
      HostTriple.getOS() != ObjTriple.getOS())
    return false;

  if (ObjTriple.getEnvironment() != Triple::UnknownEnvironment &&
      HostTriple.getEnvironment() != Triple::UnknownEnvironment &&
      HostTriple.getEnvironment() != ObjTriple.getEnvironment())
    return false;

  return true;
}

Expected<object::OwningBinary<object::ObjectFile>>
ObjectFileLoader::loadObjectFileWithOwnership(StringRef FilePath) {
  LLVM_DEBUG(dbgs() << "ObjectFileLoader: Attempting to open file " << FilePath
                    << "\n";);
  auto BinOrErr = object::createBinary(FilePath);
  if (!BinOrErr) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: Failed to open file " << FilePath
                      << "\n";);
    return BinOrErr.takeError();
  }

  LLVM_DEBUG(dbgs() << "ObjectFileLoader: Successfully opened file " << FilePath
                    << "\n";);

  auto OwningBin = BinOrErr->takeBinary();
  object::Binary *Bin = OwningBin.first.get();

  if (Bin->isArchive()) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: File is an archive, not supported: "
                      << FilePath << "\n";);
    return createStringError(std::errc::invalid_argument,
                             "Archive files are not supported: %s",
                             FilePath.str().c_str());
  }

#if defined(__APPLE__)
  if (auto *UB = dyn_cast<object::MachOUniversalBinary>(Bin)) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: Detected Mach-O universal binary: "
                      << FilePath << "\n";);
    for (auto ObjForArch : UB->objects()) {
      auto ObjOrErr = ObjForArch.getAsObjectFile();
      if (!ObjOrErr) {
        LLVM_DEBUG(
            dbgs()
                << "ObjectFileLoader: Skipping invalid architecture slice\n";);

        consumeError(ObjOrErr.takeError());
        continue;
      }

      std::unique_ptr<object::ObjectFile> Obj = std::move(ObjOrErr.get());
      if (isArchitectureCompatible(*Obj)) {
        LLVM_DEBUG(
            dbgs() << "ObjectFileLoader: Found compatible object slice\n";);

        return object::OwningBinary<object::ObjectFile>(
            std::move(Obj), std::move(OwningBin.second));

      } else {
        LLVM_DEBUG(dbgs() << "ObjectFileLoader: Incompatible architecture "
                             "slice skipped\n";);
      }
    }
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: No compatible slices found in "
                         "universal binary\n";);
    return createStringError(inconvertibleErrorCode(),
                             "No compatible object found in fat binary: %s",
                             FilePath.str().c_str());
  }
#endif

  auto ObjOrErr =
      object::ObjectFile::createObjectFile(Bin->getMemoryBufferRef());
  if (!ObjOrErr) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: Failed to create object file\n";);
    return ObjOrErr.takeError();
  }
  LLVM_DEBUG(dbgs() << "ObjectFileLoader: Detected object file\n";);

  std::unique_ptr<object::ObjectFile> Obj = std::move(*ObjOrErr);
  if (!isArchitectureCompatible(*Obj)) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: Incompatible architecture: "
                      << FilePath << "\n";);
    return createStringError(inconvertibleErrorCode(),
                             "Incompatible object file: %s",
                             FilePath.str().c_str());
  }

  LLVM_DEBUG(dbgs() << "ObjectFileLoader: Object file is compatible\n";);

  return object::OwningBinary<object::ObjectFile>(std::move(Obj),
                                                  std::move(OwningBin.second));
}

template <class ELFT>
bool isELFSharedLibrary(const object::ELFFile<ELFT> &ELFObj) {
  if (ELFObj.getHeader().e_type != ELF::ET_DYN)
    return false;

  auto PHOrErr = ELFObj.program_headers();
  if (!PHOrErr) {
    consumeError(PHOrErr.takeError());
    return true;
  }

  for (auto Phdr : *PHOrErr) {
    if (Phdr.p_type == ELF::PT_INTERP)
      return false;
  }

  return true;
}

bool isSharedLibraryObject(object::ObjectFile &Obj) {
  if (Obj.isELF()) {
    if (auto *ELF32LE = dyn_cast<object::ELF32LEObjectFile>(&Obj))
      return isELFSharedLibrary(ELF32LE->getELFFile());
    if (auto *ELF64LE = dyn_cast<object::ELF64LEObjectFile>(&Obj))
      return isELFSharedLibrary(ELF64LE->getELFFile());
    if (auto *ELF32BE = dyn_cast<object::ELF32BEObjectFile>(&Obj))
      return isELFSharedLibrary(ELF32BE->getELFFile());
    if (auto *ELF64BE = dyn_cast<object::ELF64BEObjectFile>(&Obj))
      return isELFSharedLibrary(ELF64BE->getELFFile());
  } else if (Obj.isMachO()) {
    const object::MachOObjectFile *MachO =
        dyn_cast<object::MachOObjectFile>(&Obj);
    if (!MachO) {
      LLVM_DEBUG(dbgs() << "Failed to cast to MachOObjectFile.\n";);
      return false;
    }
    LLVM_DEBUG({
      bool Result =
          MachO->getHeader().filetype == MachO::HeaderFileType::MH_DYLIB;
      dbgs() << "Mach-O filetype: " << MachO->getHeader().filetype
             << " (MH_DYLIB == " << MachO::HeaderFileType::MH_DYLIB
             << "), shared: " << Result << "\n";
    });

    return MachO->getHeader().filetype == MachO::HeaderFileType::MH_DYLIB;
  } else if (Obj.isCOFF()) {
    const object::COFFObjectFile *coff = dyn_cast<object::COFFObjectFile>(&Obj);
    if (!coff)
      return false;
    return coff->getCharacteristics() & COFF::IMAGE_FILE_DLL;
  } else {
    LLVM_DEBUG(dbgs() << "Binary is not an ObjectFile.\n";);
  }

  return false;
}

bool DylibPathValidator::isSharedLibrary(StringRef Path) {
  LLVM_DEBUG(dbgs() << "Checking if path is a shared library: " << Path
                    << "\n";);

  auto FileType = sys::fs::get_file_type(Path, /*Follow*/ true);
  if (FileType != sys::fs::file_type::regular_file) {
    LLVM_DEBUG(dbgs() << "File type is not a regular file for path: " << Path
                      << "\n";);
    return false;
  }

  file_magic MagicCode;
  identify_magic(Path, MagicCode);

  // Skip archives.
  if (MagicCode == file_magic::archive)
    return false;

  // Universal binary handling.
#if defined(__APPLE__)
  if (MagicCode == file_magic::macho_universal_binary) {
    ObjectFileLoader ObjLoader(Path);
    auto ObjOrErr = ObjLoader.getObjectFile();
    if (!ObjOrErr) {
      consumeError(ObjOrErr.takeError());
      return false;
    }
    return isSharedLibraryObject(ObjOrErr.get());
  }
#endif

  // Object file inspection for PE/COFF, ELF, and Mach-O
  bool NeedsObjectInspection =
#if defined(_WIN32)
      (MagicCode == file_magic::pecoff_executable);
#elif defined(__APPLE__)
      (MagicCode == file_magic::macho_fixed_virtual_memory_shared_lib ||
       MagicCode == file_magic::macho_dynamically_linked_shared_lib ||
       MagicCode == file_magic::macho_dynamically_linked_shared_lib_stub);
#elif defined(LLVM_ON_UNIX)
#ifdef __CYGWIN__
      (MagicCode == file_magic::pecoff_executable);
#else
      (MagicCode == file_magic::elf_shared_object);
#endif
#else
#error "Unsupported platform."
#endif

  if (NeedsObjectInspection) {
    ObjectFileLoader ObjLoader(Path);
    auto ObjOrErr = ObjLoader.getObjectFile();
    if (!ObjOrErr) {
      consumeError(ObjOrErr.takeError());
      return false;
    }
    return isSharedLibraryObject(ObjOrErr.get());
  }

  LLVM_DEBUG(dbgs() << "Path is not identified as a shared library: " << Path
                    << "\n";);
  return false;
}

void DylibSubstitutor::configure(StringRef LoaderPath) {
  SmallString<512> ExecPath(sys::fs::getMainExecutable(nullptr, nullptr));
  sys::path::remove_filename(ExecPath);

  SmallString<512> LoaderDir;
  if (LoaderPath.empty()) {
    LoaderDir = ExecPath;
  } else {
    LoaderDir = LoaderPath.str();
    if (!sys::fs::is_directory(LoaderPath))
      sys::path::remove_filename(LoaderDir);
  }

#ifdef __APPLE__
  Placeholders["@loader_path"] = std::string(LoaderDir);
  Placeholders["@executable_path"] = std::string(ExecPath);
#else
  Placeholders["$origin"] = std::string(LoaderDir);
#endif
}

std::optional<std::string>
SearchPathResolver::resolve(StringRef Stem, const DylibSubstitutor &Subst,
                            DylibPathValidator &Validator) const {
  for (const auto &SP : Paths) {
    std::string Base = Subst.substitute(SP);

    SmallString<512> FullPath(Base);
    if (!PlaceholderPrefix.empty() &&
        Stem.starts_with_insensitive(PlaceholderPrefix))
      FullPath.append(Stem.drop_front(PlaceholderPrefix.size()));
    else
      sys::path::append(FullPath, Stem);

    LLVM_DEBUG(dbgs() << "SearchPathResolver::resolve FullPath = " << FullPath
                      << "\n";);

    if (auto Valid = Validator.validate(FullPath.str()))
      return Valid;
  }

  return std::nullopt;
}

std::optional<std::string>
DylibResolverImpl::tryWithExtensions(StringRef LibStem) const {
  LLVM_DEBUG(dbgs() << "tryWithExtensions: baseName = " << LibStem << "\n";);
  SmallVector<SmallString<256>, 8> Candidates;

  // Add extensions by platform
#if defined(__APPLE__)
  Candidates.emplace_back(LibStem);
  Candidates.back() += ".dylib";
#elif defined(_WIN32)
  Candidates.emplace_back(LibStem);
  Candidates.back() += ".dll";
#else
  Candidates.emplace_back(LibStem);
  Candidates.back() += ".so";
#endif

  // Optionally try "lib" prefix if not already there
  StringRef FileName = sys::path::filename(LibStem);
  StringRef Base = sys::path::parent_path(LibStem);
  if (!FileName.starts_with("lib")) {
    SmallString<256> WithPrefix(Base);
    if (!WithPrefix.empty())
      sys::path::append(WithPrefix, ""); // ensure separator if needed
    WithPrefix += "lib";
    WithPrefix += FileName;

#if defined(__APPLE__)
    WithPrefix += ".dylib";
#elif defined(_WIN32)
    WithPrefix += ".dll";
#else
    WithPrefix += ".so";
#endif

    Candidates.push_back(std::move(WithPrefix));
  }

  LLVM_DEBUG({
    dbgs() << "  Candidates to try:\n";
    for (const auto &C : Candidates)
      dbgs() << "    " << C << "\n";
  });

  // Try all variants using tryAllPaths
  for (const auto &Name : Candidates) {

    LLVM_DEBUG(dbgs() << "  Trying candidate: " << Name << "\n";);

    for (const auto &R : Resolvers) {
      if (auto Res = R.resolve(Name, Substitutor, Validator))
        return Res;
    }
  }

  LLVM_DEBUG(dbgs() << "  -> No candidate Resolved.\n";);

  return std::nullopt;
}

std::optional<std::string>
DylibResolverImpl::resolve(StringRef LibStem, bool VariateLibStem) const {
  LLVM_DEBUG(dbgs() << "Resolving library stem: " << LibStem << "\n";);

  // If it is an absolute path, don't try iterate over the paths.
  if (sys::path::is_absolute(LibStem)) {
    LLVM_DEBUG(dbgs() << "  -> Absolute path detected.\n";);
    return Validator.validate(LibStem);
  }

  if (!LibStem.starts_with_insensitive("@rpath")) {
    if (auto norm = Validator.validate(Substitutor.substitute(LibStem))) {
      LLVM_DEBUG(dbgs() << "  -> Resolved after substitution: " << *norm
                        << "\n";);

      return norm;
    }
  }

  for (const auto &R : Resolvers) {
    LLVM_DEBUG(dbgs() << "  -> Resolving via search path ... \n";);
    if (auto Result = R.resolve(LibStem, Substitutor, Validator)) {
      LLVM_DEBUG(dbgs() << "  -> Resolved via search path: " << *Result
                        << "\n";);

      return Result;
    }
  }

  // Expand libStem with paths, extensions, etc.
  // std::string foundName;
  if (VariateLibStem) {
    LLVM_DEBUG(dbgs() << "  -> Trying with extensions...\n";);

    if (auto Norm = tryWithExtensions(LibStem)) {
      LLVM_DEBUG(dbgs() << "  -> Resolved via tryWithExtensions: " << *Norm
                        << "\n";);

      return Norm;
    }
  }

  LLVM_DEBUG(dbgs() << "  -> Could not resolve: " << LibStem << "\n";);

  return std::nullopt;
}

#ifndef _WIN32
mode_t PathResolver::lstatCached(StringRef Path) {
  // If already cached - retun cached result
  if (auto Cache = LibPathCache->read_lstat(Path))
    return *Cache;

  // Not cached: perform lstat and store
  struct stat buf{};
  mode_t st_mode = (lstat(Path.str().c_str(), &buf) == -1) ? 0 : buf.st_mode;

  LibPathCache->insert_lstat(Path, st_mode);

  return st_mode;
}

std::optional<std::string> PathResolver::readlinkCached(StringRef Path) {
  // If already cached - retun cached result
  if (auto Cache = LibPathCache->read_link(Path))
    return Cache;

  // If result not in cache - call system function and cache result
  char buf[PATH_MAX];
  ssize_t len;
  if ((len = readlink(Path.str().c_str(), buf, sizeof(buf))) != -1) {
    buf[len] = '\0';
    std::string s(buf);
    LibPathCache->insert_link(Path, s);
    return s;
  }
  return std::nullopt;
}

void createComponent(StringRef Path, StringRef BasePath, bool BaseIsResolved,
                     SmallVector<StringRef, 16> &Component) {
  StringRef Separator = sys::path::get_separator();
  if (!BaseIsResolved) {
    if (Path[0] == '~' &&
        (Path.size() == 1 || sys::path::is_separator(Path[1]))) {
      static SmallString<128> HomeP;
      if (HomeP.str().empty())
        sys::path::home_directory(HomeP);
      StringRef(HomeP).split(Component, Separator, /*MaxSplit*/ -1,
                             /*KeepEmpty*/ false);
    } else if (BasePath.empty()) {
      static SmallString<256> CurrentPath;
      if (CurrentPath.str().empty())
        sys::fs::current_path(CurrentPath);
      StringRef(CurrentPath)
          .split(Component, Separator, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
    } else {
      BasePath.split(Component, Separator, /*MaxSplit*/ -1,
                     /*KeepEmpty*/ false);
    }
  }

  Path.split(Component, Separator, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
}

void normalizePathSegments(SmallVector<StringRef, 16> &PathParts) {
  SmallVector<StringRef, 16> NormalizedPath;
  for (auto &Part : PathParts) {
    if (Part == ".") {
      continue;
    } else if (Part == "..") {
      if (!NormalizedPath.empty() && NormalizedPath.back() != "..") {
        NormalizedPath.pop_back();
      } else {
        NormalizedPath.push_back("..");
      }
    } else {
      NormalizedPath.push_back(Part);
    }
  }
  PathParts.swap(NormalizedPath);
}
#endif

std::optional<std::string> PathResolver::realpathCached(StringRef Path,
                                                        std::error_code &EC,
                                                        StringRef Base,
                                                        bool BaseIsResolved,
                                                        long SymLoopLevel) {
  EC.clear();

  if (Path.empty()) {
    EC = std::make_error_code(std::errc::no_such_file_or_directory);
    LLVM_DEBUG(dbgs() << "PathResolver::realpathCached: Empty path\n";);

    return std::nullopt;
  }

  if (SymLoopLevel <= 0) {
    EC = std::make_error_code(std::errc::too_many_symbolic_link_levels);
    LLVM_DEBUG(
        dbgs() << "PathResolver::realpathCached: Too many Symlink levels: "
               << Path << "\n";);

    return std::nullopt;
  }

  // If already cached - retun cached result
  bool isRelative = sys::path::is_relative(Path);
  if (!isRelative) {
    if (auto Cached = LibPathCache->read_realpath(Path)) {
      EC = Cached->ErrnoCode;
      if (EC) {
        LLVM_DEBUG(dbgs() << "PathResolver::realpathCached: Cached (error) for "
                          << Path << "\n";);
      } else {
        LLVM_DEBUG(
            dbgs() << "PathResolver::realpathCached: Cached (success) for "
                   << Path << " => " << Cached->canonicalPath << "\n";);
      }
      return Cached->canonicalPath.empty()
                 ? std::nullopt
                 : std::make_optional(Cached->canonicalPath);
    }
  }

  LLVM_DEBUG(dbgs() << "PathResolver::realpathCached: Resolving path: " << Path
                    << "\n";);

  // If result not in cache - call system function and cache result

  StringRef Separator(sys::path::get_separator());
  SmallString<256> Resolved(Separator);
#ifndef _WIN32
  SmallVector<StringRef, 16> Components;

  if (isRelative) {
    if (BaseIsResolved) {
      Resolved.assign(Base);
      LLVM_DEBUG(dbgs() << "  Using Resolved base: " << Base << "\n";);
    }
    createComponent(Path, Base, BaseIsResolved, Components);
  } else {
    Path.split(Components, Separator, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
  }

  normalizePathSegments(Components);
  LLVM_DEBUG({
    for (auto &C : Components)
      dbgs() << " " << C << " ";

    dbgs() << "\n";
  });

  // Handle path list items
  for (const auto &Component : Components) {
    if (Component == ".")
      continue;
    if (Component == "..") {
      // collapse "a/b/../c" to "a/c"
      size_t S = Resolved.rfind(Separator);
      if (S != llvm::StringRef::npos)
        Resolved.resize(S);
      if (Resolved.empty())
        Resolved = Separator;
      continue;
    }

    size_t oldSize = Resolved.size();
    sys::path::append(Resolved, Component);
    const char *ResolvedPath = Resolved.c_str();
    LLVM_DEBUG(dbgs() << "  Processing Component: " << Component << " => "
                      << ResolvedPath << "\n";);
    mode_t st_mode = lstatCached(ResolvedPath);

    if (S_ISLNK(st_mode)) {
      LLVM_DEBUG(dbgs() << "    Found symlink: " << ResolvedPath << "\n";);

      auto SymlinkOpt = readlinkCached(ResolvedPath);
      if (!SymlinkOpt) {
        EC = std::make_error_code(std::errc::no_such_file_or_directory);
        LibPathCache->insert_realpath(Path, LibraryPathCache::PathInfo{"", EC});
        LLVM_DEBUG(dbgs() << "    Failed to read symlink: " << ResolvedPath
                          << "\n";);

        return std::nullopt;
      }

      StringRef Symlink = *SymlinkOpt;
      LLVM_DEBUG(dbgs() << "    Symlink points to: " << Symlink << "\n";);

      std::string resolvedBase = "";
      if (sys::path::is_relative(Symlink)) {
        Resolved.resize(oldSize);
        resolvedBase = Resolved.str().str();
      }

      auto RealSymlink =
          realpathCached(Symlink, EC, resolvedBase,
                         /*BaseIsResolved=*/true, SymLoopLevel - 1);
      if (!RealSymlink) {
        LibPathCache->insert_realpath(Path, LibraryPathCache::PathInfo{"", EC});
        LLVM_DEBUG(dbgs() << "    Failed to resolve symlink target: " << Symlink
                          << "\n";);

        return std::nullopt;
      }

      Resolved.assign(*RealSymlink);
      LLVM_DEBUG(dbgs() << "    Symlink Resolved to: " << Resolved << "\n";);

    } else if (st_mode == 0) {
      EC = std::make_error_code(std::errc::no_such_file_or_directory);
      LibPathCache->insert_realpath(Path, LibraryPathCache::PathInfo{"", EC});
      LLVM_DEBUG(dbgs() << "    Component does not exist: " << ResolvedPath
                        << "\n";);

      return std::nullopt;
    }
  }
#else
  sys::fs::real_path(Path, Resolved); // Windows fallback
#endif

  std::string Canonical = Resolved.str().str();
  {
    LibPathCache->insert_realpath(Path, LibraryPathCache::PathInfo{
                                            Canonical,
                                            std::error_code() // success
                                        });
  }
  LLVM_DEBUG(dbgs() << "PathResolver::realpathCached: Final Resolved: " << Path
                    << " => " << Canonical << "\n";);
  return Canonical;
}

void LibraryScanHelper::addBasePath(const std::string &Path, PathType K) {
  std::error_code EC;
  std::string Canon = resolveCanonical(Path, EC);
  if (EC) {
    LLVM_DEBUG(
        dbgs()
            << "LibraryScanHelper::addBasePath: Failed to canonicalize path: "
            << Path << "\n";);
    return;
  }
  std::unique_lock<std::shared_mutex> Lock(Mtx);
  if (LibSearchPaths.count(Canon)) {
    LLVM_DEBUG(dbgs() << "LibraryScanHelper::addBasePath: Already added: "
                      << Canon << "\n";);
    return;
  }
  K = K == PathType::Unknown ? classifyKind(Canon) : K;
  auto SP = std::make_shared<LibrarySearchPath>(Canon, K);
  LibSearchPaths[Canon] = SP;

  if (K == PathType::User) {
    LLVM_DEBUG(dbgs() << "LibraryScanHelper::addBasePath: Added User path: "
                      << Canon << "\n";);
    UnscannedUsr.push_back(StringRef(SP->BasePath));
  } else {
    LLVM_DEBUG(dbgs() << "LibraryScanHelper::addBasePath: Added System path: "
                      << Canon << "\n";);
    UnscannedSys.push_back(StringRef(SP->BasePath));
  }
}

std::vector<std::shared_ptr<LibrarySearchPath>>
LibraryScanHelper::getNextBatch(PathType K, size_t BatchSize) {
  std::vector<std::shared_ptr<LibrarySearchPath>> Result;
  auto &Queue = (K == PathType::User) ? UnscannedUsr : UnscannedSys;

  std::unique_lock<std::shared_mutex> Lock(Mtx);

  while (!Queue.empty() && (BatchSize == 0 || Result.size() < BatchSize)) {
    StringRef Base = Queue.front();
    auto It = LibSearchPaths.find(Base);
    if (It != LibSearchPaths.end()) {
      auto &SP = It->second;
      ScanState Expected = ScanState::NotScanned;
      if (SP->State.compare_exchange_strong(Expected, ScanState::Scanning)) {
        Result.push_back(SP);
      }
    }
    Queue.pop_front();
  }

  return Result;
}

bool LibraryScanHelper::isTrackedBasePath(StringRef Path) const {
  std::error_code EC;
  std::string Canon = resolveCanonical(Path, EC);
  if (EC)
    return false;

  std::shared_lock<std::shared_mutex> Lock(Mtx);
  return LibSearchPaths.count(Canon) > 0;
}

bool LibraryScanHelper::leftToScan(PathType K) const {
  std::shared_lock<std::shared_mutex> Lock(Mtx);
  for (const auto &KV : LibSearchPaths) {
    const auto &SP = KV.second;
    if (SP->Kind == K && SP->State == ScanState::NotScanned)
      return true;
  }
  return false;
}

void LibraryScanHelper::resetToScan() {
  std::shared_lock<std::shared_mutex> Lock(Mtx);

  for (auto &[_, SP] : LibSearchPaths) {
    ScanState Expected = ScanState::Scanned;

    if (!SP->State.compare_exchange_strong(Expected, ScanState::NotScanned))
      continue;

    auto &TargetList =
        (SP->Kind == PathType::User) ? UnscannedUsr : UnscannedSys;
    TargetList.emplace_back(SP->BasePath);
  }
}

std::vector<std::shared_ptr<LibrarySearchPath>>
LibraryScanHelper::getAllUnits() const {
  std::shared_lock<std::shared_mutex> Lock(Mtx);
  std::vector<std::shared_ptr<LibrarySearchPath>> Result;
  Result.reserve(LibSearchPaths.size());
  for (const auto &[_, SP] : LibSearchPaths) {
    Result.push_back(SP);
  }
  return Result;
}

std::string LibraryScanHelper::resolveCanonical(StringRef Path,
                                                std::error_code &EC) const {
  auto Canon = LibPathResolver->resolve(Path, EC);
  return EC ? Path.str() : *Canon;
}

PathType LibraryScanHelper::classifyKind(StringRef Path) const {
  // Detect home directory
  const char *Home = getenv("HOME");
  if (Home && Path.find(Home) == 0)
    return PathType::User;

  static const std::array<std::string, 5> UserPrefixes = {
      "/usr/local",    // often used by users for manual installs
      "/opt/homebrew", // common on macOS
      "/opt/local",    // MacPorts
      "/home",         // Linux home dirs
      "/Users",        // macOS user dirs
  };

  for (const auto &Prefix : UserPrefixes) {
    if (Path.find(Prefix) == 0)
      return PathType::User;
  }

  return PathType::System;
}

Expected<LibraryDepsInfo> parseMachODeps(const object::MachOObjectFile &Obj) {
  LibraryDepsInfo Libdeps;
  LLVM_DEBUG(dbgs() << "Parsing Mach-O dependencies...\n";);
  for (const auto &Command : Obj.load_commands()) {
    switch (Command.C.cmd) {
    case MachO::LC_LOAD_DYLIB: {
      MachO::dylib_command dylibCmd = Obj.getDylibIDLoadCommand(Command);
      const char *name = Command.Ptr + dylibCmd.dylib.name;
      Libdeps.addDep(name);
      LLVM_DEBUG(dbgs() << "  Found LC_LOAD_DYLIB: " << name << "\n";);
    } break;
    case MachO::LC_LOAD_WEAK_DYLIB:
    case MachO::LC_REEXPORT_DYLIB:
    case MachO::LC_LOAD_UPWARD_DYLIB:
    case MachO::LC_LAZY_LOAD_DYLIB:
      break;
    case MachO::LC_RPATH: {
      // Extract RPATH
      MachO::rpath_command rpathCmd = Obj.getRpathCommand(Command);
      const char *rpath = Command.Ptr + rpathCmd.path;
      LLVM_DEBUG(dbgs() << "  Found LC_RPATH: " << rpath << "\n";);

      SmallVector<StringRef, 4> RawPaths;
      SplitString(StringRef(rpath), RawPaths,
                  sys::EnvPathSeparator == ':' ? ":" : ";");

      for (const auto &raw : RawPaths) {
        Libdeps.addRPath(raw.str()); // Convert to std::string
        LLVM_DEBUG(dbgs() << "    Parsed RPATH entry: " << raw << "\n";);
      }
      break;
    }
    }
  }
  return Libdeps;
}

template <class ELFT>
static Expected<StringRef> getDynamicStrTab(const object::ELFFile<ELFT> &Elf) {
  auto DynamicEntriesOrError = Elf.dynamicEntries();
  if (!DynamicEntriesOrError)
    return DynamicEntriesOrError.takeError();

  for (const typename ELFT::Dyn &Dyn : *DynamicEntriesOrError) {
    if (Dyn.d_tag == ELF::DT_STRTAB) {
      auto MappedAddrOrError = Elf.toMappedAddr(Dyn.getPtr());
      if (!MappedAddrOrError)
        return MappedAddrOrError.takeError();
      return StringRef(reinterpret_cast<const char *>(*MappedAddrOrError));
    }
  }

  // If the dynamic segment is not present, we fall back on the sections.
  auto SectionsOrError = Elf.sections();
  if (!SectionsOrError)
    return SectionsOrError.takeError();

  for (const typename ELFT::Shdr &Sec : *SectionsOrError) {
    if (Sec.sh_type == ELF::SHT_DYNSYM)
      return Elf.getStringTableForSymtab(Sec);
  }

  return make_error<StringError>("dynamic string table not found",
                                 inconvertibleErrorCode());
}

template <typename ELFT>
Expected<LibraryDepsInfo> parseELF(const object::ELFFile<ELFT> &Elf) {
  LibraryDepsInfo Deps;
  Expected<StringRef> StrTabOrErr = getDynamicStrTab(Elf);
  if (!StrTabOrErr)
    return StrTabOrErr.takeError();

  const char *Data = StrTabOrErr->data();

  auto DynamicEntriesOrError = Elf.dynamicEntries();
  if (!DynamicEntriesOrError) {
    return DynamicEntriesOrError.takeError();
  }

  for (const typename ELFT::Dyn &Dyn : *DynamicEntriesOrError) {
    switch (Dyn.d_tag) {
    case ELF::DT_NEEDED:
      Deps.addDep(Data + Dyn.d_un.d_val);
      break;
    case ELF::DT_RPATH: {
      SmallVector<StringRef, 4> RawPaths;
      SplitString(Data + Dyn.d_un.d_val, RawPaths,
                  sys::EnvPathSeparator == ':' ? ":" : ";");
      for (const auto &raw : RawPaths)
        Deps.addRPath(raw.str());
      break;
    }
    case ELF::DT_RUNPATH: {
      SmallVector<StringRef, 4> RawPaths;
      SplitString(Data + Dyn.d_un.d_val, RawPaths,
                  sys::EnvPathSeparator == ':' ? ":" : ";");
      for (const auto &raw : RawPaths)
        Deps.addRunPath(raw.str());
      break;
    }
    case ELF::DT_FLAGS_1:
      // Check if this is not a pie executable.
      if (Dyn.d_un.d_val & ELF::DF_1_PIE)
        Deps.isPIE = true;
      break;
      // (Dyn.d_tag == ELF::DT_NULL) continue;
      // (Dyn.d_tag == ELF::DT_AUXILIARY || Dyn.d_tag == ELF::DT_FILTER)
    default:
      break;
    }
  }
  return Deps;
}

Expected<LibraryDepsInfo> parseELFDeps(const object::ELFObjectFileBase &Obj) {
  using namespace object;
  LLVM_DEBUG(dbgs() << "parseELFDeps: Detected ELF object\n";);
  if (const auto *ELF = dyn_cast<ELF32LEObjectFile>(&Obj))
    return parseELF(ELF->getELFFile());
  else if (const auto *ELF = dyn_cast<ELF32BEObjectFile>(&Obj))
    return parseELF(ELF->getELFFile());
  else if (const auto *ELF = dyn_cast<ELF64LEObjectFile>(&Obj))
    return parseELF(ELF->getELFFile());
  else if (const auto *ELF = dyn_cast<ELF64BEObjectFile>(&Obj))
    return parseELF(ELF->getELFFile());

  LLVM_DEBUG(dbgs() << "parseELFDeps: Unknown ELF format\n";);
  return createStringError(std::errc::not_supported, "Unknown ELF format");
}

Expected<LibraryDepsInfo> LibraryScanner::extractDeps(StringRef FilePath) {
  LLVM_DEBUG(dbgs() << "extractDeps: Attempting to open file " << FilePath
                    << "\n";);

  ObjectFileLoader ObjLoader(FilePath);
  auto ObjOrErr = ObjLoader.getObjectFile();
  if (!ObjOrErr) {
    LLVM_DEBUG(dbgs() << "extractDeps: Failed to open " << FilePath << "\n";);
    return ObjOrErr.takeError();
  }

  object::ObjectFile *Obj = &ObjOrErr.get();

  if (auto *elfObj = dyn_cast<object::ELFObjectFileBase>(Obj)) {
    LLVM_DEBUG(dbgs() << "extractDeps: File " << FilePath
                      << " is an ELF object\n";);

    return parseELFDeps(*elfObj);
  }

  if (auto *macho = dyn_cast<object::MachOObjectFile>(Obj)) {
    LLVM_DEBUG(dbgs() << "extractDeps: File " << FilePath
                      << " is a Mach-O object\n";);
    return parseMachODeps(*macho);
  }

  if (Obj->isCOFF()) {
    // TODO: COFF support
    return LibraryDepsInfo();
  }

  LLVM_DEBUG(dbgs() << "extractDeps: Unsupported binary format for file "
                    << FilePath << "\n";);
  return createStringError(inconvertibleErrorCode(),
                           "Unsupported binary format: %s",
                           FilePath.str().c_str());
}

std::optional<std::string> LibraryScanner::shouldScan(StringRef FilePath) {
  std::error_code EC;

  LLVM_DEBUG(dbgs() << "[shouldScan] Checking: " << FilePath << "\n";);

  // [1] Check file existence early
  if (!sys::fs::exists(FilePath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: file does not exist.\n";);

    return std::nullopt;
  }

  // [2] Resolve to canonical path
  auto CanonicalPathOpt = ScanHelper.resolve(FilePath, EC);
  if (EC || !CanonicalPathOpt) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: failed to resolve path (EC="
                      << EC.message() << ").\n";);

    return std::nullopt;
  }

  const std::string &CanonicalPath = *CanonicalPathOpt;
  LLVM_DEBUG(dbgs() << "  -> Canonical path: " << CanonicalPath << "\n");

  // [3] Check if it's a directory — skip directories
  if (sys::fs::is_directory(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: path is a directory.\n";);

    return std::nullopt;
  }

  // [4] Skip if it's not a shared library.
  if (!DylibPathValidator::isSharedLibrary(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: not a shared library.\n";);
    return std::nullopt;
  }

  // [5] Skip if we've already seen this path (via cache)
  if (ScanHelper.hasSeenOrMark(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: already seen.\n";);

    return std::nullopt;
  }

  // [6] Already tracked in LibraryManager?
  if (LibMgr.hasLibrary(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: already tracked by LibraryManager.\n";);

    return std::nullopt;
  }

  // [7] Run user-defined hook (default: always true)
  if (!ShouldScanCall(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: user-defined hook rejected.\n";);

    return std::nullopt;
  }

  LLVM_DEBUG(dbgs() << "  -> Accepted: ready to scan " << CanonicalPath
                    << "\n";);
  return CanonicalPath;
}

void LibraryScanner::handleLibrary(StringRef FilePath, PathType K, int level) {
  LLVM_DEBUG(dbgs() << "LibraryScanner::handleLibrary: Scanning: " << FilePath
                    << ", level=" << level << "\n";);
  auto CanonPathOpt = shouldScan(FilePath);
  if (!CanonPathOpt) {
    LLVM_DEBUG(dbgs() << "  Skipped (shouldScan returned false): " << FilePath
                      << "\n";);

    return;
  }
  const std::string CanonicalPath = *CanonPathOpt;

  auto DepsOrErr = extractDeps(CanonicalPath);
  if (!DepsOrErr) {
    LLVM_DEBUG(dbgs() << "  Failed to extract deps for: " << CanonicalPath
                      << "\n";);
    handleError(DepsOrErr.takeError());
    return;
  }

  LibraryDepsInfo &Deps = *DepsOrErr;

  LLVM_DEBUG({
    dbgs() << "    Found deps : \n";
    for (const auto &dep : Deps.deps)
      dbgs() << "        : " << dep << "\n";
    dbgs() << "    Found @rpath : " << Deps.rpath.size() << "\n";
    for (const auto &r : Deps.rpath)
      dbgs() << "     : " << r << "\n";
    dbgs() << "    Found @runpath : \n";
    for (const auto &r : Deps.runPath)
      dbgs() << "     : " << r << "\n";
  });

  if (Deps.isPIE && level == 0) {
    LLVM_DEBUG(dbgs() << "  Skipped PIE executable at top level: "
                      << CanonicalPath << "\n";);

    return;
  }

  bool Added = LibMgr.addLibrary(CanonicalPath, K);
  if (!Added) {
    LLVM_DEBUG(dbgs() << "  Already added: " << CanonicalPath << "\n";);
    return;
  }

  // Heuristic 1: No RPATH/RUNPATH, skip deps
  if (Deps.rpath.empty() && Deps.runPath.empty()) {
    LLVM_DEBUG(
        dbgs() << "LibraryScanner::handleLibrary: Skipping deps (Heuristic1): "
               << CanonicalPath << "\n";);
    return;
  }

  // Heuristic 2: All RPATH and RUNPATH already tracked
  auto allTracked = [&](const auto &Paths) {
    LLVM_DEBUG(dbgs() << "   Checking : " << Paths.size() << "\n";);
    return std::all_of(Paths.begin(), Paths.end(), [&](StringRef P) {
      LLVM_DEBUG(dbgs() << "      Checking isTrackedBasePath : " << P << "\n";);
      return ScanHelper.isTrackedBasePath(
          DylibResolver::resolvelinkerFlag(P, CanonicalPath));
    });
  };

  if (allTracked(Deps.rpath) && allTracked(Deps.runPath)) {
    LLVM_DEBUG(
        dbgs() << "LibraryScanner::handleLibrary: Skipping deps (Heuristic2): "
               << CanonicalPath << "\n";);
    return;
  }

  DylibPathValidator Validator(ScanHelper.getPathResolver());
  DylibResolver Resolver(Validator);
  Resolver.configure(CanonicalPath,
                     {{Deps.rpath, SearchPathType::RPath},
                      {ScanHelper.getSearchPaths(), SearchPathType::UsrOrSys},
                      {Deps.runPath, SearchPathType::RunPath}});
  for (StringRef Dep : Deps.deps) {
    LLVM_DEBUG(dbgs() << "  Resolving dep: " << Dep << "\n";);
    auto DepFullOpt = Resolver.resolve(Dep);
    if (!DepFullOpt) {
      LLVM_DEBUG(dbgs() << "    Failed to resolve dep: " << Dep << "\n";);

      continue;
    }
    LLVM_DEBUG(dbgs() << "    Resolved dep to: " << *DepFullOpt << "\n";);

    handleLibrary(*DepFullOpt, K, level + 1);
  }
}

void LibraryScanner::scanBaseDir(std::shared_ptr<LibrarySearchPath> SP) {
  if (!sys::fs::is_directory(SP->BasePath) || SP->BasePath.empty()) {
    LLVM_DEBUG(
        dbgs() << "LibraryScanner::scanBaseDir: Invalid or empty basePath: "
               << SP->BasePath << "\n";);
    return;
  }

  LLVM_DEBUG(dbgs() << "LibraryScanner::scanBaseDir: Scanning directory: "
                    << SP->BasePath << "\n";);
  std::error_code EC;

  SP->State.store(ScanState::Scanning);

  for (sys::fs::directory_iterator It(SP->BasePath, EC), end; It != end && !EC;
       It.increment(EC)) {
    auto Entry = *It;
    if (!Entry.status())
      continue;

    auto Status = *Entry.status();
    if (sys::fs::is_regular_file(Status) || sys::fs::is_symlink_file(Status)) {
      LLVM_DEBUG(dbgs() << "  Found file: " << Entry.path() << "\n";);
      // async support ?
      handleLibrary(Entry.path(), SP->Kind);
    }
  }

  SP->State.store(ScanState::Scanned);
}

void LibraryScanner::scanNext(PathType K, size_t BatchSize) {
  LLVM_DEBUG(dbgs() << "LibraryScanner::scanNext: Scanning next batch of size "
                    << BatchSize << " for kind "
                    << (K == PathType::User ? "User" : "System") << "\n";);

  auto SearchPaths = ScanHelper.getNextBatch(K, BatchSize);
  for (auto &SP : SearchPaths) {
    LLVM_DEBUG(dbgs() << "  Scanning unit with basePath: " << SP->BasePath
                      << "\n";);

    scanBaseDir(SP);
  }
}

} // end namespace llvm::orc
