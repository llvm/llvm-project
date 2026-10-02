//===- DTLTOInputFiles.cpp - Integrated Distributed ThinLTO implementation ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements support functions for Integrated Distributed ThinLTO,
// focusing on preparing input files for distribution.
//
//===----------------------------------------------------------------------===//

#include "llvm/DTLTO/DTLTO.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#ifdef _WIN32
#include "llvm/Support/Windows/WindowsSupport.h"
#endif

#include <string>

using namespace llvm;

// Saves the content of Buffer to Path overwriting any existing file.
Error lto::DTLTO::save(StringRef Buffer, StringRef Path) {
  std::error_code EC;
  raw_fd_ostream OS(Path.str(), EC, sys::fs::OpenFlags::OF_None);
  if (EC)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create file %s: %s", Path.data(),
                             EC.message().c_str());
  OS.write(Buffer.data(), Buffer.size());
  if (OS.has_error())
    return createStringError(inconvertibleErrorCode(),
                             "Failed writing to file %s", Path.data());
  return Error::success();
}

namespace {
// Normalize and save a path. Aside from expanding Windows 8.3 short paths,
// no other normalization is currently required here. These paths are
// machine-local and break distribution systems; other normalization is
// handled by the DTLTO distributors.
Expected<StringRef> normalizePath(StringRef Path, StringSaver &Saver) {
#if defined(_WIN32)
  if (Path.empty())
    return Path;
  SmallString<256> Expanded;
  if (std::error_code EC = sys::windows::makeLongFormPath(Path, Expanded))
    return createStringError(inconvertibleErrorCode(),
                             "Normalization failed for path %s: %s",
                             Path.str().c_str(), EC.message().c_str());
  return Saver.save(Expanded.str());
#else
  return Saver.save(Path);
#endif
}

// Compute the file path for a thin archive member.
//
// For thin archives, an archive member name is typically a file path relative
// to the archive file's directory. This function resolves that path.
SmallString<256> computeThinArchiveMemberPath(StringRef ArchivePath,
                                              StringRef MemberName) {
  assert(!ArchivePath.empty() && "An archive file path must be non empty.");
  SmallString<256> MemberPath;
  if (sys::path::is_relative(MemberName)) {
    MemberPath = sys::path::parent_path(ArchivePath);
    sys::path::append(MemberPath, MemberName);
  } else {
    MemberPath = MemberName;
  }
  sys::path::remove_dots(MemberPath, /*remove_dot_dot=*/true);
  return MemberPath;
}

} // namespace

// Determines if a file at the given path is a thin archive file.
Expected<bool> lto::DTLTO::isThinArchive(const StringRef ArchivePath) {
  // Return cached result if available.
  auto Cached = ArchiveIsThinCache.find(ArchivePath);
  if (Cached != ArchiveIsThinCache.end())
    return Cached->second;

  uint64_t FileSize = -1;
  std::error_code EC = sys::fs::file_size(ArchivePath, FileSize);
  if (EC)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to get file size from archive %s: %s",
                             ArchivePath.data(), EC.message().c_str());
  if (FileSize < sizeof(object::ThinArchiveMagic))
    return createStringError(inconvertibleErrorCode(),
                             "Archive file size is too small %s",
                             ArchivePath.data());

  // Read only the first few bytes containing the magic signature.
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr = MemoryBuffer::getFileSlice(
      ArchivePath, sizeof(object::ThinArchiveMagic), 0);
  if ((EC = MBOrErr.getError()))
    return createStringError(inconvertibleErrorCode(),
                             "Failed to read from archive %s: %s",
                             ArchivePath.data(), EC.message().c_str());

  StringRef Buf = (*MBOrErr)->getBuffer();
  if (file_magic::archive != identify_magic(Buf))
    return createStringError(inconvertibleErrorCode(),
                             "Unknown format for archive %s",
                             ArchivePath.data());

  bool IsThin = Buf.starts_with(object::ThinArchiveMagic);

  // Cache the result.
  ArchiveIsThinCache[ArchivePath] = IsThin;

  return IsThin;
}

// Add an input file and prepare it for distribution.
Expected<std::shared_ptr<lto::InputFile>>
lto::DTLTO::addInput(std::unique_ptr<InputFile> InputPtr) {
  TimeTraceScope TimeScope("Add input for DTLTO");

  // Add the input file to the LTO object.
  InputFiles.emplace_back(InputPtr.release());
  auto &Input = InputFiles.back();
  BitcodeModule &BM = Input->getPrimaryBitcodeModule();

  auto setIdFromPath = [&](StringRef Path) -> Error {
    auto N = normalizePath(Path, Saver);
    if (!N)
      return N.takeError();
    BM.setModuleIdentifier(*N);
    return Error::success();
  };

  StringRef ArchivePath = Input->getArchivePath();

  // In most cases, the module ID already points to an individual bitcode file
  // on disk, so no further preparation for distribution is required. However,
  // on Windows we overwite the module ID to expand Windows 8.3 short form
  // paths. These paths are machine-local and break distribution systems; other
  // normalization is handled by the DTLTO distributors.
  if (ArchivePath.empty() && !Input->isFatLTOObject()) {
#if defined(_WIN32)
    if (Error E = setIdFromPath(Input->getName()))
      return std::move(E);
#endif
    return Input;
  }

  // For a member of a thin archive that is not a FatLTO object, there is an
  // existing file on disk that can be used, so we can avoid having to extract.
  Expected<bool> UseThinMember =
      Input->isFatLTOObject() ? false : isThinArchive(ArchivePath);
  if (!UseThinMember)
    return UseThinMember.takeError();
  if (*UseThinMember) {
    // For thin archives, use the path to the actual member file on disk.
    auto MemberPath =
        computeThinArchiveMemberPath(ArchivePath, Input->getMemberName());
    if (Error E = setIdFromPath(MemberPath))
      return std::move(E);
    return Input;
  }

  // A new file on disk will be needed for archive members and FatLTO objects.
  Input->setExtractForDistribution(true);

  // Get the normalized output directory, if we haven't already.
  if (LinkerOutputDir.empty()) {
    auto N = normalizePath(
        sys::path::parent_path(DistributorParams.LinkerOutputFile), Saver);
    if (!N)
      return N.takeError();
    LinkerOutputDir = *N;
  }

  // Create a unique path by including the process ID and sequence number in the
  // filename.
  SmallString<256> Id(LinkerOutputDir);
  sys::path::append(Id,
                    Twine(sys::path::filename(Input->getName())) + "." +
                        std::to_string(InputFiles.size()) /*Sequence number*/ +
                        "." + utohexstr(sys::Process::getProcessId()) + ".o");
  BM.setModuleIdentifier(Saver.save(Id.str()));
  return Input;
}

// Save the contents of ThinLTO-enabled input files that must be extracted for
// distribution.
Error lto::DTLTO::extractLTOInputs() {
  for (auto &Input : InputFiles) {
    if (!Input->isThinLTO() || !Input->getExtractForDistribution())
      continue;

    // Save the content of the input file to a file named after the module ID.
    StringRef ModuleID = Input->getName();
    if (!InputModuleIDsToExtract.contains(ModuleID))
      continue;
    TimeTraceScope TimeScope("Extract bitcode input for DTLTO", ModuleID);
    MemoryBufferRef Buf = Input->getFileBuffer();
    if (Error Err = save(Buf.getBuffer(), ModuleID))
      return Err;
    // Cleanup this file on abnormal process exit.
    if (!SaveTemps)
      addToCleanup(ModuleID);
  }
  return Error::success();
}
