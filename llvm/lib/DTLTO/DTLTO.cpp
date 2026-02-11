//===- Dtlto.cpp - Distributed ThinLTO implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements support functions for Distributed ThinLTO, focusing on
// preparing input files for distribution.
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
#include "llvm/Support/Signals.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

namespace {

// Saves the content of Buffer to Path overwriting any existing file.
Error save(StringRef Buffer, StringRef Path) {
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

// Saves the content of Input to Path overwriting any existing file.
Error save(lto::InputFile *Input, StringRef Path) {
  MemoryBufferRef MB = Input->getFileBuffer();
  return save(MB.getBuffer(), Path);
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
  } else
    MemberPath = MemberName;
  sys::path::remove_dots(MemberPath, /*remove_dot_dot=*/true);
  return MemberPath;
}

} // namespace

// Determines if a file at the given path is a thin archive file.
//
// This function uses a cache to avoid repeatedly reading the same file.
// It reads only the header portion (magic bytes) of the file to identify
// the archive type.
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
//
// This function performs the following tasks:
// 1. Add the input file to the LTO object's list of input files.
// 2. For thin archive members, overwrite the module ID with the path to the
//    member file on disk.
// 3. For archive members and FatLTO objects, overwrite the module ID with a
//    unique path naming a file that will contain the member content. The file
//    is created and populated later (see serializeInputs()).
Expected<std::shared_ptr<lto::InputFile>>
lto::DTLTO::addInput(std::unique_ptr<InputFile> InputPtr) {
  TimeTraceScope TimeScope("Add input for DTLTO");

  // Add the input file to the LTO object.
  InputFiles.emplace_back(InputPtr.release());
  auto &Input = InputFiles.back();
  BitcodeModule &BM = Input->getPrimaryBitcodeModule();

  StringRef ArchivePath = Input->getArchivePath();

  // In most cases, the module ID already points to an individual bitcode file
  // on disk, so no further preparation for distribution is required.
  if (ArchivePath.empty() && !Input->isFatLTOObject())
    return Input;

  // For a member of a thin archive that is not a FatLTO object, there is an
  // existing file on disk that can be used, so we can avoid having to
  // serialize.
  Expected<bool> UseThinMember =
      Input->isFatLTOObject() ? false : isThinArchive(ArchivePath);
  if (!UseThinMember)
    return UseThinMember.takeError();
  if (*UseThinMember) {
    // For thin archives, use the path to the actual member file on disk.
    auto MemberPath =
        computeThinArchiveMemberPath(ArchivePath, Input->getMemberName());
    BM.setModuleIdentifier(Saver.save(MemberPath.str()));
    return Input;
  }

  // A new file on disk will be needed for archive members and FatLTO objects.
  Input->setSerializeForDistribution(true);

  // Create a unique path by including the process ID and sequence number in the
  // filename.
  SmallString<256> Id(sys::path::parent_path(LinkerOutputFile));
  sys::path::append(Id,
                    Twine(sys::path::filename(Input->getName())) + "." +
                        std::to_string(InputFiles.size()) /*Sequence number*/ +
                        "." + utohexstr(sys::Process::getProcessId()) + ".o");
  BM.setModuleIdentifier(Saver.save(Id.str()));
  return Input;
}

// Save the contents of ThinLTO-enabled input files that must be serialized for
// distribution, such as archive members and FatLTO objects, to individual
// bitcode files named after the module ID.
//
// Must be called after all input files are added but before optimization
// begins. If a file with that name already exists, it is likely a leftover from
// a previously terminated linker process and can be safely overwritten.
llvm::Error lto::DTLTO::serializeInputsForDistribution() {
  for (auto &Input : InputFiles) {
    if (!Input->isThinLTO() || !Input->getSerializeForDistribution())
      continue;
    // Save the content of the input file to a file named after the module ID.
    StringRef ModuleId = Input->getName();
    TimeTraceScope TimeScope("Serialize bitcode input for DTLTO", ModuleId);
    // Cleanup this file on abnormal process exit.
    if (!SaveTemps)
      llvm::sys::RemoveFileOnSignal(ModuleId);
    if (Error EC = save(Input.get(), ModuleId))
      return EC;
  }

  return Error::success();
}

// Remove serialized inputs created to enable distribution.
void lto::DTLTO::cleanup() {
  if (!SaveTemps) {
    TimeTraceScope TimeScope("Remove temporary inputs for DTLTO");
    for (auto &Input : InputFiles) {
      if (!Input->getSerializeForDistribution())
        continue;
      std::error_code EC =
          sys::fs::remove(Input->getName(), /*IgnoreNonExisting=*/true);
      if (EC &&
          EC != std::make_error_code(std::errc::no_such_file_or_directory))
        errs() << "warning: could not remove temporary DTLTO input file '"
               << Input->getName() << "': " << EC.message() << "\n";
    }
  }
  Base::cleanup();
}
