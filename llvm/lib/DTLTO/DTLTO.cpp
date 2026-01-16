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
// archive file handling.
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
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
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

// Compute the path for a thin archive member.
//
// For thin archives, the member name is typically a path relative to the thin
// archive's directory. This function resolves that path.
SmallString<64> computeThinArchiveMemberPath(StringRef ArchivePath,
                                             StringRef MemberName) {
  assert(!ArchivePath.empty() && "An archive file path must be non empty.");
  SmallString<64> MemberPath;
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
// This function uses a cache to avoid repeatedly reading the same file. It
// reads only the header portion (magic bytes) to identify the archive type.
Expected<bool> lto::DTLTO::isThinArchive(StringRef ArchivePath) {
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
// 3. For archive members, overwrite the module ID with a unique path naming a
//    file that will contain the member content. The file is created and
//    populated later (see handleArchiveInputs()).
Expected<std::shared_ptr<lto::InputFile>>
lto::DTLTO::addInput(std::unique_ptr<InputFile> InputPtr) {
  TimeTraceScope TimeScope("Add input for DTLTO");

  // Add the input file to the LTO object.
  InputFiles.emplace_back(InputPtr.release());
  std::shared_ptr<InputFile> &Input = InputFiles.back();

  StringRef ModuleId = Input->getName();
  StringRef ArchivePath = Input->getArchivePath();

  // Only process archive members and thin archive members.
  if (ArchivePath.empty())
    return Input;

  SmallString<64> NewModuleId;
  BitcodeModule &BM = Input->getPrimaryBitcodeModule();

  // Check if this is a member of a thin archive.
  Expected<bool> IsThin = isThinArchive(ArchivePath);
  if (!IsThin)
    return IsThin.takeError();

  if (*IsThin) {
    // Use the path to the actual file.
    NewModuleId =
        computeThinArchiveMemberPath(ArchivePath, Input->getMemberName());
  } else {
    // Generate a unique name.
    Input->memberOfArchive(true);

    // Create unique identifier using process ID and sequence number.
    std::string PID = utohexstr(sys::Process::getProcessId());
    std::string Seq = std::to_string(InputFiles.size());

    NewModuleId = {sys::path::filename(ModuleId), ".", Seq, ".", PID, ".o"};
  }

  // Update the module ID.
  BM.setModuleIdentifier(Saver.save(NewModuleId.str()));

  return Input;
}

// Save the content of ThinLTO-enabled archive members to individual bitcode
// files named after the module ID.
//
// Must be called after all input files are added but before optimization
// begins. If a file with that name already exists, it's likely a leftover from
// a previously terminated linker process and can be safely overwritten.
llvm::Error lto::DTLTO::handleArchiveInputs() {
  for (auto &Input : InputFiles) {
    if (!Input->isThinLTO() || !Input->isMemberOfArchive())
      continue;
    // Save the content of the input file to a file named after the module ID.
    StringRef ModuleId = Input->getName();
    TimeTraceScope TimeScope("Save input archive member for DTLTO", ModuleId);
    if (Error EC = save(Input.get(), ModuleId))
      return EC;
  }

  return Error::success();
}

// Remove temporary archive member files created to enable distribution.
void lto::DTLTO::cleanup() {
  {
    TimeTraceScope TimeScope("Remove temporary inputs for DTLTO");
    for (auto &Input : InputFiles)
      if (Input->isMemberOfArchive())
        sys::fs::remove(Input->getName(), /*IgnoreNonExisting=*/true);
  }
  Base::cleanup();
}
