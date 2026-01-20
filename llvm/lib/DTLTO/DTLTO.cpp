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

#include <iostream>
#include <string>

using namespace llvm;

namespace {

// Writes the content of a memory buffer into a file.
llvm::Error saveBuffer(StringRef FileBuffer, StringRef FilePath) {
  std::error_code EC;
  raw_fd_ostream OS(FilePath.str(), EC, sys::fs::OpenFlags::OF_None);
  if (EC) {
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create file %s: %s", FilePath.data(),
                             EC.message().c_str());
  }
  OS.write(FileBuffer.data(), FileBuffer.size());
  if (OS.has_error()) {
    return createStringError(inconvertibleErrorCode(),
                             "Failed writing to file %s", FilePath.data());
  }
  return Error::success();
}

// Compute the file path for a thin archive member.
//
// For thin archives, an archive member name is typically a file path relative
// to the archive file's directory. This function resolves that path.
SmallString<64> computeThinArchiveMemberPath(const StringRef ArchivePath,
                                             const StringRef MemberName) {
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
// This function uses a cache to avoid repeatedly reading the same file.
// It reads only the header portion (magic bytes) of the file to identify
// the archive type.
Expected<bool> lto::DTLTO::isThinArchive(const StringRef ArchivePath) {
  // Return cached result if available.
  auto Cached = ArchiveFiles.find(ArchivePath);
  if (Cached != ArchiveFiles.end())
    return Cached->second;

  uint64_t FileSize = -1;
  bool IsThin = false;
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
  ErrorOr<std::unique_ptr<MemoryBuffer>> MemBufferOrError =
      MemoryBuffer::getFileSlice(ArchivePath, sizeof(object::ThinArchiveMagic),
                                 0);

  if ((EC = MemBufferOrError.getError()))
    return createStringError(inconvertibleErrorCode(),
                             "Failed to read from archive %s: %s",
                             ArchivePath.data(), EC.message().c_str());

  StringRef MemBuf = (*MemBufferOrError.get()).getBuffer();
  if (file_magic::archive != identify_magic(MemBuf))
    return createStringError(inconvertibleErrorCode(),
                             "Unknown format for archive %s",
                             ArchivePath.data());

  IsThin = MemBuf.starts_with(object::ThinArchiveMagic);

  // Cache the result
  ArchiveFiles[ArchivePath] = IsThin;
  return IsThin;
}

// This function performs the following tasks:
// 1. Adds the input file to the LTO object's list of input files.
// 2. For thin archive members, generates a new module ID which is a path to a
// thin archive member file.
// 3. For regular archive members, generates a new unique module ID.
// 4. Updates the bitcode module's identifier.
Expected<std::shared_ptr<lto::InputFile>>
lto::DTLTO::addInput(std::unique_ptr<lto::InputFile> InputPtr) {
  TimeTraceScope TimeScope("Add input for DTLTO");

  // Add the input file to the LTO object.
  InputFiles.emplace_back(InputPtr.release());
  std::shared_ptr<lto::InputFile> &Input = InputFiles.back();

  StringRef ModuleId = Input->getName();
  StringRef ArchivePath = Input->getArchivePath();

  // Only process archive members.
  if (ArchivePath.empty())
    return Input;

  SmallString<64> NewModuleId;
  BitcodeModule &BM = Input->getPrimaryBitcodeModule();

  // Check if the archive is a thin archive.
  Expected<bool> IsThin = isThinArchive(ArchivePath);
  if (!IsThin)
    return IsThin.takeError();

  if (*IsThin) {
    // For thin archives, use the path to the actual file.
    NewModuleId =
        computeThinArchiveMemberPath(ArchivePath, Input->getMemberName());
  } else {
    // For regular archives, generate a unique name.
    Input->memberOfArchive(true);

    // Create unique identifier using process ID and sequence number.
    std::string PID = utohexstr(sys::Process::getProcessId());
    std::string Seq = std::to_string(InputFiles.size());

    NewModuleId = {sys::path::filename(ModuleId), ".", Seq, ".", PID, ".o"};
  }

  // Update the module identifier and save it.
  BM.setModuleIdentifier(Saver.save(NewModuleId.str()));

  return Input;
}

// Write the archive member content to a file named after the module ID.
// If a file with that name already exists, it's likely a leftover from a
// previously terminated linker process and can be safely overwritten.
Error lto::DTLTO::saveInputArchiveMember(lto::InputFile *Input) {
  StringRef ModuleId = Input->getName();
  if (Input->isMemberOfArchive()) {
    TimeTraceScope TimeScope("Save input archive member for DTLTO", ModuleId);
    MemoryBufferRef MemoryBufferRef = Input->getFileBuffer();
    if (Error EC = saveBuffer(MemoryBufferRef.getBuffer(), ModuleId))
      return EC;
  }
  return Error::success();
}

// Iterates through all ThinLTO-enabled input files and saves their content
// to separate files if they are regular archive members.
Error lto::DTLTO::saveInputArchiveMembers() {
  for (auto &Input : InputFiles) {
    if (!Input->isThinLTO())
      continue;
    if (Error EC = saveInputArchiveMember(Input.get()))
      return EC;
  }
  return Error::success();
}

// Entry point for DTLTO archives support.
//
// Sets up the temporary file remover and processes archive members.
// Must be called after all inputs are added but before optimization begins.
llvm::Error lto::DTLTO::handleArchiveInputs() {

  // Process and save archive members to separate files if needed.
  if (Error EC = saveInputArchiveMembers())
    return EC;
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
