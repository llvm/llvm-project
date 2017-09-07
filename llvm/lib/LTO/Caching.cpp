//===-Caching.cpp - LLVM Link Time Optimizer Cache Handling ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Caching for ThinLTO.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/Caching.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::lto;

Expected<NativeObjectCache> lto::localCache(StringRef CacheDirectoryPath,
                                            AddBufferFn AddBuffer) {
  if (std::error_code EC = sys::fs::create_directories(CacheDirectoryPath))
    return errorCodeToError(EC);

  return [=](unsigned Task, StringRef Key) -> AddStreamFn {
    // This choice of file name allows the cache to be pruned (see pruneCache()
    // in include/llvm/Support/CachePruning.h).
    SmallString<64> EntryPath;
    sys::path::append(EntryPath, CacheDirectoryPath, "llvmcache-" + Key);
    // First, see if we have a cache hit.
    ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
        MemoryBuffer::getFile(EntryPath);
    if (MBOrErr) {
      AddBuffer(Task, std::move(*MBOrErr), EntryPath);
      return AddStreamFn();
    }

    if (MBOrErr.getError() != errc::no_such_file_or_directory)
      report_fatal_error(Twine("Failed to open cache file ") + EntryPath +
                         ": " + MBOrErr.getError().message() + "\n");

    // This native object stream is responsible for commiting the resulting
    // file to the cache and calling AddBuffer to add it to the link.
    struct CacheStream : NativeObjectStream {
      AddBufferFn AddBuffer;
      std::string TempFilename;
      std::string EntryPath;
      unsigned Task;

      CacheStream(std::unique_ptr<raw_pwrite_stream> OS, AddBufferFn AddBuffer,
                  std::string TempFilename, std::string EntryPath,
                  unsigned Task)
          : NativeObjectStream(std::move(OS)), AddBuffer(std::move(AddBuffer)),
            TempFilename(std::move(TempFilename)),
            EntryPath(std::move(EntryPath)), Task(Task) {}

      ~CacheStream() {
        // Make sure the file is closed before committing it.
        OS.reset();

        // Open the file first to avoid racing with a cache pruner.
        ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
            MemoryBuffer::getFile(TempFilename);

        // This is atomic on POSIX systems.
        if (auto EC = sys::fs::rename(TempFilename, EntryPath))
          report_fatal_error(Twine("Failed to rename temporary file ") +
                             TempFilename + ": " + EC.message() + "\n");

        if (!MBOrErr)
          report_fatal_error(Twine("Failed to open cache file ") + EntryPath +
                             ": " + MBOrErr.getError().message() + "\n");
        AddBuffer(Task, std::move(*MBOrErr), EntryPath);
      }
    };

    return [=](size_t Task) -> std::unique_ptr<NativeObjectStream> {
      // Write to a temporary to avoid race condition
      int TempFD;
      SmallString<64> TempFilenameModel, TempFilename;
      sys::path::append(TempFilenameModel, CacheDirectoryPath, "Thin-%%%%%%.tmp.o");
      std::error_code EC =
          sys::fs::createUniqueFile(TempFilenameModel, TempFD, TempFilename,
                                    sys::fs::owner_read | sys::fs::owner_write);
      if (EC) {
        errs() << "Error: " << EC.message() << "\n";
        report_fatal_error("ThinLTO: Can't get a temporary file");
      }

      // This CacheStream will move the temporary file into the cache when done.
      return llvm::make_unique<CacheStream>(
          llvm::make_unique<raw_fd_ostream>(TempFD, /* ShouldClose */ true),
          AddBuffer, TempFilename.str(), EntryPath.str(), Task);
    };
  };
}
