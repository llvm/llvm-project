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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::lto;

static void commitEntry(StringRef TempFilename, StringRef EntryPath) {
  // Rename to final destination (hopefully race condition won't matter here)
  auto EC = sys::fs::rename(TempFilename, EntryPath);
  if (EC) {
    // Renaming failed, probably not the same filesystem, copy and delete.
    // FIXME: Avoid needing to do this by creating the temporary file in the
    // cache directory.
    {
      auto ReloadedBufferOrErr = MemoryBuffer::getFile(TempFilename);
      if (auto EC = ReloadedBufferOrErr.getError())
        report_fatal_error(Twine("Failed to open temp file '") + TempFilename +
                           "': " + EC.message() + "\n");

      raw_fd_ostream OS(EntryPath, EC, sys::fs::F_None);
      if (EC)
        report_fatal_error(Twine("Failed to open ") + EntryPath +
                           " to save cached entry\n");
      // I'm not sure what are the guarantee if two processes are doing this
      // at the same time.
      OS << (*ReloadedBufferOrErr)->getBuffer();
    }
    sys::fs::remove(TempFilename);
  }
}

NativeObjectCache lto::localCache(std::string CacheDirectoryPath,
                                  AddFileFn AddFile) {
  return [=](unsigned Task, StringRef Key) -> AddStreamFn {
    // First, see if we have a cache hit.
    SmallString<64> EntryPath;
    sys::path::append(EntryPath, CacheDirectoryPath, Key);
    if (sys::fs::exists(EntryPath)) {
      AddFile(Task, EntryPath);
      return AddStreamFn();
    }

    // This native object stream is responsible for commiting the resulting
    // file to the cache and calling AddFile to add it to the link.
    struct CacheStream : NativeObjectStream {
      AddFileFn AddFile;
      std::string TempFilename;
      std::string EntryPath;
      unsigned Task;

      CacheStream(std::unique_ptr<raw_pwrite_stream> OS, AddFileFn AddFile,
                  std::string TempFilename, std::string EntryPath,
                  unsigned Task)
          : NativeObjectStream(std::move(OS)), AddFile(AddFile),
            TempFilename(TempFilename), EntryPath(EntryPath), Task(Task) {}

      ~CacheStream() {
        // Make sure the file is closed before committing it.
        OS.reset();
        commitEntry(TempFilename, EntryPath);
        AddFile(Task, EntryPath);
      }
    };

    return [=](size_t Task) -> std::unique_ptr<NativeObjectStream> {
      // Write to a temporary to avoid race condition
      int TempFD;
      SmallString<64> TempFilename;
      std::error_code EC =
          sys::fs::createTemporaryFile("Thin", "tmp.o", TempFD, TempFilename);
      if (EC) {
        errs() << "Error: " << EC.message() << "\n";
        report_fatal_error("ThinLTO: Can't get a temporary file");
      }

      // This CacheStream will move the temporary file into the cache when done.
      return llvm::make_unique<CacheStream>(
          llvm::make_unique<raw_fd_ostream>(TempFD, /* ShouldClose */ true),
          AddFile, TempFilename.str(), EntryPath.str(), Task);
    };
  };
}
