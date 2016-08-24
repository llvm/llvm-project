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

#ifdef HAVE_LLVM_REVISION
#include "LLVMLTORevision.h"
#endif

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::lto;

static void commitEntry(StringRef TempFilename, StringRef EntryPath) {
  // Rename to final destination (hopefully race condition won't matter here)
  auto EC = sys::fs::rename(TempFilename, EntryPath);
  if (EC) {
    // Renaming failed, probably not the same filesystem, copy and delete.
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

CacheObjectOutput::~CacheObjectOutput() {
  if (EntryPath.empty())
    // The entry was never used by the client (tryLoadFromCache() wasn't called)
    return;
  // TempFilename is only set if getStream() was called, i.e. on cache miss when
  // tryLoadFromCache() returned false. And EntryPath is valid if a Key was
  // submitted, otherwise it has been set to CacheDirectoryPath in
  // tryLoadFromCache.
  if (!TempFilename.empty()) {
    if (EntryPath == CacheDirectoryPath)
      // The Key supplied to tryLoadFromCache was empty, do not commit the temp.
      EntryPath = TempFilename;
    else
      // We commit the tempfile into the cache now, by moving it to EntryPath.
      commitEntry(TempFilename, EntryPath);
  }
  // Load the entry from the cache now.
  auto ReloadedBufferOrErr = MemoryBuffer::getFile(EntryPath);
  if (auto EC = ReloadedBufferOrErr.getError())
    report_fatal_error(Twine("Can't reload cached file '") + EntryPath + "': " +
                       EC.message() + "\n");

  // Supply the resulting buffer to the user.
  AddBuffer(std::move(*ReloadedBufferOrErr));
}

// Return an allocated stream for the output, or null in case of failure.
std::unique_ptr<raw_pwrite_stream> CacheObjectOutput::getStream() {
  assert(!EntryPath.empty() && "API Violation: client didn't call "
                               "tryLoadFromCache() before getStream()");
  // Write to a temporary to avoid race condition
  int TempFD;
  std::error_code EC =
      sys::fs::createTemporaryFile("Thin", "tmp.o", TempFD, TempFilename);
  if (EC) {
    errs() << "Error: " << EC.message() << "\n";
    report_fatal_error("ThinLTO: Can't get a temporary file");
  }
  return llvm::make_unique<raw_fd_ostream>(TempFD, /* ShouldClose */ true);
}

// Try loading from a possible cache first, return true on cache hit.
bool CacheObjectOutput::tryLoadFromCache(StringRef Key) {
  assert(!CacheDirectoryPath.empty() &&
         "CacheObjectOutput was initialized without a cache path");
  if (Key.empty()) {
    // Client didn't compute a valid key. EntryPath has been set to
    // CacheDirectoryPath.
    EntryPath = CacheDirectoryPath;
    return false;
  }
  sys::path::append(EntryPath, CacheDirectoryPath, Key);
  return sys::fs::exists(EntryPath);
}
