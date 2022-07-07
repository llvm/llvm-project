//===-- llvm/Debuginfod/Debuginfod.cpp - Debuginfod client library --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// This file contains several definitions for the debuginfod client and server.
/// For the client, this file defines the fetchInfo function. For the server,
/// this file defines the DebuginfodLogEntry and DebuginfodServer structs, as
/// well as the DebuginfodLog, DebuginfodCollection classes. The fetchInfo
/// function retrieves any of the three supported artifact types: (executable,
/// debuginfo, source file) associated with a build-id from debuginfod servers.
/// If a source file is to be fetched, its absolute path must be specified in
/// the Description argument to fetchInfo. The DebuginfodLogEntry,
/// DebuginfodLog, and DebuginfodCollection are used by the DebuginfodServer to
/// scan the local filesystem for binaries and serve the debuginfod protocol.
///
//===----------------------------------------------------------------------===//

#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CachePruning.h"
#include "llvm/Support/Caching.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/xxhash.h"

#include <atomic>

namespace llvm {
static std::string uniqueKey(llvm::StringRef S) { return utostr(xxHash64(S)); }

// Returns a binary BuildID as a normalized hex string.
// Uses lowercase for compatibility with common debuginfod servers.
static std::string buildIDToString(BuildIDRef ID) {
  return llvm::toHex(ID, /*LowerCase=*/true);
}

Expected<SmallVector<StringRef>> getDefaultDebuginfodUrls() {
  const char *DebuginfodUrlsEnv = std::getenv("DEBUGINFOD_URLS");
  if (DebuginfodUrlsEnv == nullptr)
    return SmallVector<StringRef>();

  SmallVector<StringRef> DebuginfodUrls;
  StringRef(DebuginfodUrlsEnv).split(DebuginfodUrls, " ");
  return DebuginfodUrls;
}

/// Finds a default local file caching directory for the debuginfod client,
/// first checking DEBUGINFOD_CACHE_PATH.
Expected<std::string> getDefaultDebuginfodCacheDirectory() {
  if (const char *CacheDirectoryEnv = std::getenv("DEBUGINFOD_CACHE_PATH"))
    return CacheDirectoryEnv;

  SmallString<64> CacheDirectory;
  if (!sys::path::cache_directory(CacheDirectory))
    return createStringError(
        errc::io_error, "Unable to determine appropriate cache directory.");
  sys::path::append(CacheDirectory, "llvm-debuginfod", "client");
  return std::string(CacheDirectory);
}

std::chrono::milliseconds getDefaultDebuginfodTimeout() {
  long Timeout;
  const char *DebuginfodTimeoutEnv = std::getenv("DEBUGINFOD_TIMEOUT");
  if (DebuginfodTimeoutEnv &&
      to_integer(StringRef(DebuginfodTimeoutEnv).trim(), Timeout, 10))
    return std::chrono::milliseconds(Timeout * 1000);

  return std::chrono::milliseconds(90 * 1000);
}

/// The following functions fetch a debuginfod artifact to a file in a local
/// cache and return the cached file path. They first search the local cache,
/// followed by the debuginfod servers.

Expected<std::string> getCachedOrDownloadSource(BuildIDRef ID,
                                                StringRef SourceFilePath) {
  SmallString<64> UrlPath;
  sys::path::append(UrlPath, sys::path::Style::posix, "buildid",
                    buildIDToString(ID), "source",
                    sys::path::convert_to_slash(SourceFilePath));
  return getCachedOrDownloadArtifact(uniqueKey(UrlPath), UrlPath);
}

Expected<std::string> getCachedOrDownloadExecutable(BuildIDRef ID) {
  SmallString<64> UrlPath;
  sys::path::append(UrlPath, sys::path::Style::posix, "buildid",
                    buildIDToString(ID), "executable");
  return getCachedOrDownloadArtifact(uniqueKey(UrlPath), UrlPath);
}

Expected<std::string> getCachedOrDownloadDebuginfo(BuildIDRef ID) {
  SmallString<64> UrlPath;
  sys::path::append(UrlPath, sys::path::Style::posix, "buildid",
                    buildIDToString(ID), "debuginfo");
  return getCachedOrDownloadArtifact(uniqueKey(UrlPath), UrlPath);
}

// General fetching function.
Expected<std::string> getCachedOrDownloadArtifact(StringRef UniqueKey,
                                                  StringRef UrlPath) {
  SmallString<10> CacheDir;

  Expected<std::string> CacheDirOrErr = getDefaultDebuginfodCacheDirectory();
  if (!CacheDirOrErr)
    return CacheDirOrErr.takeError();
  CacheDir = *CacheDirOrErr;

  Expected<SmallVector<StringRef>> DebuginfodUrlsOrErr =
      getDefaultDebuginfodUrls();
  if (!DebuginfodUrlsOrErr)
    return DebuginfodUrlsOrErr.takeError();
  SmallVector<StringRef> &DebuginfodUrls = *DebuginfodUrlsOrErr;
  return getCachedOrDownloadArtifact(UniqueKey, UrlPath, CacheDir,
                                     DebuginfodUrls,
                                     getDefaultDebuginfodTimeout());
}

namespace {

/// A simple handler which streams the returned data to a cache file. The cache
/// file is only created if a 200 OK status is observed.
class StreamedHTTPResponseHandler : public HTTPResponseHandler {
  using CreateStreamFn =
      std::function<Expected<std::unique_ptr<CachedFileStream>>()>;
  CreateStreamFn CreateStream;
  HTTPClient &Client;
  std::unique_ptr<CachedFileStream> FileStream;

public:
  StreamedHTTPResponseHandler(CreateStreamFn CreateStream, HTTPClient &Client)
      : CreateStream(CreateStream), Client(Client) {}
  virtual ~StreamedHTTPResponseHandler() = default;

  Error handleBodyChunk(StringRef BodyChunk) override;
};

} // namespace

Error StreamedHTTPResponseHandler::handleBodyChunk(StringRef BodyChunk) {
  if (!FileStream) {
    if (Client.responseCode() != 200)
      return Error::success();
    Expected<std::unique_ptr<CachedFileStream>> FileStreamOrError =
        CreateStream();
    if (!FileStreamOrError)
      return FileStreamOrError.takeError();
    FileStream = std::move(*FileStreamOrError);
  }
  *FileStream->OS << BodyChunk;
  return Error::success();
}

Expected<std::string> getCachedOrDownloadArtifact(
    StringRef UniqueKey, StringRef UrlPath, StringRef CacheDirectoryPath,
    ArrayRef<StringRef> DebuginfodUrls, std::chrono::milliseconds Timeout) {
  SmallString<64> AbsCachedArtifactPath;
  sys::path::append(AbsCachedArtifactPath, CacheDirectoryPath,
                    "llvmcache-" + UniqueKey);

  Expected<FileCache> CacheOrErr =
      localCache("Debuginfod-client", ".debuginfod-client", CacheDirectoryPath);
  if (!CacheOrErr)
    return CacheOrErr.takeError();

  FileCache Cache = *CacheOrErr;
  // We choose an arbitrary Task parameter as we do not make use of it.
  unsigned Task = 0;
  Expected<AddStreamFn> CacheAddStreamOrErr = Cache(Task, UniqueKey);
  if (!CacheAddStreamOrErr)
    return CacheAddStreamOrErr.takeError();
  AddStreamFn &CacheAddStream = *CacheAddStreamOrErr;
  if (!CacheAddStream)
    return std::string(AbsCachedArtifactPath);
  // The artifact was not found in the local cache, query the debuginfod
  // servers.
  if (!HTTPClient::isAvailable())
    return createStringError(errc::io_error,
                             "No working HTTP client is available.");

  if (!HTTPClient::IsInitialized)
    return createStringError(
        errc::io_error,
        "A working HTTP client is available, but it is not initialized. To "
        "allow Debuginfod to make HTTP requests, call HTTPClient::initialize() "
        "at the beginning of main.");

  HTTPClient Client;
  Client.setTimeout(Timeout);
  for (StringRef ServerUrl : DebuginfodUrls) {
    SmallString<64> ArtifactUrl;
    sys::path::append(ArtifactUrl, sys::path::Style::posix, ServerUrl, UrlPath);

    // Perform the HTTP request and if successful, write the response body to
    // the cache.
    StreamedHTTPResponseHandler Handler([&]() { return CacheAddStream(Task); },
                                        Client);
    HTTPRequest Request(ArtifactUrl);
    Error Err = Client.perform(Request, Handler);
    if (Err)
      return std::move(Err);

    if (Client.responseCode() != 200)
      continue;

    // Return the path to the artifact on disk.
    return std::string(AbsCachedArtifactPath);
  }

  return createStringError(errc::argument_out_of_domain, "build id not found");
}

DebuginfodLogEntry::DebuginfodLogEntry(const Twine &Message)
    : Message(Message.str()) {}

void DebuginfodLog::push(const Twine &Message) {
  push(DebuginfodLogEntry(Message));
}

void DebuginfodLog::push(DebuginfodLogEntry Entry) {
  {
    std::lock_guard<std::mutex> Guard(QueueMutex);
    LogEntryQueue.push(Entry);
  }
  QueueCondition.notify_one();
}

DebuginfodLogEntry DebuginfodLog::pop() {
  {
    std::unique_lock<std::mutex> Guard(QueueMutex);
    // Wait for messages to be pushed into the queue.
    QueueCondition.wait(Guard, [&] { return !LogEntryQueue.empty(); });
  }
  std::lock_guard<std::mutex> Guard(QueueMutex);
  if (!LogEntryQueue.size())
    llvm_unreachable("Expected message in the queue.");

  DebuginfodLogEntry Entry = LogEntryQueue.front();
  LogEntryQueue.pop();
  return Entry;
}

DebuginfodCollection::DebuginfodCollection(ArrayRef<StringRef> PathsRef,
                                           DebuginfodLog &Log, ThreadPool &Pool,
                                           double MinInterval)
    : Log(Log), Pool(Pool), MinInterval(MinInterval) {
  for (StringRef Path : PathsRef)
    Paths.push_back(Path.str());
}

Error DebuginfodCollection::update() {
  std::lock_guard<sys::Mutex> Guard(UpdateMutex);
  if (UpdateTimer.isRunning())
    UpdateTimer.stopTimer();
  UpdateTimer.clear();
  for (const std::string &Path : Paths) {
    Log.push("Updating binaries at path " + Path);
    if (Error Err = findBinaries(Path))
      return Err;
  }
  Log.push("Updated collection");
  UpdateTimer.startTimer();
  return Error::success();
}

Expected<bool> DebuginfodCollection::updateIfStale() {
  if (!UpdateTimer.isRunning())
    return false;
  UpdateTimer.stopTimer();
  double Time = UpdateTimer.getTotalTime().getWallTime();
  UpdateTimer.startTimer();
  if (Time < MinInterval)
    return false;
  if (Error Err = update())
    return std::move(Err);
  return true;
}

Error DebuginfodCollection::updateForever(std::chrono::milliseconds Interval) {
  while (true) {
    if (Error Err = update())
      return Err;
    std::this_thread::sleep_for(Interval);
  }
  llvm_unreachable("updateForever loop should never end");
}

static bool isDebugBinary(object::ObjectFile *Object) {
  // TODO: handle PDB debuginfo
  std::unique_ptr<DWARFContext> Context = DWARFContext::create(
      *Object, DWARFContext::ProcessDebugRelocations::Process);
  const DWARFObject &DObj = Context->getDWARFObj();
  unsigned NumSections = 0;
  DObj.forEachInfoSections([&](const DWARFSection &S) { NumSections++; });
  return NumSections;
}

static bool hasELFMagic(StringRef FilePath) {
  file_magic Type;
  std::error_code EC = identify_magic(FilePath, Type);
  if (EC)
    return false;
  switch (Type) {
  case file_magic::elf:
  case file_magic::elf_relocatable:
  case file_magic::elf_executable:
  case file_magic::elf_shared_object:
  case file_magic::elf_core:
    return true;
  default:
    return false;
  }
}

Error DebuginfodCollection::findBinaries(StringRef Path) {
  std::error_code EC;
  sys::fs::recursive_directory_iterator I(Twine(Path), EC), E;
  std::mutex IteratorMutex;
  ThreadPoolTaskGroup IteratorGroup(Pool);
  for (unsigned WorkerIndex = 0; WorkerIndex < Pool.getThreadCount();
       WorkerIndex++) {
    IteratorGroup.async([&, this]() -> void {
      std::string FilePath;
      while (true) {
        {
          // Check if iteration is over or there is an error during iteration
          std::lock_guard<std::mutex> Guard(IteratorMutex);
          if (I == E || EC)
            return;
          // Grab a file path from the directory iterator and advance the
          // iterator.
          FilePath = I->path();
          I.increment(EC);
        }

        // Inspect the file at this path to determine if it is debuginfo.
        if (!hasELFMagic(FilePath))
          continue;

        Expected<object::OwningBinary<object::Binary>> BinOrErr =
            object::createBinary(FilePath);

        if (!BinOrErr) {
          consumeError(BinOrErr.takeError());
          continue;
        }
        object::Binary *Bin = std::move(BinOrErr.get().getBinary());
        if (!Bin->isObject())
          continue;

        // TODO: Support non-ELF binaries
        object::ELFObjectFileBase *Object =
            dyn_cast<object::ELFObjectFileBase>(Bin);
        if (!Object)
          continue;

        Optional<BuildIDRef> ID = symbolize::getBuildID(Object);
        if (!ID)
          continue;

        std::string IDString = buildIDToString(ID.getValue());
        if (isDebugBinary(Object)) {
          std::lock_guard<sys::RWMutex> DebugBinariesGuard(DebugBinariesMutex);
          DebugBinaries[IDString] = FilePath;
        } else {
          std::lock_guard<sys::RWMutex> BinariesGuard(BinariesMutex);
          Binaries[IDString] = FilePath;
        }
      }
    });
  }
  IteratorGroup.wait();
  std::unique_lock<std::mutex> Guard(IteratorMutex);
  if (EC)
    return errorCodeToError(EC);
  return Error::success();
}

Expected<Optional<std::string>>
DebuginfodCollection::getBinaryPath(BuildIDRef ID) {
  Log.push("getting binary path of ID " + buildIDToString(ID));
  std::shared_lock<sys::RWMutex> Guard(BinariesMutex);
  auto Loc = Binaries.find(buildIDToString(ID));
  if (Loc != Binaries.end()) {
    std::string Path = Loc->getValue();
    return Path;
  }
  return None;
}

Expected<Optional<std::string>>
DebuginfodCollection::getDebugBinaryPath(BuildIDRef ID) {
  Log.push("getting debug binary path of ID " + buildIDToString(ID));
  std::shared_lock<sys::RWMutex> Guard(DebugBinariesMutex);
  auto Loc = DebugBinaries.find(buildIDToString(ID));
  if (Loc != DebugBinaries.end()) {
    std::string Path = Loc->getValue();
    return Path;
  }
  return None;
}

Expected<std::string> DebuginfodCollection::findBinaryPath(BuildIDRef ID) {
  {
    // Check collection; perform on-demand update if stale.
    Expected<Optional<std::string>> PathOrErr = getBinaryPath(ID);
    if (!PathOrErr)
      return PathOrErr.takeError();
    Optional<std::string> Path = *PathOrErr;
    if (!Path) {
      Expected<bool> UpdatedOrErr = updateIfStale();
      if (!UpdatedOrErr)
        return UpdatedOrErr.takeError();
      if (*UpdatedOrErr) {
        // Try once more.
        PathOrErr = getBinaryPath(ID);
        if (!PathOrErr)
          return PathOrErr.takeError();
        Path = *PathOrErr;
      }
    }
    if (Path)
      return Path.getValue();
  }

  // Try federation.
  Expected<std::string> PathOrErr = getCachedOrDownloadExecutable(ID);
  if (!PathOrErr)
    consumeError(PathOrErr.takeError());

  // Fall back to debug binary.
  return findDebugBinaryPath(ID);
}

Expected<std::string> DebuginfodCollection::findDebugBinaryPath(BuildIDRef ID) {
  // Check collection; perform on-demand update if stale.
  Expected<Optional<std::string>> PathOrErr = getDebugBinaryPath(ID);
  if (!PathOrErr)
    return PathOrErr.takeError();
  Optional<std::string> Path = *PathOrErr;
  if (!Path) {
    Expected<bool> UpdatedOrErr = updateIfStale();
    if (!UpdatedOrErr)
      return UpdatedOrErr.takeError();
    if (*UpdatedOrErr) {
      // Try once more.
      PathOrErr = getBinaryPath(ID);
      if (!PathOrErr)
        return PathOrErr.takeError();
      Path = *PathOrErr;
    }
  }
  if (Path)
    return Path.getValue();

  // Try federation.
  return getCachedOrDownloadDebuginfo(ID);
}

DebuginfodServer::DebuginfodServer(DebuginfodLog &Log,
                                   DebuginfodCollection &Collection)
    : Log(Log), Collection(Collection) {
  cantFail(
      Server.get(R"(/buildid/(.*)/debuginfo)", [&](HTTPServerRequest Request) {
        Log.push("GET " + Request.UrlPath);
        std::string IDString;
        if (!tryGetFromHex(Request.UrlPathMatches[0], IDString)) {
          Request.setResponse(
              {404, "text/plain", "Build ID is not a hex string\n"});
          return;
        }
        BuildID ID(IDString.begin(), IDString.end());
        Expected<std::string> PathOrErr = Collection.findDebugBinaryPath(ID);
        if (Error Err = PathOrErr.takeError()) {
          consumeError(std::move(Err));
          Request.setResponse({404, "text/plain", "Build ID not found\n"});
          return;
        }
        streamFile(Request, *PathOrErr);
      }));
  cantFail(
      Server.get(R"(/buildid/(.*)/executable)", [&](HTTPServerRequest Request) {
        Log.push("GET " + Request.UrlPath);
        std::string IDString;
        if (!tryGetFromHex(Request.UrlPathMatches[0], IDString)) {
          Request.setResponse(
              {404, "text/plain", "Build ID is not a hex string\n"});
          return;
        }
        BuildID ID(IDString.begin(), IDString.end());
        Expected<std::string> PathOrErr = Collection.findBinaryPath(ID);
        if (Error Err = PathOrErr.takeError()) {
          consumeError(std::move(Err));
          Request.setResponse({404, "text/plain", "Build ID not found\n"});
          return;
        }
        streamFile(Request, *PathOrErr);
      }));
}

} // namespace llvm
