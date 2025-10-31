//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the VirtualOutputBackend types, including:
/// * NullOutputBackend: Outputs to NullOutputBackend are discarded.
/// * FilteringOutputBackend: Filter paths from output.
/// * MirroringOutputBackend: Mirror the output into two different backend.
/// * OnDiskOutputBackend: Write output files to disk.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualOutputBackends.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/VirtualOutputConfig.h"
#include "llvm/Support/VirtualOutputError.h"

using namespace llvm;
using namespace llvm::vfs;

void ProxyOutputBackend::anchor() {}
void OnDiskOutputBackend::anchor() {}

IntrusiveRefCntPtr<OutputBackend> vfs::makeNullOutputBackend() {
  struct NullOutputBackend : public OutputBackend {
    IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
      return const_cast<NullOutputBackend *>(this);
    }
    Expected<std::unique_ptr<OutputFileImpl>>
    createFileImpl(StringRef Path, std::optional<OutputConfig>) override {
      return std::make_unique<NullOutputFileImpl>();
    }
  };

  return makeIntrusiveRefCnt<NullOutputBackend>();
}

IntrusiveRefCntPtr<OutputBackend> vfs::makeFilteringOutputBackend(
    IntrusiveRefCntPtr<OutputBackend> UnderlyingBackend,
    std::function<bool(StringRef, std::optional<OutputConfig>)> Filter) {
  struct FilteringOutputBackend : public ProxyOutputBackend {
    Expected<std::unique_ptr<OutputFileImpl>>
    createFileImpl(StringRef Path,
                   std::optional<OutputConfig> Config) override {
      if (Filter(Path, Config))
        return ProxyOutputBackend::createFileImpl(Path, Config);
      return std::make_unique<NullOutputFileImpl>();
    }

    IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
      return makeIntrusiveRefCnt<FilteringOutputBackend>(
          getUnderlyingBackend().clone(), Filter);
    }

    FilteringOutputBackend(
        IntrusiveRefCntPtr<OutputBackend> UnderlyingBackend,
        std::function<bool(StringRef, std::optional<OutputConfig>)> Filter)
        : ProxyOutputBackend(std::move(UnderlyingBackend)),
          Filter(std::move(Filter)) {
      assert(this->Filter && "Expected a non-null function");
    }
    std::function<bool(StringRef, std::optional<OutputConfig>)> Filter;
  };

  return makeIntrusiveRefCnt<FilteringOutputBackend>(
      std::move(UnderlyingBackend), std::move(Filter));
}

IntrusiveRefCntPtr<OutputBackend>
vfs::makeMirroringOutputBackend(IntrusiveRefCntPtr<OutputBackend> Backend1,
                                IntrusiveRefCntPtr<OutputBackend> Backend2) {
  struct ProxyOutputBackend1 : public ProxyOutputBackend {
    using ProxyOutputBackend::ProxyOutputBackend;
  };
  struct ProxyOutputBackend2 : public ProxyOutputBackend {
    using ProxyOutputBackend::ProxyOutputBackend;
  };
  struct MirroringOutput final : public OutputFileImpl, raw_pwrite_stream {
    Error keep() final {
      flush();
      return joinErrors(F1->keep(), F2->keep());
    }
    Error discard() final {
      flush();
      return joinErrors(F1->discard(), F2->discard());
    }
    raw_pwrite_stream &getOS() final { return *this; }

    void write_impl(const char *Ptr, size_t Size) override {
      F1->getOS().write(Ptr, Size);
      F2->getOS().write(Ptr, Size);
    }
    void pwrite_impl(const char *Ptr, size_t Size, uint64_t Offset) override {
      this->flush();
      F1->getOS().pwrite(Ptr, Size, Offset);
      F2->getOS().pwrite(Ptr, Size, Offset);
    }
    uint64_t current_pos() const override { return F1->getOS().tell(); }
    size_t preferred_buffer_size() const override {
      return PreferredBufferSize;
    }
    void reserveExtraSpace(uint64_t ExtraSize) override {
      F1->getOS().reserveExtraSpace(ExtraSize);
      F2->getOS().reserveExtraSpace(ExtraSize);
    }
    bool is_displayed() const override {
      return F1->getOS().is_displayed() && F2->getOS().is_displayed();
    }
    bool has_colors() const override {
      return F1->getOS().has_colors() && F2->getOS().has_colors();
    }
    void enable_colors(bool enable) override {
      raw_pwrite_stream::enable_colors(enable);
      F1->getOS().enable_colors(enable);
      F2->getOS().enable_colors(enable);
    }

    MirroringOutput(std::unique_ptr<OutputFileImpl> F1,
                    std::unique_ptr<OutputFileImpl> F2)
        : PreferredBufferSize(std::max(F1->getOS().GetBufferSize(),
                                       F1->getOS().GetBufferSize())),
          F1(std::move(F1)), F2(std::move(F2)) {
      // Don't double buffer.
      this->F1->getOS().SetUnbuffered();
      this->F2->getOS().SetUnbuffered();
    }
    size_t PreferredBufferSize;
    std::unique_ptr<OutputFileImpl> F1;
    std::unique_ptr<OutputFileImpl> F2;
  };
  struct MirroringOutputBackend : public ProxyOutputBackend1,
                                  public ProxyOutputBackend2 {
    Expected<std::unique_ptr<OutputFileImpl>>
    createFileImpl(StringRef Path,
                   std::optional<OutputConfig> Config) override {
      std::unique_ptr<OutputFileImpl> File1;
      std::unique_ptr<OutputFileImpl> File2;
      if (Error E =
              ProxyOutputBackend1::createFileImpl(Path, Config).moveInto(File1))
        return std::move(E);
      if (Error E =
              ProxyOutputBackend2::createFileImpl(Path, Config).moveInto(File2))
        return joinErrors(std::move(E), File1->discard());

      // Skip the extra indirection if one of these is a null output.
      if (isa<NullOutputFileImpl>(*File1)) {
        consumeError(File1->discard());
        return std::move(File2);
      }
      if (isa<NullOutputFileImpl>(*File2)) {
        consumeError(File2->discard());
        return std::move(File1);
      }
      return std::make_unique<MirroringOutput>(std::move(File1),
                                               std::move(File2));
    }

    IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
      return IntrusiveRefCntPtr<ProxyOutputBackend1>(
          makeIntrusiveRefCnt<MirroringOutputBackend>(
              ProxyOutputBackend1::getUnderlyingBackend().clone(),
              ProxyOutputBackend2::getUnderlyingBackend().clone()));
    }
    void Retain() const { ProxyOutputBackend1::Retain(); }
    void Release() const { ProxyOutputBackend1::Release(); }

    MirroringOutputBackend(IntrusiveRefCntPtr<OutputBackend> Backend1,
                           IntrusiveRefCntPtr<OutputBackend> Backend2)
        : ProxyOutputBackend1(std::move(Backend1)),
          ProxyOutputBackend2(std::move(Backend2)) {}
  };

  assert(Backend1 && "Expected actual backend");
  assert(Backend2 && "Expected actual backend");
  return IntrusiveRefCntPtr<ProxyOutputBackend1>(
      makeIntrusiveRefCnt<MirroringOutputBackend>(std::move(Backend1),
                                                  std::move(Backend2)));
}

static OutputConfig
applySettings(std::optional<OutputConfig> &&Config,
              const OnDiskOutputBackend::OutputSettings &Settings) {
  if (!Config)
    Config = Settings.DefaultConfig;
  if (!Settings.UseTemporaries)
    Config->setNoAtomicWrite();
  if (!Settings.RemoveOnSignal)
    Config->setNoDiscardOnSignal();
  return *Config;
}

namespace {
class OnDiskOutputFile final : public OutputFileImpl {
public:
  Error keep() override;
  Error discard() override;
  raw_pwrite_stream &getOS() override {
    assert(FileOS && "Expected valid file");
    if (BufferOS)
      return *BufferOS;
    return *FileOS;
  }

  /// Attempt to open a temporary file for \p OutputPath.
  ///
  /// This tries to open a uniquely-named temporary file for \p OutputPath,
  /// possibly also creating any missing directories if \a
  /// OnDiskOutputConfig::UseTemporaryCreateMissingDirectories is set in \a
  /// Config.
  ///
  /// \post FD and \a TempPath are initialized if this is successful.
  Error tryToCreateTemporary(std::optional<int> &FD);

  Error initializeFile(std::optional<int> &FD);
  Error initializeStream();
  Error reset();

  OnDiskOutputFile(StringRef OutputPath, std::optional<OutputConfig> Config,
                   const OnDiskOutputBackend::OutputSettings &Settings)
      : Config(applySettings(std::move(Config), Settings)),
        OutputPath(OutputPath.str()) {}

  OutputConfig Config;
  const std::string OutputPath;
  std::optional<std::string> TempPath;
  std::optional<raw_fd_ostream> FileOS;
  std::optional<buffer_ostream> BufferOS;
};
} // end namespace

static Error createDirectoriesOnDemand(StringRef OutputPath,
                                       OutputConfig Config,
                                       llvm::function_ref<Error()> CreateFile) {
  return handleErrors(CreateFile(), [&](std::unique_ptr<ECError> EC) {
    if (EC->convertToErrorCode() != std::errc::no_such_file_or_directory ||
        Config.getNoImplyCreateDirectories())
      return Error(std::move(EC));

    StringRef ParentPath = sys::path::parent_path(OutputPath);
    if (std::error_code EC = sys::fs::create_directories(ParentPath))
      return make_error<OutputError>(ParentPath, EC);
    return CreateFile();
  });
}

Error OnDiskOutputFile::tryToCreateTemporary(std::optional<int> &FD) {
  // Create a temporary file.
  // Insert -%%%%%%%% before the extension (if any), and because some tools
  // (noticeable, clang's own GlobalModuleIndex.cpp) glob for build
  // artifacts, also append .tmp.
  StringRef OutputExtension = sys::path::extension(OutputPath);
  SmallString<128> ModelPath =
      StringRef(OutputPath).drop_back(OutputExtension.size());
  ModelPath += "-%%%%%%%%";
  ModelPath += OutputExtension;
  ModelPath += ".tmp";

  return createDirectoriesOnDemand(OutputPath, Config, [&]() -> Error {
    int NewFD;
    SmallString<128> UniquePath;
    if (std::error_code EC =
            sys::fs::createUniqueFile(ModelPath, NewFD, UniquePath))
      return make_error<TempFileOutputError>(ModelPath, OutputPath, EC);

    if (Config.getDiscardOnSignal())
      sys::RemoveFileOnSignal(UniquePath);

    TempPath = UniquePath.str().str();
    FD.emplace(NewFD);
    return Error::success();
  });
}

Error OnDiskOutputFile::initializeFile(std::optional<int> &FD) {
  assert(OutputPath != "-" && "Unexpected request for FD of stdout");

  // Disable temporary file for other non-regular files, and if we get a status
  // object, also check if we can write and disable write-through buffers if
  // appropriate.
  if (Config.getAtomicWrite()) {
    sys::fs::file_status Status;
    sys::fs::status(OutputPath, Status);
    if (sys::fs::exists(Status)) {
      if (!sys::fs::is_regular_file(Status))
        Config.setNoAtomicWrite();

      // Fail now if we can't write to the final destination.
      if (!sys::fs::can_write(OutputPath))
        return make_error<OutputError>(
            OutputPath,
            std::make_error_code(std::errc::operation_not_permitted));
    }
  }

  // If (still) using a temporary file, try to create it (and return success if
  // that works).
  if (Config.getAtomicWrite())
    if (!errorToBool(tryToCreateTemporary(FD)))
      return Error::success();

  // Not using a temporary file. Open the final output file.
  return createDirectoriesOnDemand(OutputPath, Config, [&]() -> Error {
    int NewFD;
    sys::fs::OpenFlags OF = sys::fs::OF_None;
    if (Config.getTextWithCRLF())
      OF |= sys::fs::OF_TextWithCRLF;
    else if (Config.getText())
      OF |= sys::fs::OF_Text;
    if (Config.getAppend())
      OF |= sys::fs::OF_Append;
    if (std::error_code EC = sys::fs::openFileForWrite(
            OutputPath, NewFD, sys::fs::CD_CreateAlways, OF))
      return convertToOutputError(OutputPath, EC);
    FD.emplace(NewFD);

    if (Config.getDiscardOnSignal())
      sys::RemoveFileOnSignal(OutputPath);
    return Error::success();
  });
}

Error OnDiskOutputFile::initializeStream() {
  // Open the file stream.
  if (OutputPath == "-") {
    std::error_code EC;
    FileOS.emplace(OutputPath, EC);
    if (EC)
      return make_error<OutputError>(OutputPath, EC);
  } else {
    std::optional<int> FD;
    if (Error E = initializeFile(FD))
      return E;
    FileOS.emplace(*FD, /*shouldClose=*/true);
  }

  // Buffer the stream if necessary.
  if (!FileOS->supportsSeeking() && !Config.getText())
    BufferOS.emplace(*FileOS);

  return Error::success();
}

namespace {
class OpenFileRAII {
  static const int InvalidFd = -1;

public:
  int Fd = InvalidFd;

  ~OpenFileRAII() {
    if (Fd != InvalidFd)
      llvm::sys::Process::SafelyCloseFileDescriptor(Fd);
  }
};

enum class FileDifference : uint8_t {
  /// The source and destination paths refer to the exact same file.
  IdenticalFile,
  /// The source and destination paths refer to separate files with identical
  /// contents.
  SameContents,
  /// The source and destination paths refer to separate files with different
  /// contents.
  DifferentContents
};
} // end anonymous namespace

static Expected<FileDifference>
areFilesDifferent(const llvm::Twine &Source, const llvm::Twine &Destination) {
  if (sys::fs::equivalent(Source, Destination))
    return FileDifference::IdenticalFile;

  OpenFileRAII SourceFile;
  sys::fs::file_status SourceStatus;
  // If we can't open the source file, fail.
  if (std::error_code EC = sys::fs::openFileForRead(Source, SourceFile.Fd))
    return convertToOutputError(Source, EC);

  // If we can't stat the source file, fail.
  if (std::error_code EC = sys::fs::status(SourceFile.Fd, SourceStatus))
    return convertToOutputError(Source, EC);

  OpenFileRAII DestFile;
  sys::fs::file_status DestStatus;
  // If we can't open the destination file, report different.
  if (std::error_code Error =
          sys::fs::openFileForRead(Destination, DestFile.Fd))
    return FileDifference::DifferentContents;

  // If we can't open the destination file, report different.
  if (std::error_code Error = sys::fs::status(DestFile.Fd, DestStatus))
    return FileDifference::DifferentContents;

  // If the files are different sizes, they must be different.
  uint64_t Size = SourceStatus.getSize();
  if (Size != DestStatus.getSize())
    return FileDifference::DifferentContents;

  // If both files are zero size, they must be the same.
  if (Size == 0)
    return FileDifference::SameContents;

  // The two files match in size, so we have to compare the bytes to determine
  // if they're the same.
  std::error_code SourceRegionErr;
  sys::fs::mapped_file_region SourceRegion(
      sys::fs::convertFDToNativeFile(SourceFile.Fd),
      sys::fs::mapped_file_region::readonly, Size, 0, SourceRegionErr);
  if (SourceRegionErr)
    return convertToOutputError(Source, SourceRegionErr);

  std::error_code DestRegionErr;
  sys::fs::mapped_file_region DestRegion(
      sys::fs::convertFDToNativeFile(DestFile.Fd),
      sys::fs::mapped_file_region::readonly, Size, 0, DestRegionErr);

  if (DestRegionErr)
    return FileDifference::DifferentContents;

  if (memcmp(SourceRegion.const_data(), DestRegion.const_data(), Size) != 0)
    return FileDifference::DifferentContents;

  return FileDifference::SameContents;
}

Error OnDiskOutputFile::reset() {
  // Destroy the streams to flush them.
  BufferOS.reset();
  if (!FileOS)
    return Error::success();

  // Remember the error in raw_fd_ostream to be reported later.
  std::error_code EC = FileOS->error();
  // Clear the error to avoid fatal error when reset.
  FileOS->clear_error();
  FileOS.reset();
  return errorCodeToError(EC);
}

Error OnDiskOutputFile::keep() {
  if (auto E = reset())
    return E;

  // Close the file descriptor and remove crash cleanup before exit.
  auto RemoveDiscardOnSignal = make_scope_exit([&]() {
    if (Config.getDiscardOnSignal())
      sys::DontRemoveFileOnSignal(TempPath ? *TempPath : OutputPath);
  });

  if (!TempPath)
    return Error::success();

  // See if we should append instead of move.
  if (Config.getAppend() && OutputPath != "-") {
    // Read TempFile for the content to append.
    auto Content = MemoryBuffer::getFile(*TempPath);
    if (!Content)
      return convertToTempFileOutputError(*TempPath, OutputPath,
                                          Content.getError());
    while (1) {
      // Attempt to lock the output file.
      // Only one process is allowed to append to this file at a time.
      llvm::LockFileManager Lock(OutputPath);
      bool Owned;
      if (Error Err = Lock.tryLock().moveInto(Owned)) {
        // If we error acquiring a lock, we cannot ensure appends
        // to the trace file are atomic - cannot ensure output correctness.
        Lock.unsafeMaybeUnlock();
        return convertToOutputError(
            OutputPath, std::make_error_code(std::errc::no_lock_available));
      }
      if (Owned) {
        // Lock acquired, perform the write and release the lock.
        std::error_code EC;
        llvm::raw_fd_ostream Out(OutputPath, EC, llvm::sys::fs::OF_Append);
        if (EC)
          return convertToOutputError(OutputPath, EC);
        Out << (*Content)->getBuffer();
        Out.close();
        Lock.unsafeMaybeUnlock();
        if (Out.has_error())
          return convertToOutputError(OutputPath, Out.error());
        // Remove temp file and done.
        (void)sys::fs::remove(*TempPath);
        return Error::success();
      }
      // Someone else owns the lock on this file, wait.
      switch (Lock.waitForUnlockFor(std::chrono::seconds(256))) {
      case WaitForUnlockResult::Success:
        [[fallthrough]];
      case WaitForUnlockResult::OwnerDied: {
        continue; // try again to get the lock.
      }
      case WaitForUnlockResult::Timeout: {
        // We could error on timeout to avoid potentially hanging forever, but
        // it may be more likely that an interrupted process failed to clear
        // the lock, causing other waiting processes to time-out. Let's clear
        // the lock and try again right away. If we do start seeing compiler
        // hangs in this location, we will need to re-consider.
        Lock.unsafeMaybeUnlock();
        continue;
      }
      }
      break;
    }
  }

  if (Config.getOnlyIfDifferent()) {
    auto Result = areFilesDifferent(*TempPath, OutputPath);
    if (!Result)
      return Result.takeError();
    switch (*Result) {
    case FileDifference::IdenticalFile:
      // Do nothing for a self-move.
      return Error::success();

    case FileDifference::SameContents:
      // Files are identical; remove the source file.
      (void)sys::fs::remove(*TempPath);
      return Error::success();

    case FileDifference::DifferentContents:
      break; // Rename the file.
    }
  }

  // Move temporary to the final output path and remove it if that fails.
  std::error_code RenameEC = sys::fs::rename(*TempPath, OutputPath);
  if (!RenameEC)
    return Error::success();

  // FIXME: TempPath should be in the same directory as OutputPath but try to
  // copy the output to see if makes any difference. If this path is used,
  // investigate why we need to copy.
  RenameEC = sys::fs::copy_file(*TempPath, OutputPath);
  (void)sys::fs::remove(*TempPath);

  if (!RenameEC)
    return Error::success();

  return make_error<TempFileOutputError>(*TempPath, OutputPath, RenameEC);
}

Error OnDiskOutputFile::discard() {
  // Destroy the streams to flush them.
  if (auto E = reset())
    return E;

  // Nothing on the filesystem to remove for stdout.
  if (OutputPath == "-")
    return Error::success();

  auto discardPath = [&](StringRef Path) {
    std::error_code EC = sys::fs::remove(Path);
    sys::DontRemoveFileOnSignal(Path);
    return EC;
  };

  // Clean up the file that's in-progress.
  if (!TempPath)
    return convertToOutputError(OutputPath, discardPath(OutputPath));
  return convertToTempFileOutputError(*TempPath, OutputPath,
                                      discardPath(*TempPath));
}

Error OnDiskOutputBackend::makeAbsolute(SmallVectorImpl<char> &Path) const {
  return convertToOutputError(StringRef(Path.data(), Path.size()),
                              sys::fs::make_absolute(Path));
}

Expected<std::unique_ptr<OutputFileImpl>>
OnDiskOutputBackend::createFileImpl(StringRef Path,
                                    std::optional<OutputConfig> Config) {
  SmallString<256> AbsPath;
  if (Path != "-") {
    AbsPath = Path;
    if (Error E = makeAbsolute(AbsPath))
      return std::move(E);
    Path = AbsPath;
  }

  auto File = std::make_unique<OnDiskOutputFile>(Path, Config, Settings);
  if (Error E = File->initializeStream())
    return std::move(E);

  return std::move(File);
}
