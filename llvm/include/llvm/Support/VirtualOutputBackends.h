//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the concrete VirtualOutputBackend
/// classes, which are the implementation for different output style and
/// functions. This file contains:
/// * NullOutputBackend: discard all outputs written.
/// * OnDiskOutputBackend: write output to disk, with support for common output
/// types, like append, or atomic update.
/// * FilteringOutputBackend: filer some output paths to underlying output
/// backend.
/// * MirrorOutputBackend: mirror output to two output backends.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_VIRTUALOUTPUTBACKENDS_H
#define LLVM_SUPPORT_VIRTUALOUTPUTBACKENDS_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/VirtualOutputBackend.h"
#include "llvm/Support/VirtualOutputConfig.h"
#include "llvm/Support/VirtualOutputError.h"
#include <map>

namespace llvm::vfs {

/// Create a backend that ignores all output.
LLVM_ABI IntrusiveRefCntPtr<OutputBackend> makeNullOutputBackend();

/// Make a backend where \a OutputBackend::createFile() forwards to
/// \p UnderlyingBackend when \p Filter is true, and otherwise returns a
/// \a NullOutput.
LLVM_ABI IntrusiveRefCntPtr<OutputBackend> makeFilteringOutputBackend(
    IntrusiveRefCntPtr<OutputBackend> UnderlyingBackend,
    std::function<bool(StringRef, std::optional<OutputConfig>)> Filter);

/// Create a backend that forwards \a OutputBackend::createFile() to both \p
/// Backend1 and \p Backend2. Writing to such backend will create identical
/// outputs using two different backends.
LLVM_ABI IntrusiveRefCntPtr<OutputBackend>
makeMirroringOutputBackend(IntrusiveRefCntPtr<OutputBackend> Backend1,
                           IntrusiveRefCntPtr<OutputBackend> Backend2);

/// A helper class for proxying another backend, with the default
/// implementation to forward to the underlying backend.
class ProxyOutputBackend : public OutputBackend {
  LLVM_ABI void anchor() override;

protected:
  // Require subclass to implement cloneImpl().
  //
  // IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override;

  Expected<std::unique_ptr<OutputFileImpl>>
  createFileImpl(StringRef Path, std::optional<OutputConfig> Config) override {
    OutputFile File;
    if (Error E = UnderlyingBackend->createFile(Path, Config).moveInto(File))
      return std::move(E);
    return File.takeImpl();
  }

  OutputBackend &getUnderlyingBackend() const { return *UnderlyingBackend; }

public:
  ProxyOutputBackend(IntrusiveRefCntPtr<OutputBackend> UnderlyingBackend)
      : UnderlyingBackend(std::move(UnderlyingBackend)) {
    assert(this->UnderlyingBackend && "Expected non-null backend");
  }

private:
  IntrusiveRefCntPtr<OutputBackend> UnderlyingBackend;
};

/// An output backend that creates files on disk, wrapping APIs in sys::fs.
class OnDiskOutputBackend : public OutputBackend {
  LLVM_ABI void anchor() override;

protected:
  IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
    return clone();
  }

  LLVM_ABI Expected<std::unique_ptr<OutputFileImpl>>
  createFileImpl(StringRef Path, std::optional<OutputConfig> Config) override;

public:
  /// Resolve an absolute path.
  Error makeAbsolute(SmallVectorImpl<char> &Path) const;

  /// On disk output settings.
  struct OutputSettings {
    /// Register output files to be deleted if a signal is received. Also
    /// enabled for outputs with \a OutputConfig::getDiscardOnSignal().
    bool RemoveOnSignal = true;

    /// Use temporary files. Also enabled for outputs with \a
    /// OutputConfig::getAtomicWrite().
    bool UseTemporaries = true;

    // Default configuration for this backend.
    OutputConfig DefaultConfig;
  };

  IntrusiveRefCntPtr<OnDiskOutputBackend> clone() const {
    auto Clone = makeIntrusiveRefCnt<OnDiskOutputBackend>();
    Clone->Settings = Settings;
    return Clone;
  }

  OnDiskOutputBackend() = default;

  /// Settings for this backend.
  ///
  /// Access is not thread-safe.
  OutputSettings Settings;
};

/// And output backend that creates files in memory, backed by string buffers in
/// a map. This is useful for unittesting.
class InMemoryOutputBackend : public OutputBackend {
protected:
  Expected<std::unique_ptr<OutputFileImpl>>
  createFileImpl(StringRef Path, std::optional<OutputConfig>) override {
    auto [It, Inserted] = Buffers.try_emplace(Path.str());
    if (Inserted)
      return std::make_unique<StringBackedOutputFileImpl>(It->second);
    return make_error<OutputError>(
        Path, std::make_error_code(std::errc::file_exists));
  }

  IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
    return const_cast<InMemoryOutputBackend *>(this);
  }

public:
  auto getBuffers() const & { return llvm::iterator_range(Buffers); }

private:
  LLVM_ABI void anchor() override;

  std::map<std::string, llvm::SmallString<100>> Buffers;
};

} // namespace llvm::vfs

#endif // LLVM_SUPPORT_VIRTUALOUTPUTBACKENDS_H
