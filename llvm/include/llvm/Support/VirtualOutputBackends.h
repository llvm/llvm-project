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
#include "llvm/Support/VirtualOutputBackend.h"
#include "llvm/Support/VirtualOutputConfig.h"

namespace llvm::vfs {

/// Create a backend that ignores all output.
IntrusiveRefCntPtr<OutputBackend> makeNullOutputBackend();

/// Make a backend where \a OutputBackend::createFile() forwards to
/// \p UnderlyingBackend when \p Filter is true, and otherwise returns a
/// \a NullOutput.
IntrusiveRefCntPtr<OutputBackend> makeFilteringOutputBackend(
    IntrusiveRefCntPtr<OutputBackend> UnderlyingBackend,
    std::function<bool(StringRef, std::optional<OutputConfig>)> Filter);

/// Create a backend that forwards \a OutputBackend::createFile() to both \p
/// Backend1 and \p Backend2. Writing to such backend will create identical
/// outputs using two different backends.
IntrusiveRefCntPtr<OutputBackend>
makeMirroringOutputBackend(IntrusiveRefCntPtr<OutputBackend> Backend1,
                           IntrusiveRefCntPtr<OutputBackend> Backend2);

/// A helper class for proxying another backend, with the default
/// implementation to forward to the underlying backend.
class ProxyOutputBackend : public OutputBackend {
  void anchor() override;

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
  void anchor() override;

protected:
  IntrusiveRefCntPtr<OutputBackend> cloneImpl() const override {
    return clone();
  }

  Expected<std::unique_ptr<OutputFileImpl>>
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

} // namespace llvm::vfs

#endif // LLVM_SUPPORT_VIRTUALOUTPUTBACKENDS_H
