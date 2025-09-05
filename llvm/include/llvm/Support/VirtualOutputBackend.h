//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the VirtualOutputBackend class, which
/// can be used to virtualized output files from LLVM tools.
/// VirtualOutputBackend provides an unified interface to write outputs and a
/// configurable interface for tools to operate on those outputs.
/// VirtualOutputBackend contains basic implementations like writing to disk
/// with different configurations, or advanced logics like output filtering
/// and duplicating.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_VIRTUALOUTPUTBACKEND_H
#define LLVM_SUPPORT_VIRTUALOUTPUTBACKEND_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualOutputConfig.h"
#include "llvm/Support/VirtualOutputFile.h"

namespace llvm::vfs {

/// Interface for virtualized outputs.
///
/// If virtual functions are added here, also add them to \a
/// ProxyOutputBackend.
class OutputBackend : public RefCountedBase<OutputBackend> {
  virtual void anchor();

public:
  /// Get a backend that points to the same destination as this one but that
  /// has independent settings.
  ///
  /// Not thread-safe, but all operations are thread-safe when performed on
  /// separate clones of the same backend.
  IntrusiveRefCntPtr<OutputBackend> clone() const { return cloneImpl(); }

  /// Create a file. If \p Config is \c std::nullopt, uses the backend's default
  /// OutputConfig (may match \a OutputConfig::OutputConfig(), or may
  /// have been customized).
  ///
  /// Thread-safe.
  Expected<OutputFile>
  createFile(const Twine &Path,
             std::optional<OutputConfig> Config = std::nullopt);

protected:
  /// Must be thread-safe. Virtual function has a different name than \a
  /// clone() so that implementations can override the return value.
  virtual IntrusiveRefCntPtr<OutputBackend> cloneImpl() const = 0;

  /// Create a file for \p Path. Must be thread-safe.
  ///
  /// \pre \p Config is valid or std::nullopt.
  virtual Expected<std::unique_ptr<OutputFileImpl>>
  createFileImpl(StringRef Path, std::optional<OutputConfig> Config) = 0;

  OutputBackend() = default;

public:
  virtual ~OutputBackend() = default;
};

} // namespace llvm::vfs

#endif // LLVM_SUPPORT_VIRTUALOUTPUTBACKEND_H
