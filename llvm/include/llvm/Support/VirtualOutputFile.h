//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the llvm::vfs::OutputFile class,
/// which is a virtualized output file from output backend. \c OutputFile can be
/// use a \c raw_pwrite_stream for writing, and are required to be `keep()` or
/// `discard()` in the end.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_VIRTUALOUTPUTFILE_H
#define LLVM_SUPPORT_VIRTUALOUTPUTFILE_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::vfs {

class OutputFileImpl : public RTTIExtends<OutputFileImpl, RTTIRoot> {
  void anchor() override;

public:
  static char ID;
  ~OutputFileImpl() override = default;

  virtual Error keep() = 0;
  virtual Error discard() = 0;
  virtual raw_pwrite_stream &getOS() = 0;
};

class NullOutputFileImpl final
    : public RTTIExtends<NullOutputFileImpl, OutputFileImpl> {
  void anchor() override;

public:
  static char ID;
  Error keep() final { return Error::success(); }
  Error discard() final { return Error::success(); }
  raw_pwrite_stream &getOS() final { return OS; }

private:
  raw_null_ostream OS;
};

/// A virtualized output file that writes to a specific backend.
///
/// One of \a keep(), \a discard(), or \a discardOnDestroy() must be called
/// before destruction.
class OutputFile {
public:
  StringRef getPath() const { return Path; }

  /// Check if \a keep() or \a discard() has already been called.
  bool isOpen() const { return bool(Impl); }

  explicit operator bool() const { return isOpen(); }

  raw_pwrite_stream &getOS() {
    assert(isOpen() && "Expected open output stream");
    return Impl->getOS();
  }
  operator raw_pwrite_stream &() { return getOS(); }
  template <class T> raw_ostream &operator<<(T &&V) {
    return getOS() << std::forward<T>(V);
  }

  /// Keep an output. Errors if this fails.
  ///
  /// If it has already been closed, calls \a report_fatal_error().
  ///
  /// If there's an open proxy from \a createProxy(), calls \a discard() to
  /// clean up temporaries followed by \a report_fatal_error().
  Error keep();

  /// Discard an output, cleaning up any temporary state. Errors if clean-up
  /// fails.
  ///
  /// If it has already been closed, calls \a report_fatal_error().
  Error discard();

  /// Discard the output when destroying it if it's still open, sending the
  /// result to \a Handler.
  void discardOnDestroy(unique_function<void(Error E)> Handler) {
    DiscardOnDestroyHandler = std::move(Handler);
  }

  /// Create a proxy stream for clients that need to pass an owned stream to a
  /// producer. Errors if there's already a proxy. The proxy must be deleted
  /// before calling \a keep(). The proxy will crash if it's written to after
  /// calling \a discard().
  Expected<std::unique_ptr<raw_pwrite_stream>> createProxy();

  bool hasOpenProxy() const { return OpenProxy; }

  /// Take the implementation.
  ///
  /// \pre \a hasOpenProxy() is false.
  /// \pre \a discardOnDestroy() has not been called.
  std::unique_ptr<OutputFileImpl> takeImpl() {
    assert(!hasOpenProxy() && "Unexpected open proxy");
    assert(!DiscardOnDestroyHandler && "Unexpected discard handler");
    return std::move(Impl);
  }

  /// Check whether this is a null output file.
  bool isNull() const { return Impl && isa<NullOutputFileImpl>(*Impl); }

  OutputFile() = default;

  explicit OutputFile(const Twine &Path, std::unique_ptr<OutputFileImpl> Impl)
      : Path(Path.str()), Impl(std::move(Impl)) {
    assert(this->Impl && "Expected open output file");
  }

  ~OutputFile() { destroy(); }
  OutputFile(OutputFile &&O) { moveFrom(O); }
  OutputFile &operator=(OutputFile &&O) {
    destroy();
    return moveFrom(O);
  }

private:
  /// Destroy \a Impl. Reports fatal error if the file is open and there's no
  /// handler from \a discardOnDestroy().
  void destroy();
  OutputFile &moveFrom(OutputFile &O) {
    Path = std::move(O.Path);
    Impl = std::move(O.Impl);
    DiscardOnDestroyHandler = std::move(O.DiscardOnDestroyHandler);
    OpenProxy = O.OpenProxy;
    O.OpenProxy = nullptr;
    return *this;
  }

  std::string Path;
  std::unique_ptr<OutputFileImpl> Impl;
  unique_function<void(Error E)> DiscardOnDestroyHandler;

  class TrackedProxy;
  TrackedProxy *OpenProxy = nullptr;
};

/// Update \p File to silently discard itself if it's still open when it's
/// destroyed.
inline void consumeDiscardOnDestroy(OutputFile &File) {
  File.discardOnDestroy(consumeError);
}

/// Update \p File to silently discard itself if it's still open when it's
/// destroyed.
inline Expected<OutputFile> consumeDiscardOnDestroy(Expected<OutputFile> File) {
  if (File)
    consumeDiscardOnDestroy(*File);
  return File;
}

} // namespace llvm::vfs

#endif // LLVM_SUPPORT_VIRTUALOUTPUTFILE_H
