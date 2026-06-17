//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Abstract class for LLVM tools that can be re-invoked in the same process.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/ToolInterface.h"

using namespace llvm;

namespace {
/// Wraps a `MemoryBuffer`, overriding the identifier returned from
/// `getBufferIdentifier`. This is needed to daemonize lots of tests which
/// include the buffer identifier in check lines as the module ID.
class NameOverrideMemoryBuffer : public MemoryBuffer {
public:
  NameOverrideMemoryBuffer(std::unique_ptr<MemoryBuffer> &&MB,
                           StringRef Identifier)
      : MemoryBuffer(), Inner(std::move(MB)), Identifier(Identifier) {
    init(Inner->getBufferStart(), Inner->getBufferEnd(),
         /*RequiresNullTerminator=*/false);
  }

  StringRef getBufferIdentifier() const override { return Identifier; }

  void dontNeedIfMmap() override { return Inner->dontNeedIfMmap(); }

  void willNeedIfMmap() override { return Inner->willNeedIfMmap(); }

  BufferKind getBufferKind() const override { return Inner->getBufferKind(); }

private:
  std::unique_ptr<MemoryBuffer> Inner;
  std::string Identifier;
};

} // namespace

ErrorOr<std::unique_ptr<MemoryBuffer>>
llvm::StandardInputSource::getInput() const {
  // Always pretend to be standard input, so that tests which are affected by
  // the buffer identifier do not get broken. (For example if there is a check
  // string that includes the module name)
  constexpr StringRef BufferIdentifier = "<stdin>";

  switch (SourceKind) {
  case Kind::Stdin: {
    return MemoryBuffer::getSTDIN();
  }
  case Kind::File: {
    auto Buffer = MemoryBuffer::getFile(StringValue, /*IsText=*/true);
    if (!Buffer)
      return Buffer.getError();

    return std::make_unique<NameOverrideMemoryBuffer>(std::move(*Buffer),
                                                      BufferIdentifier);
  }
  case Kind::String: {
    return MemoryBuffer::getMemBuffer(StringValue, BufferIdentifier);
  }
  }
}
