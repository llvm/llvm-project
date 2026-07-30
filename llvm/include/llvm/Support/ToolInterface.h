//===- llvm/Support/ToolInterface.h - Abstract tool interface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract class for LLVM tools that can be re-invoked in the same process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TOOLINTERFACE_H
#define LLVM_SUPPORT_TOOLINTERFACE_H

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace llvm {

/// Many tools operate on standard input, which is problematic for in-process
/// execution/daemonization as the process may not want to close its standard
/// input stream. This class allows the tools to function as if a different
/// source, either an in-memory string or a file on disk, were standard input.
/// Tool implementations should read from this class instead of standard input,
/// for example using `StandardInputSource::getInput` instead of
/// `MemoryBuffer::getSTDIN` and `StandardInputSource::getFileOrInput` instead
/// of `MemoryBuffer::getFileOrSTDIN`.
class LLVM_ABI StandardInputSource final {
public:
  /// Construct a `StandardInputSource` that draws input from the process's
  /// standard input stream. This is used for regular tool invocations.
  static StandardInputSource fromStdin() {
    return StandardInputSource(Kind::Stdin, std::string());
  }

  /// Construct a `StandardInputSource` that draws input from a file.
  static StandardInputSource fromFile(std::string &&Filename) {
    return StandardInputSource(Kind::File, std::move(Filename));
  }

  /// Construct a `StandardInputSource` that pretends that the given string
  /// is the contents of standard input.
  static StandardInputSource fromString(std::string &&StringValue) {
    return StandardInputSource(Kind::String, std::move(StringValue));
  }

  /// Replacement for `MemoryBuffer::getSTDIN`. The returned memory buffer
  /// always has the identifier "<stdin>".
  ErrorOr<std::unique_ptr<MemoryBuffer>> getInput() const;

  /// Replacement for `MemoryBuffer::getFileOrInput`. Returns `getInput()` if
  /// the file name is "-", otherwise returns a memory buffer of the file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> getFileOrInput(StringRef Filename,
                                                        bool IsText) const {
    if (Filename == "-")
      return getInput();

    return MemoryBuffer::getFile(Filename, IsText);
  }

private:
  enum class Kind {
    Stdin,
    File,
    String,
  };

  StandardInputSource(Kind SourceKind, std::string &&StringValue)
      : SourceKind(SourceKind), StringValue(StringValue) {}

  Kind SourceKind;
  std::string StringValue;
};

/// This class represents a shared interface for LLVM tools that can be
/// re-invoked in the same process, for example when run by the daemon driver.
class LLVM_ABI LLVMTool {
public:
  virtual ~LLVMTool() = default;

  /// This is the function called to run the tool with a given input and set of
  /// arguments. All code to run the tool except for one-time initialization and
  /// finalization (for example `InitLLVM`) should be done in this function.
  ///
  /// Note that as `run` may use global state, it is not thread-safe.
  virtual int run(int Argc, char **Argv,
                  const StandardInputSource &InputSource) = 0;

  /// This function is called between `run` invocations to reset any persistent
  /// state, for example static command line options, statistics and debug
  /// counters, for the next invocation. This function is not called when the
  /// tool is run normally.
  virtual void resetState() = 0;
};
} // namespace llvm

#endif // LLVM_SUPPORT_TOOLINTERFACE_H
