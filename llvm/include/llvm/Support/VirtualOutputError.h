//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the OutputError class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_VIRTUALOUTPUTERROR_H
#define LLVM_SUPPORT_VIRTUALOUTPUTERROR_H

#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualOutputConfig.h"

namespace llvm::vfs {

const std::error_category &output_category();

enum class OutputErrorCode {
  // Error code 0 is absent. Use std::error_code() instead.
  not_closed = 1,
  invalid_config,
  already_closed,
  has_open_proxy,
};

inline std::error_code make_error_code(OutputErrorCode EV) {
  return std::error_code(static_cast<int>(EV), output_category());
}

/// Error related to an \a OutputFile. Derives from \a ECError and adds \a
/// getOutputPath().
class OutputError : public ErrorInfo<OutputError, ECError> {
  void anchor() override;

public:
  StringRef getOutputPath() const { return OutputPath; }
  void log(raw_ostream &OS) const override;

  // Used by ErrorInfo::classID.
  static char ID;

  OutputError(const Twine &OutputPath, std::error_code EC)
      : ErrorInfo<OutputError, ECError>(EC), OutputPath(OutputPath.str()) {
    assert(EC && "Cannot create OutputError from success EC");
  }

  OutputError(const Twine &OutputPath, OutputErrorCode EV)
      : ErrorInfo<OutputError, ECError>(make_error_code(EV)),
        OutputPath(OutputPath.str()) {
    assert(EC && "Cannot create OutputError from success EC");
  }

private:
  std::string OutputPath;
};

/// Return \a Error::success() or use \p OutputPath to create an \a
/// OutputError, depending on \p EC.
inline Error convertToOutputError(const Twine &OutputPath, std::error_code EC) {
  if (EC)
    return make_error<OutputError>(OutputPath, EC);
  return Error::success();
}

/// Error related to an OutputConfig for an \a OutputFile. Derives from \a
/// OutputError and adds \a getConfig().
class OutputConfigError : public ErrorInfo<OutputConfigError, OutputError> {
  void anchor() override;

public:
  OutputConfig getConfig() const { return Config; }
  void log(raw_ostream &OS) const override;

  // Used by ErrorInfo::classID.
  static char ID;

  OutputConfigError(OutputConfig Config, const Twine &OutputPath)
      : ErrorInfo<OutputConfigError, OutputError>(
            OutputPath, OutputErrorCode::invalid_config),
        Config(Config) {}

private:
  OutputConfig Config;
};

/// Error related to a temporary file for an \a OutputFile. Derives from \a
/// OutputError and adds \a getTempPath().
class TempFileOutputError : public ErrorInfo<TempFileOutputError, OutputError> {
  void anchor() override;

public:
  StringRef getTempPath() const { return TempPath; }
  void log(raw_ostream &OS) const override;

  // Used by ErrorInfo::classID.
  static char ID;

  TempFileOutputError(const Twine &TempPath, const Twine &OutputPath,
                      std::error_code EC)
      : ErrorInfo<TempFileOutputError, OutputError>(OutputPath, EC),
        TempPath(TempPath.str()) {}

  TempFileOutputError(const Twine &TempPath, const Twine &OutputPath,
                      OutputErrorCode EV)
      : ErrorInfo<TempFileOutputError, OutputError>(OutputPath, EV),
        TempPath(TempPath.str()) {}

private:
  std::string TempPath;
};

/// Return \a Error::success() or use \p TempPath and \p OutputPath to create a
/// \a TempFileOutputError, depending on \p EC.
inline Error convertToTempFileOutputError(const Twine &TempPath,
                                          const Twine &OutputPath,
                                          std::error_code EC) {
  if (EC)
    return make_error<TempFileOutputError>(TempPath, OutputPath, EC);
  return Error::success();
}

} // namespace llvm::vfs

#endif // LLVM_SUPPORT_VIRTUALOUTPUTERROR_H
