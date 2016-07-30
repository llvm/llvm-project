//===- MSFError.h - Error extensions for MSF Files --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_MSFERROR_H
#define LLVM_DEBUGINFO_MSF_MSFERROR_H

#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
namespace msf {
enum class msf_error_code {
  unspecified = 1,
  insufficient_buffer,
  not_writable,
  no_stream,
  invalid_format,
  block_in_use
};

/// Base class for errors originating when parsing raw PDB files
class MSFError : public ErrorInfo<MSFError> {
public:
  static char ID;
  MSFError(msf_error_code C);
  MSFError(const std::string &Context);
  MSFError(msf_error_code C, const std::string &Context);

  void log(raw_ostream &OS) const override;
  const std::string &getErrorMessage() const;
  std::error_code convertToErrorCode() const override;

private:
  std::string ErrMsg;
  msf_error_code Code;
};
} // namespace msf
} // namespace llvm

#endif // LLVM_DEBUGINFO_MSF_MSFERROR_H
