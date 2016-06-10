//===- CodeViewError.h - Error extensions for CodeView ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_CODEVIEW_CODEVIEWERROR_H
#define LLVM_DEBUGINFO_PDB_CODEVIEW_CODEVIEWERROR_H

#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
namespace codeview {
enum class cv_error_code {
  unspecified = 1,
  insufficient_buffer,
  operation_unsupported,
  corrupt_record,
};

/// Base class for errors originating when parsing raw PDB files
class CodeViewError : public ErrorInfo<CodeViewError> {
public:
  static char ID;
  CodeViewError(cv_error_code C);
  CodeViewError(const std::string &Context);
  CodeViewError(cv_error_code C, const std::string &Context);

  void log(raw_ostream &OS) const override;
  const std::string &getErrorMessage() const;
  std::error_code convertToErrorCode() const override;

private:
  std::string ErrMsg;
  cv_error_code Code;
};
}
}
#endif
