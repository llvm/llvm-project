//===- CodeViewError.cpp - Error extensions for CodeView --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace llvm::codeview;

namespace {
// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class CodeViewErrorCategory : public std::error_category {
public:
  const char *name() const LLVM_NOEXCEPT override { return "llvm.codeview"; }

  std::string message(int Condition) const override {
    switch (static_cast<cv_error_code>(Condition)) {
    case cv_error_code::unspecified:
      return "An unknown error has occurred.";
    case cv_error_code::insufficient_buffer:
      return "The buffer is not large enough to read the requested number of "
             "bytes.";
    case cv_error_code::corrupt_record:
      return "The CodeView record is corrupted.";
    case cv_error_code::operation_unsupported:
      return "The requested operation is not supported.";
    }
    llvm_unreachable("Unrecognized cv_error_code");
  }
};
} // end anonymous namespace

static ManagedStatic<CodeViewErrorCategory> Category;

char CodeViewError::ID = 0;

CodeViewError::CodeViewError(cv_error_code C) : CodeViewError(C, "") {}

CodeViewError::CodeViewError(const std::string &Context)
    : CodeViewError(cv_error_code::unspecified, Context) {}

CodeViewError::CodeViewError(cv_error_code C, const std::string &Context)
    : Code(C) {
  ErrMsg = "CodeView Error: ";
  std::error_code EC = convertToErrorCode();
  if (Code != cv_error_code::unspecified)
    ErrMsg += EC.message() + "  ";
  if (!Context.empty())
    ErrMsg += Context;
}

void CodeViewError::log(raw_ostream &OS) const { OS << ErrMsg << "\n"; }

const std::string &CodeViewError::getErrorMessage() const { return ErrMsg; }

std::error_code CodeViewError::convertToErrorCode() const {
  return std::error_code(static_cast<int>(Code), *Category);
}
