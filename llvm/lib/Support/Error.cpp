//===----- lib/Support/Error.cpp - Error and associated utilities ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

namespace {

  enum class ErrorErrorCode {
    MultipleErrors
  };

  // FIXME: This class is only here to support the transition to llvm::Error. It
  // will be removed once this transition is complete. Clients should prefer to
  // deal with the Error value directly, rather than converting to error_code.
  class ErrorErrorCategory : public std::error_category {
  public:
    const char *name() const LLVM_NOEXCEPT override { return "Error"; }

    std::string message(int condition) const override {
      switch (static_cast<ErrorErrorCode>(condition)) {
      case ErrorErrorCode::MultipleErrors:
        return "Multiple errors";
      };
      llvm_unreachable("Unhandled error code");
    }
  };

}

void ErrorInfoBase::anchor() {}
char ErrorInfoBase::ID = 0;
char ErrorList::ID = 0;
char ECError::ID = 0;

static ManagedStatic<ErrorErrorCategory> ErrorErrorCat;

std::error_code ErrorList::convertToErrorCode() const {
  return std::error_code(static_cast<int>(ErrorErrorCode::MultipleErrors),
                         *ErrorErrorCat);
}
