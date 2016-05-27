//===----- lib/Support/Error.cpp - Error and associated utilities ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"


using namespace llvm;

namespace {

  enum class ErrorErrorCode : int {
    MultipleErrors = 1,
    InconvertibleError
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
      case ErrorErrorCode::InconvertibleError:
        return "Inconvertible error value. An error has occurred that could "
               "not be converted to a known std::error_code. Please file a "
               "bug.";
      }
      llvm_unreachable("Unhandled error code");
    }
  };

}

static ManagedStatic<ErrorErrorCategory> ErrorErrorCat;

namespace llvm {

void ErrorInfoBase::anchor() {}
char ErrorInfoBase::ID = 0;
char ErrorList::ID = 0;
char ECError::ID = 0;
char StringError::ID = 0;


std::error_code ErrorList::convertToErrorCode() const {
  return std::error_code(static_cast<int>(ErrorErrorCode::MultipleErrors),
                         *ErrorErrorCat);
}

std::error_code inconvertibleErrorCode() {
  return std::error_code(static_cast<int>(ErrorErrorCode::InconvertibleError),
                         *ErrorErrorCat);
}

Error errorCodeToError(std::error_code EC) {
  if (!EC)
    return Error::success();
  return Error(llvm::make_unique<ECError>(ECError(EC)));
}

std::error_code errorToErrorCode(Error Err) {
  std::error_code EC;
  handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
    EC = EI.convertToErrorCode();
  });
  if (EC == inconvertibleErrorCode())
    report_fatal_error(EC.message());
  return EC;
}

StringError::StringError(const Twine &S, std::error_code EC)
    : Msg(S.str()), EC(EC) {}

void StringError::log(raw_ostream &OS) const { OS << Msg; }

std::error_code StringError::convertToErrorCode() const {
  return EC;
}

}
