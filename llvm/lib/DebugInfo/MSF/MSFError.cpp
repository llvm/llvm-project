//===- MSFError.cpp - Error extensions for MSF files ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace llvm::msf;

namespace {
// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class MSFErrorCategory : public std::error_category {
public:
  const char *name() const LLVM_NOEXCEPT override { return "llvm.msf"; }

  std::string message(int Condition) const override {
    switch (static_cast<msf_error_code>(Condition)) {
    case msf_error_code::unspecified:
      return "An unknown error has occurred.";
    case msf_error_code::insufficient_buffer:
      return "The buffer is not large enough to read the requested number of "
             "bytes.";
    case msf_error_code::not_writable:
      return "The specified stream is not writable.";
    case msf_error_code::no_stream:
      return "The specified stream does not exist.";
    case msf_error_code::invalid_format:
      return "The data is in an unexpected format.";
    case msf_error_code::block_in_use:
      return "The block is already in use.";
    }
    llvm_unreachable("Unrecognized msf_error_code");
  }
};
} // end anonymous namespace

static ManagedStatic<MSFErrorCategory> Category;

char MSFError::ID = 0;

MSFError::MSFError(msf_error_code C) : MSFError(C, "") {}

MSFError::MSFError(const std::string &Context)
    : MSFError(msf_error_code::unspecified, Context) {}

MSFError::MSFError(msf_error_code C, const std::string &Context) : Code(C) {
  ErrMsg = "MSF Error: ";
  std::error_code EC = convertToErrorCode();
  if (Code != msf_error_code::unspecified)
    ErrMsg += EC.message() + "  ";
  if (!Context.empty())
    ErrMsg += Context;
}

void MSFError::log(raw_ostream &OS) const { OS << ErrMsg << "\n"; }

const std::string &MSFError::getErrorMessage() const { return ErrMsg; }

std::error_code MSFError::convertToErrorCode() const {
  return std::error_code(static_cast<int>(Code), *Category);
}
