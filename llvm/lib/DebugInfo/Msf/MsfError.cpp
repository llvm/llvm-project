//===- MsfError.cpp - Error extensions for Msf files ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Msf/MsfError.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace llvm::msf;

namespace {
// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class MsfErrorCategory : public std::error_category {
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

static ManagedStatic<MsfErrorCategory> Category;

char MsfError::ID = 0;

MsfError::MsfError(msf_error_code C) : MsfError(C, "") {}

MsfError::MsfError(const std::string &Context)
    : MsfError(msf_error_code::unspecified, Context) {}

MsfError::MsfError(msf_error_code C, const std::string &Context) : Code(C) {
  ErrMsg = "Msf Error: ";
  std::error_code EC = convertToErrorCode();
  if (Code != msf_error_code::unspecified)
    ErrMsg += EC.message() + "  ";
  if (!Context.empty())
    ErrMsg += Context;
}

void MsfError::log(raw_ostream &OS) const { OS << ErrMsg << "\n"; }

const std::string &MsfError::getErrorMessage() const { return ErrMsg; }

std::error_code MsfError::convertToErrorCode() const {
  return std::error_code(static_cast<int>(Code), *Category);
}
