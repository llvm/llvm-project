//===- Error.cpp - system_error extensions for PDB --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace llvm::pdb;

namespace {
// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class GenericErrorCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "llvm.pdb"; }

  std::string message(int Condition) const override {
    switch (static_cast<generic_error_code>(Condition)) {
    case generic_error_code::unspecified:
      return "An unknown error has occurred.";
    case generic_error_code::type_server_not_found:
      return "Type server PDB was not found.";
    case generic_error_code::dia_sdk_not_present:
      return "LLVM was not compiled with support for DIA.  This usually means "
             "that you are not using MSVC, or your Visual Studio "
             "installation "
             "is corrupt.";
    case generic_error_code::invalid_path:
      return "Unable to load PDB.  Make sure the file exists and is readable.";
    }
    llvm_unreachable("Unrecognized generic_error_code");
  }
};
} // end anonymous namespace

static ManagedStatic<GenericErrorCategory> Category;

char GenericError::ID = 0;

GenericError::GenericError(generic_error_code C) : GenericError(C, "") {}

GenericError::GenericError(StringRef Context)
    : GenericError(generic_error_code::unspecified, Context) {}

GenericError::GenericError(generic_error_code C, StringRef Context) : Code(C) {
  ErrMsg = "PDB Error: ";
  std::error_code EC = convertToErrorCode();
  if (Code != generic_error_code::unspecified)
    ErrMsg += EC.message() + "  ";
  if (!Context.empty())
    ErrMsg += Context;
}

void GenericError::log(raw_ostream &OS) const { OS << ErrMsg << "\n"; }

StringRef GenericError::getErrorMessage() const { return ErrMsg; }

std::error_code GenericError::convertToErrorCode() const {
  return std::error_code(static_cast<int>(Code), *Category);
}
