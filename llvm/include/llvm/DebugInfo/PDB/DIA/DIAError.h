//===- DIAError.h - Error extensions for PDB DIA implementation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_DIA_DIAERROR_H
#define LLVM_DEBUGINFO_PDB_DIA_DIAERROR_H

#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
namespace pdb {
enum class dia_error_code {
  unspecified = 1,
  could_not_create_impl,
  invalid_file_format,
  invalid_parameter,
  already_loaded,
  debug_info_mismatch,
};

/// Base class for errors originating in DIA SDK, e.g. COM calls
class DIAError : public ErrorInfo<DIAError> {
public:
  static char ID;
  DIAError(dia_error_code C);
  DIAError(const std::string &Context);
  DIAError(dia_error_code C, const std::string &Context);

  void log(raw_ostream &OS) const override;
  const std::string &getErrorMessage() const;
  std::error_code convertToErrorCode() const override;

private:
  std::string ErrMsg;
  dia_error_code Code;
};
}
}
#endif
