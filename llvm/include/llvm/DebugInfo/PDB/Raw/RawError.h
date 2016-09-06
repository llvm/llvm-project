//===- RawError.h - Error extensions for raw PDB implementation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_RAWERROR_H
#define LLVM_DEBUGINFO_PDB_RAW_RAWERROR_H

#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
namespace pdb {
enum class raw_error_code {
  unspecified = 1,
  feature_unsupported,
  invalid_format,
  corrupt_file,
  insufficient_buffer,
  no_stream,
  index_out_of_bounds,
  invalid_block_address,
  duplicate_entry,
  no_entry,
  not_writable,
  invalid_tpi_hash,
};

/// Base class for errors originating when parsing raw PDB files
class RawError : public ErrorInfo<RawError> {
public:
  static char ID;
  RawError(raw_error_code C);
  RawError(const std::string &Context);
  RawError(raw_error_code C, const std::string &Context);

  void log(raw_ostream &OS) const override;
  const std::string &getErrorMessage() const;
  std::error_code convertToErrorCode() const override;

private:
  std::string ErrMsg;
  raw_error_code Code;
};
}
}
#endif
