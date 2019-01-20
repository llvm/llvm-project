//===- Error.cpp - system_error extensions for obj2yaml ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class _obj2yaml_error_category : public std::error_category {
public:
  const char *name() const noexcept override;
  std::string message(int ev) const override;
};
} // namespace

const char *_obj2yaml_error_category::name() const noexcept {
  return "obj2yaml";
}

std::string _obj2yaml_error_category::message(int ev) const {
  switch (static_cast<obj2yaml_error>(ev)) {
  case obj2yaml_error::success:
    return "Success";
  case obj2yaml_error::file_not_found:
    return "No such file.";
  case obj2yaml_error::unrecognized_file_format:
    return "Unrecognized file type.";
  case obj2yaml_error::unsupported_obj_file_format:
    return "Unsupported object file format.";
  case obj2yaml_error::not_implemented:
    return "Feature not yet implemented.";
  }
  llvm_unreachable("An enumerator of obj2yaml_error does not have a message "
                   "defined.");
}

namespace llvm {

const std::error_category &obj2yaml_category() {
  static _obj2yaml_error_category o;
  return o;
}

char Obj2YamlError::ID = 0;

void Obj2YamlError::log(raw_ostream &OS) const { OS << ErrMsg; }

std::error_code Obj2YamlError::convertToErrorCode() const {
  return std::error_code(static_cast<int>(Code), obj2yaml_category());
}

} // namespace llvm
