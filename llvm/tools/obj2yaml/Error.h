//===- Error.h - system_error extensions for obj2yaml -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJ2YAML_ERROR_H
#define LLVM_TOOLS_OBJ2YAML_ERROR_H

#include "llvm/Support/Error.h"

#include <system_error>

namespace llvm {
const std::error_category &obj2yaml_category();

enum class obj2yaml_error {
  success = 0,
  file_not_found,
  unrecognized_file_format,
  unsupported_obj_file_format,
  not_implemented
};

inline std::error_code make_error_code(obj2yaml_error e) {
  return std::error_code(static_cast<int>(e), obj2yaml_category());
}

class Obj2YamlError : public ErrorInfo<Obj2YamlError> {
public:
  static char ID;
  Obj2YamlError(obj2yaml_error C) : Code(C) {}
  Obj2YamlError(std::string ErrMsg) : ErrMsg(std::move(ErrMsg)) {}
  Obj2YamlError(obj2yaml_error C, std::string ErrMsg)
      : ErrMsg(std::move(ErrMsg)), Code(C) {}
  void log(raw_ostream &OS) const override;
  const std::string &getErrorMessage() const { return ErrMsg; }
  std::error_code convertToErrorCode() const override;

private:
  std::string ErrMsg;
  obj2yaml_error Code = obj2yaml_error::success;
};

} // namespace llvm

namespace std {
template <> struct is_error_code_enum<llvm::obj2yaml_error> : std::true_type {};
}

#endif
