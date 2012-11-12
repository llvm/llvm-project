//===- Error.h - system_error extensions for Object -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This declares a new error_category for the Object library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ERROR_H
#define LLVM_OBJECT_ERROR_H

#include "llvm/Support/system_error.h"

namespace llvm {
namespace object {

const error_category &object_category();

struct object_error {
enum _ {
  success = 0,
  invalid_file_type,
  parse_failed,
  unexpected_eof
};
  _ v_;

  object_error(_ v) : v_(v) {}
  explicit object_error(int v) : v_(_(v)) {}
  operator int() const {return v_;}
};

inline error_code make_error_code(object_error e) {
  return error_code(static_cast<int>(e), object_category());
}

} // end namespace object.

template <> struct is_error_code_enum<object::object_error> : true_type { };

template <> struct is_error_code_enum<object::object_error::_> : true_type { };

} // end namespace llvm.

#endif
