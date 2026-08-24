//===- llvm/IR/Formatters.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// @file
/// Describes an interface used to make textual form of IR more readable. It
/// allows to represent constant values as strings, for example:
///
/// \code
///    call i1 @llvm.is.fpclass.f32(float %x, i32 fc"pnormal pinf")
/// \endcode
///
/// where "fc" specifies a formatter. In IR this string would be replaced with
/// the constant value 768.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_FORMATTERS_H
#define LLVM_IR_FORMATTERS_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <string>

namespace llvm {

class Value;

class ParsedValue {
public:
  enum Kind { None, Int };

  ParsedValue() : ValKind(None) {}
  ~ParsedValue();

  bool isEmpty() const { return ValKind == None; }
  bool isInt() const { return ValKind == Int; }

  void clear();
  void setInt(const APSInt &V);

  APSInt getInt() const {
    assert(isInt());
    return IntVal;
  }

private:
  Kind ValKind;
  APSInt IntVal;
};

LLVM_ABI bool parseFormattedValue(StringRef Formatter, StringRef Str,
                                  ParsedValue &Val, std::string &ErrorMsg);

} // namespace llvm

#endif
