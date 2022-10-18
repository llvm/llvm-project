//===-- LVSupport.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines support functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSUPPORT_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSUPPORT_H

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <sstream>

namespace llvm {
namespace logicalview {

const int HEX_WIDTH = 12;
inline FormattedNumber hexValue(uint64_t N, unsigned Width = HEX_WIDTH,
                                bool Upper = false) {
  return format_hex(N, Width, Upper);
}

// Output the hexadecimal representation of 'Value' using '[0x%08x]' format.
inline std::string hexString(uint64_t Value, size_t Width = HEX_WIDTH) {
  std::string String;
  raw_string_ostream Stream(String);
  Stream << hexValue(Value, Width, false);
  return Stream.str();
}

// Get a hexadecimal string representation for the given value.
inline std::string hexSquareString(uint64_t Value) {
  return (Twine("[") + Twine(hexString(Value)) + Twine("]")).str();
}

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSUPPORT_H
