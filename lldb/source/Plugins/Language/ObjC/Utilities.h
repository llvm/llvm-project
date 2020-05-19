//===-- Utilities.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_LANGUAGE_OBJC_UTILITIES_H
#define LLDB_PLUGINS_LANGUAGE_OBJC_UTILITIES_H

namespace lldb_private {

class Stream;

namespace formatters {
namespace NSDate {

/// Format the date_value field of a NSDate.
bool FormatDateValue(double date_value, Stream &stream);

} // namespace NSDate
} // namespace formatters
} // namespace lldb_private

#endif // LLDB_PLUGINS_LANGUAGE_OBJC_UTILITIES_H
