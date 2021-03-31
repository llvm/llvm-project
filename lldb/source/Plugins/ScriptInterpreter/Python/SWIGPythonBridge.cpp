//===-- SWIGPythonBridge.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// LLDB Python header must be included first
#include "lldb-python.h"

#include "SWIGPythonBridge.h"

using namespace lldb;

namespace lldb_private {

template <typename T> const char *GetPythonValueFormatString(T t);
template <> const char *GetPythonValueFormatString(char *) { return "s"; }
template <> const char *GetPythonValueFormatString(char) { return "b"; }
template <> const char *GetPythonValueFormatString(unsigned char) {
  return "B";
}
template <> const char *GetPythonValueFormatString(short) { return "h"; }
template <> const char *GetPythonValueFormatString(unsigned short) {
  return "H";
}
template <> const char *GetPythonValueFormatString(int) { return "i"; }
template <> const char *GetPythonValueFormatString(unsigned int) { return "I"; }
template <> const char *GetPythonValueFormatString(long) { return "l"; }
template <> const char *GetPythonValueFormatString(unsigned long) {
  return "k";
}
template <> const char *GetPythonValueFormatString(long long) { return "L"; }
template <> const char *GetPythonValueFormatString(unsigned long long) {
  return "K";
}
template <> const char *GetPythonValueFormatString(float) { return "f"; }
template <> const char *GetPythonValueFormatString(double) { return "d"; }

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
