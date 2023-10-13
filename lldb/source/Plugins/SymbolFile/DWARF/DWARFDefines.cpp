//===-- DWARFDefines.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDefines.h"
#include "lldb/Utility/ConstString.h"
#include <cstdio>
#include <cstring>
#include <string>

namespace lldb_private::plugin {
namespace dwarf {

const char *DW_TAG_value_to_name(uint32_t val) {
  static char invalid[100];

  if (val == 0)
    return "NULL";

  llvm::StringRef llvmstr = llvm::dwarf::TagString(val);
  if (llvmstr.empty()) {
    snprintf(invalid, sizeof(invalid), "Unknown DW_TAG constant: 0x%x", val);
    return invalid;
  }
  return llvmstr.data();
}

const char *DW_AT_value_to_name(uint32_t val) {
  static char invalid[100];
  llvm::StringRef llvmstr = llvm::dwarf::AttributeString(val);
  if (llvmstr.empty()) {
    snprintf(invalid, sizeof(invalid), "Unknown DW_AT constant: 0x%x", val);
    return invalid;
  }
  return llvmstr.data();
}

const char *DW_FORM_value_to_name(uint32_t val) {
  static char invalid[100];
  llvm::StringRef llvmstr = llvm::dwarf::FormEncodingString(val);
  if (llvmstr.empty()) {
    snprintf(invalid, sizeof(invalid), "Unknown DW_FORM constant: 0x%x", val);
    return invalid;
  }
  return llvmstr.data();
}

const char *DW_OP_value_to_name(uint32_t val) {
  static char invalid[100];
  llvm::StringRef llvmstr = llvm::dwarf::OperationEncodingString(val);
  if (llvmstr.empty()) {
    snprintf(invalid, sizeof(invalid), "Unknown DW_OP constant: 0x%x", val);
    return invalid;
  }
  return llvmstr.data();
}

const char *DW_ATE_value_to_name(uint32_t val) {
  static char invalid[100];
  llvm::StringRef llvmstr = llvm::dwarf::AttributeEncodingString(val);
  if (llvmstr.empty()) {
    snprintf(invalid, sizeof(invalid), "Unknown DW_ATE constant: 0x%x", val);
    return invalid;
  }
  return llvmstr.data();
}

const char *DW_LANG_value_to_name(uint32_t val) {
  static char invalid[100];
  llvm::StringRef llvmstr = llvm::dwarf::LanguageString(val);
  if (llvmstr.empty()) {
    snprintf(invalid, sizeof(invalid), "Unknown DW_LANG constant: 0x%x", val);
    return invalid;
  }
  return llvmstr.data();
}

const char *DW_LNS_value_to_name(uint32_t val) {
  static char invalid[100];
  llvm::StringRef llvmstr = llvm::dwarf::LNStandardString(val);
  if (llvmstr.empty()) {
    snprintf(invalid, sizeof(invalid), "Unknown DW_LNS constant: 0x%x", val);
    return invalid;
  }
  return llvmstr.data();
}

} // namespace dwarf
} // namespace lldb_private::plugin
