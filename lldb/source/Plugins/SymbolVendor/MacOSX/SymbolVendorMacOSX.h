//===-- SymbolVendorMacOSX.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolVendorMacOSX_h_
#define liblldb_SymbolVendorMacOSX_h_

#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/lldb-private.h"

class SymbolVendorMacOSX : public lldb_private::SymbolVendor {
public:
  // Static Functions
  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static lldb_private::SymbolVendor *
  CreateInstance(const lldb::ModuleSP &module_sp,
                 lldb_private::Stream *feedback_strm);

  // Constructors and Destructors
  SymbolVendorMacOSX(const lldb::ModuleSP &module_sp);

  virtual ~SymbolVendorMacOSX();

  // PluginInterface protocol
  virtual lldb_private::ConstString GetPluginName();

  virtual uint32_t GetPluginVersion();

private:
  DISALLOW_COPY_AND_ASSIGN(SymbolVendorMacOSX);
};

#endif // liblldb_SymbolVendorMacOSX_h_
