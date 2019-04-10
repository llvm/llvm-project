//===-- SymbolVendorELF.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolVendorELF_h_
#define liblldb_SymbolVendorELF_h_

#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/lldb-private.h"

class SymbolVendorELF : public lldb_private::SymbolVendor {
public:
  // Constructors and Destructors
  SymbolVendorELF(const lldb::ModuleSP &module_sp);

  ~SymbolVendorELF() override;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static lldb_private::SymbolVendor *
  CreateInstance(const lldb::ModuleSP &module_sp,
                 lldb_private::Stream *feedback_strm);

  // PluginInterface protocol
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

private:
  DISALLOW_COPY_AND_ASSIGN(SymbolVendorELF);
};

#endif // liblldb_SymbolVendorELF_h_
