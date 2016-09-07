//===-- SymbolVendorMacOSX.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolVendorMacOSX_h_
#define liblldb_SymbolVendorMacOSX_h_

#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/lldb-private.h"

class SymbolVendorMacOSX : public lldb_private::SymbolVendor {
public:
  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static lldb_private::SymbolVendor *
  CreateInstance(const lldb::ModuleSP &module_sp,
                 lldb_private::Stream *feedback_strm);

  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  SymbolVendorMacOSX(const lldb::ModuleSP &module_sp);

  virtual ~SymbolVendorMacOSX();

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  virtual lldb_private::ConstString GetPluginName();

  virtual uint32_t GetPluginVersion();

private:
  DISALLOW_COPY_AND_ASSIGN(SymbolVendorMacOSX);
};

#endif // liblldb_SymbolVendorMacOSX_h_
