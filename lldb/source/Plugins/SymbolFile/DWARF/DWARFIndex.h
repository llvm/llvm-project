//===-- DWARFIndex.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DWARFINDEX_H
#define LLDB_DWARFINDEX_H

#include "Plugins/SymbolFile/DWARF/DIERef.h"
#include "Plugins/SymbolFile/DWARF/DWARFFormValue.h"

class DWARFDebugInfo;
class DWARFDeclContext;
class DWARFDIE;

namespace lldb_private {
class DWARFIndex {
public:
  DWARFIndex(Module &module) : m_module(module) {}
  virtual ~DWARFIndex();

  virtual void Preload() = 0;

  virtual void GetGlobalVariables(ConstString name, DIEArray &offsets) = 0;
  virtual void GetGlobalVariables(const RegularExpression &regex,
                                  DIEArray &offsets) = 0;
  virtual void GetGlobalVariables(const DWARFUnit &cu, DIEArray &offsets) = 0;
  virtual void GetObjCMethods(ConstString class_name, DIEArray &offsets) = 0;
  virtual void GetCompleteObjCClass(ConstString class_name,
                                    bool must_be_implementation,
                                    DIEArray &offsets) = 0;
  virtual void GetTypes(ConstString name, DIEArray &offsets) = 0;
  virtual void GetTypes(const DWARFDeclContext &context, DIEArray &offsets) = 0;
  virtual void GetNamespaces(ConstString name, DIEArray &offsets) = 0;
  virtual void GetFunctions(
      ConstString name, DWARFDebugInfo &info,
      llvm::function_ref<bool(const DWARFDIE &die, bool include_inlines,
                              lldb_private::SymbolContextList &sc_list)>
          resolve_function,
      llvm::function_ref<CompilerDeclContext(lldb::user_id_t type_uid)>
          get_decl_context_containing_uid,
      const CompilerDeclContext *parent_decl_ctx, uint32_t name_type_mask,
      bool include_inlines, SymbolContextList &sc_list) = 0;
  virtual void GetFunctions(
      const RegularExpression &regex, DWARFDebugInfo &info,
      llvm::function_ref<bool(const DWARFDIE &die, bool include_inlines,
                              lldb_private::SymbolContextList &sc_list)>
          resolve_function,
      bool include_inlines, SymbolContextList &sc_list) = 0;

  virtual void ReportInvalidDIEOffset(dw_offset_t offset,
                                      llvm::StringRef name) = 0;
  virtual void Dump(Stream &s) = 0;

protected:
  Module &m_module;

  void ParseFunctions(
      const DIEArray &offsets, DWARFDebugInfo &info,
      llvm::function_ref<bool(const DWARFDIE &die, bool include_inlines,
                              lldb_private::SymbolContextList &sc_list)>
          resolve_function,
      bool include_inlines, SymbolContextList &sc_list);
};
} // namespace lldb_private

#endif // LLDB_DWARFINDEX_H
