//===-- ManulaDWARFIndex.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_MANUALDWARFINDEX_H
#define LLDB_MANUALDWARFINDEX_H

#include "Plugins/SymbolFile/DWARF/DWARFIndex.h"
#include "Plugins/SymbolFile/DWARF/NameToDIE.h"

namespace lldb_private {
class ManualDWARFIndex : public DWARFIndex {
public:
  ManualDWARFIndex(Module &module, DWARFDebugInfo *debug_info)
      : DWARFIndex(module), m_debug_info(debug_info) {}

  void Preload() override { Index(); }

  void GetGlobalVariables(ConstString name, DIEArray &offsets) override;
  void GetGlobalVariables(const RegularExpression &regex,
                          DIEArray &offsets) override;
  void GetGlobalVariables(const DWARFUnit &cu, DIEArray &offsets) override;
  void GetObjCMethods(ConstString class_name, DIEArray &offsets) override;
  void GetCompleteObjCClass(ConstString class_name, bool must_be_implementation,
                            DIEArray &offsets) override;
  void GetTypes(ConstString name, DIEArray &offsets) override;
  void GetTypes(const DWARFDeclContext &context, DIEArray &offsets) override;
  void GetNamespaces(ConstString name, DIEArray &offsets) override;
  void GetFunctions(
      ConstString name, DWARFDebugInfo &info,
      llvm::function_ref<bool(const DWARFDIE &die, bool include_inlines,
                              lldb_private::SymbolContextList &sc_list)>
          resolve_function,
      llvm::function_ref<CompilerDeclContext(lldb::user_id_t type_uid)>
          get_decl_context_containing_uid,
      const CompilerDeclContext *parent_decl_ctx, uint32_t name_type_mask,
      bool include_inlines, SymbolContextList &sc_list) override;
  void GetFunctions(
      const RegularExpression &regex, DWARFDebugInfo &info,
      llvm::function_ref<bool(const DWARFDIE &die, bool include_inlines,
                              lldb_private::SymbolContextList &sc_list)>
          resolve_function,
      bool include_inlines, SymbolContextList &sc_list) override;

  void ReportInvalidDIEOffset(dw_offset_t offset,
                              llvm::StringRef name) override {}
  void Dump(Stream &s) override;

private:
  void Index();

  /// Non-null value means we haven't built the index yet.
  DWARFDebugInfo *m_debug_info;

  NameToDIE m_function_basenames;
  NameToDIE m_function_fullnames;
  NameToDIE m_function_methods;
  NameToDIE m_function_selectors;
  NameToDIE m_objc_class_selectors;
  NameToDIE m_globals;
  NameToDIE m_types;
  NameToDIE m_namespaces;
};
} // namespace lldb_private

#endif // LLDB_MANUALDWARFINDEX_H
