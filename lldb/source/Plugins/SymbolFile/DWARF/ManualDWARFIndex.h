//===-- ManualDWARFIndex.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_MANUALDWARFINDEX_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_MANUALDWARFINDEX_H

#include "Plugins/SymbolFile/DWARF/DWARFIndex.h"
#include "Plugins/SymbolFile/DWARF/NameToDIE.h"
#include "llvm/ADT/DenseSet.h"

class DWARFDebugInfo;
class SymbolFileDWARFDwo;

namespace lldb_private {
class ManualDWARFIndex : public DWARFIndex {
public:
  ManualDWARFIndex(Module &module, SymbolFileDWARF &dwarf,
                   llvm::DenseSet<dw_offset_t> units_to_avoid = {})
      : DWARFIndex(module), m_dwarf(&dwarf),
        m_units_to_avoid(std::move(units_to_avoid)) {}

  void Preload() override { Index(); }

  void
  GetGlobalVariables(ConstString basename,
                     llvm::function_ref<bool(DWARFDIE die)> callback) override;
  void
  GetGlobalVariables(const RegularExpression &regex,
                     llvm::function_ref<bool(DWARFDIE die)> callback) override;
  void
  GetGlobalVariables(const DWARFUnit &unit,
                     llvm::function_ref<bool(DWARFDIE die)> callback) override;
  void GetObjCMethods(ConstString class_name,
                      llvm::function_ref<bool(DWARFDIE die)> callback) override;
  void GetCompleteObjCClass(
      ConstString class_name, bool must_be_implementation,
      llvm::function_ref<bool(DWARFDIE die)> callback) override;
  void GetTypes(ConstString name,
                llvm::function_ref<bool(DWARFDIE die)> callback) override;
  void GetTypes(const DWARFDeclContext &context,
                llvm::function_ref<bool(DWARFDIE die)> callback) override;
  void GetNamespaces(ConstString name,
                     llvm::function_ref<bool(DWARFDIE die)> callback) override;
  void GetFunctions(ConstString name, SymbolFileDWARF &dwarf,
                    const CompilerDeclContext &parent_decl_ctx,
                    uint32_t name_type_mask,
                    llvm::function_ref<bool(DWARFDIE die)> callback) override;
  void GetFunctions(const RegularExpression &regex,
                    llvm::function_ref<bool(DWARFDIE die)> callback) override;

  void Dump(Stream &s) override;

private:
  struct IndexSet {
    NameToDIE function_basenames;
    NameToDIE function_fullnames;
    NameToDIE function_methods;
    NameToDIE function_selectors;
    NameToDIE objc_class_selectors;
    NameToDIE globals;
    NameToDIE types;
    NameToDIE namespaces;
  };
  void Index();
  void IndexUnit(DWARFUnit &unit, SymbolFileDWARFDwo *dwp, IndexSet &set);

  static void IndexUnitImpl(DWARFUnit &unit,
                            const lldb::LanguageType cu_language,
                            IndexSet &set);

  /// The DWARF file which we are indexing. Set to nullptr after the index is
  /// built.
  SymbolFileDWARF *m_dwarf;
  /// Which dwarf units should we skip while building the index.
  llvm::DenseSet<dw_offset_t> m_units_to_avoid;

  IndexSet m_set;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_MANUALDWARFINDEX_H
