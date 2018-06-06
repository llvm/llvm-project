//===-- DebugNamesDWARFIndex.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DEBUGNAMESDWARFINDEX_H
#define LLDB_DEBUGNAMESDWARFINDEX_H

#include "Plugins/SymbolFile/DWARF/DWARFIndex.h"
#include "lldb/Utility/ConstString.h"
#include "llvm/DebugInfo/DWARF/DWARFAcceleratorTable.h"

namespace lldb_private {
class DebugNamesDWARFIndex : public DWARFIndex {
public:
  static llvm::Expected<std::unique_ptr<DebugNamesDWARFIndex>>
  Create(Module &module, DWARFDataExtractor debug_names,
         DWARFDataExtractor debug_str, DWARFDebugInfo *debug_info);

  void Preload() override {}

  void GetGlobalVariables(ConstString name, DIEArray &offsets) override {}
  void GetGlobalVariables(const RegularExpression &regex,
                          DIEArray &offsets) override {}
  void GetGlobalVariables(const DWARFUnit &cu, DIEArray &offsets) override {}
  void GetObjCMethods(ConstString class_name, DIEArray &offsets) override {}
  void GetCompleteObjCClass(ConstString class_name, bool must_be_implementation,
                            DIEArray &offsets) override {}
  void GetTypes(ConstString name, DIEArray &offsets) override {}
  void GetTypes(const DWARFDeclContext &context, DIEArray &offsets) override {}
  void GetNamespaces(ConstString name, DIEArray &offsets) override {}
  void GetFunctions(ConstString name, DWARFDebugInfo &info,
                    const CompilerDeclContext &parent_decl_ctx,
                    uint32_t name_type_mask,
                    std::vector<DWARFDIE> &dies) override {}
  void GetFunctions(const RegularExpression &regex,
                    DIEArray &offsets) override {}

  void ReportInvalidDIEOffset(dw_offset_t offset,
                              llvm::StringRef name) override {}
  void Dump(Stream &s) override {}

private:
  DebugNamesDWARFIndex(Module &module,
                       std::unique_ptr<llvm::DWARFDebugNames> debug_names_up,
                       DWARFDataExtractor debug_names_data,
                       DWARFDataExtractor debug_str_data,
                       DWARFDebugInfo *debug_info)
      : DWARFIndex(module), m_debug_names_up(std::move(debug_names_up)) {}

  // LLVM DWARFDebugNames will hold a non-owning reference to this data, so keep
  // track of the ownership here.
  DWARFDataExtractor m_debug_names_data;
  DWARFDataExtractor m_debug_str_data;

  std::unique_ptr<llvm::DWARFDebugNames> m_debug_names_up;
};

} // namespace lldb_private

#endif // LLDB_DEBUGNAMESDWARFINDEX_H
