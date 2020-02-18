//===-- DWARFIndex.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFINDEX_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFINDEX_H

#include "Plugins/SymbolFile/DWARF/DIERef.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Plugins/SymbolFile/DWARF/DWARFFormValue.h"

class DWARFDeclContext;
class DWARFDIE;

namespace lldb_private {
class DWARFIndex {
public:
  DWARFIndex(Module &module) : m_module(module) {}
  virtual ~DWARFIndex();

  virtual void Preload() = 0;

  /// Finds global variables with the given base name. Any additional filtering
  /// (e.g., to only retrieve variables from a given context) should be done by
  /// the consumer.
  virtual void GetGlobalVariables(ConstString basename, DIEArray &offsets) = 0;

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
  virtual void GetFunctions(ConstString name, SymbolFileDWARF &dwarf,
                            const CompilerDeclContext &parent_decl_ctx,
                            uint32_t name_type_mask,
                            std::vector<DWARFDIE> &dies) = 0;
  virtual void GetFunctions(const RegularExpression &regex,
                            DIEArray &offsets) = 0;

  virtual void ReportInvalidDIERef(const DIERef &ref, llvm::StringRef name) = 0;
  virtual void Dump(Stream &s) = 0;

protected:
  Module &m_module;

  /// Helper function implementing common logic for processing function dies. If
  /// the function given by "ref" matches search criteria given by
  /// "parent_decl_ctx" and "name_type_mask", it is inserted into the "dies"
  /// vector.
  void ProcessFunctionDIE(llvm::StringRef name, DIERef ref,
                          SymbolFileDWARF &dwarf,
                          const CompilerDeclContext &parent_decl_ctx,
                          uint32_t name_type_mask, std::vector<DWARFDIE> &dies);
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFINDEX_H
