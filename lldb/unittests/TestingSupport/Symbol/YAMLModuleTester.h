//===- YAMLModuleTester.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_TESTINGSUPPORT_SYMBOL_YAMLMODULETESTER_H
#define LLDB_UNITTESTS_TESTINGSUPPORT_SYMBOL_YAMLMODULETESTER_H

#include "Plugins/SymbolFile/DWARF/DWARFUnit.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/HostInfo.h"

namespace lldb_private {

/// Helper class that can construct a module from YAML and evaluate
/// DWARF expressions on it.
class YAMLModuleTester {
protected:
  SubsystemRAII<FileSystem, HostInfo, TypeSystemClang> subsystems;
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> m_sections_map;
  lldb::ModuleSP m_module_sp;
  lldb::ObjectFileSP m_objfile_sp;
  DWARFUnitSP m_dwarf_unit;
  std::unique_ptr<SymbolFileDWARF> m_symfile_dwarf;

public:
  /// Parse the debug info sections from the YAML description.
  YAMLModuleTester(llvm::StringRef yaml_data, llvm::StringRef triple);
  DWARFUnitSP GetDwarfUnit() { return m_dwarf_unit; }
};

} // namespace lldb_private

#endif // LLDB_UNITTESTS_TESTINGSUPPORT_SYMBOL_YAMLMODULETESTER_H
