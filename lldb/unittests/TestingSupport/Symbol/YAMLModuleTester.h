//===- YAMLModuleTester.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_TESTINGSUPPORT_SYMBOL_YAMLMODULETESTER_H
#define LLDB_UNITTESTS_TESTINGSUPPORT_SYMBOL_YAMLMODULETESTER_H

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/DWARF/DWARFUnit.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/HostInfo.h"

namespace lldb_private {

/// Helper class that can construct a module from YAML and evaluate
/// DWARF expressions on it.
class YAMLModuleTester {
protected:
  SubsystemRAII<FileSystem, HostInfo, TypeSystemClang, ObjectFileELF,
                SymbolFileDWARF>
      subsystems;
  llvm::Optional<TestFile> m_file;
  lldb::ModuleSP m_module_sp;
  DWARFUnit *m_dwarf_unit;

public:
  /// Parse the debug info sections from the YAML description.
  YAMLModuleTester(llvm::StringRef yaml_data);
  DWARFUnit *GetDwarfUnit() const { return m_dwarf_unit; }
  lldb::ModuleSP GetModule() const { return m_module_sp; }
};

} // namespace lldb_private

#endif // LLDB_UNITTESTS_TESTINGSUPPORT_SYMBOL_YAMLMODULETESTER_H
