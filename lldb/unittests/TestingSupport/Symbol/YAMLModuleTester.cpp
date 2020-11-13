//===-- YAMLModuleTester.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Section.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"

using namespace lldb_private;

YAMLModuleTester::YAMLModuleTester(llvm::StringRef yaml_data) {
  llvm::Expected<TestFile> File = TestFile::fromYaml(yaml_data);
  EXPECT_THAT_EXPECTED(File, llvm::Succeeded());
  m_file = std::move(*File);

  m_module_sp = std::make_shared<Module>(m_file->moduleSpec());
  auto &symfile = *llvm::cast<SymbolFileDWARF>(m_module_sp->GetSymbolFile());

  m_dwarf_unit = symfile.DebugInfo().GetUnitAtIndex(0);
}
