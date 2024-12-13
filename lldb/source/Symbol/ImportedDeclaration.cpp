//===-- ImportedDeclaration.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ImportedDeclaration.h"
#include "lldb/Symbol/SymbolFile.h"

using namespace lldb;
using namespace lldb_private;

std::vector<lldb_private::CompilerContext>
ImportedDeclaration::GetDeclContext() const {
  return m_symbol_file->GetCompilerContextForUID(GetID());
}
