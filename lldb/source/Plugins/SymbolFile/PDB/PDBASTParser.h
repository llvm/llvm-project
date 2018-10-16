//===-- PDBASTParser.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYMBOLFILE_PDB_PDBASTPARSER_H
#define LLDB_PLUGINS_SYMBOLFILE_PDB_PDBASTPARSER_H

#include "lldb/lldb-forward.h"

#include "lldb/Symbol/ClangASTImporter.h"

namespace clang {
class CharUnits;
class CXXRecordDecl;
class FieldDecl;
class RecordDecl;
} // namespace clang

namespace lldb_private {
class ClangASTContext;
class CompilerType;
} // namespace lldb_private

namespace llvm {
namespace pdb {
class PDBSymbol;
class PDBSymbolData;
class PDBSymbolTypeBuiltin;
} // namespace pdb
} // namespace llvm

class PDBASTParser {
public:
  PDBASTParser(lldb_private::ClangASTContext &ast);
  ~PDBASTParser();

  lldb::TypeSP CreateLLDBTypeFromPDBType(const llvm::pdb::PDBSymbol &type);

private:
  bool AddEnumValue(lldb_private::CompilerType enum_type,
                    const llvm::pdb::PDBSymbolData &data) const;

  lldb_private::ClangASTContext &m_ast;
  lldb_private::ClangASTImporter m_ast_importer;
};

#endif // LLDB_PLUGINS_SYMBOLFILE_PDB_PDBASTPARSER_H
