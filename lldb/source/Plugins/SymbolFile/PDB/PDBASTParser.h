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
template <typename ChildType> class ConcreteSymbolEnumerator;

class PDBSymbol;
class PDBSymbolData;
class PDBSymbolFunc;
class PDBSymbolTypeBaseClass;
class PDBSymbolTypeBuiltin;
class PDBSymbolTypeUDT;
} // namespace pdb
} // namespace llvm

class PDBASTParser {
public:
  PDBASTParser(lldb_private::ClangASTContext &ast);
  ~PDBASTParser();

  lldb::TypeSP CreateLLDBTypeFromPDBType(const llvm::pdb::PDBSymbol &type);
  bool CompleteTypeFromPDB(lldb_private::CompilerType &compiler_type);

  lldb_private::ClangASTImporter &GetClangASTImporter() {
    return m_ast_importer;
  }

private:
  typedef llvm::DenseMap<lldb::opaque_compiler_type_t, lldb::user_id_t>
      ClangTypeToUidMap;
  typedef llvm::pdb::ConcreteSymbolEnumerator<llvm::pdb::PDBSymbolData>
      PDBDataSymbolEnumerator;
  typedef llvm::pdb::ConcreteSymbolEnumerator<llvm::pdb::PDBSymbolTypeBaseClass>
      PDBBaseClassSymbolEnumerator;
  typedef llvm::pdb::ConcreteSymbolEnumerator<llvm::pdb::PDBSymbolFunc>
      PDBFuncSymbolEnumerator;

  bool AddEnumValue(lldb_private::CompilerType enum_type,
                    const llvm::pdb::PDBSymbolData &data) const;
  bool CompleteTypeFromUDT(lldb_private::SymbolFile &symbol_file,
                           lldb_private::CompilerType &compiler_type,
                           llvm::pdb::PDBSymbolTypeUDT &udt);
  void AddRecordMembers(
      lldb_private::SymbolFile &symbol_file,
      lldb_private::CompilerType &record_type,
      PDBDataSymbolEnumerator &members_enum,
      lldb_private::ClangASTImporter::LayoutInfo &layout_info) const;
  void AddRecordBases(
      lldb_private::SymbolFile &symbol_file,
      lldb_private::CompilerType &record_type,
      int record_kind,
      PDBBaseClassSymbolEnumerator &bases_enum,
      lldb_private::ClangASTImporter::LayoutInfo &layout_info) const;
  void AddRecordMethods(
      lldb_private::SymbolFile &symbol_file,
      lldb_private::CompilerType &record_type,
      PDBFuncSymbolEnumerator &methods_enum) const;

  lldb_private::ClangASTContext &m_ast;
  lldb_private::ClangASTImporter m_ast_importer;
  ClangTypeToUidMap m_forward_decl_clang_type_to_uid;
};

#endif // LLDB_PLUGINS_SYMBOLFILE_PDB_PDBASTPARSER_H
