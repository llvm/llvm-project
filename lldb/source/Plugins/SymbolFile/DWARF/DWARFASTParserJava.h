//===-- DWARFASTParserJava.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFASTParserJava_h_
#define SymbolFileDWARF_DWARFASTParserJava_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

// Project includes
#include "DWARFASTParser.h"
#include "DWARFDIE.h"
#include "DWARFDefines.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Symbol/JavaASTContext.h"

class DWARFDebugInfoEntry;
class DWARFDIECollection;

class DWARFASTParserJava : public DWARFASTParser {
public:
  DWARFASTParserJava(lldb_private::JavaASTContext &ast);
  ~DWARFASTParserJava() override;

  lldb::TypeSP ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                                  const DWARFDIE &die, lldb_private::Log *log,
                                  bool *type_is_new_ptr) override;

  lldb_private::Function *
  ParseFunctionFromDWARF(const lldb_private::SymbolContext &sc,
                         const DWARFDIE &die) override;

  bool CompleteTypeFromDWARF(const DWARFDIE &die, lldb_private::Type *type,
                             lldb_private::CompilerType &java_type) override;

  lldb_private::CompilerDeclContext
  GetDeclContextForUIDFromDWARF(const DWARFDIE &die) override {
    return lldb_private::CompilerDeclContext();
  }

  lldb_private::CompilerDeclContext
  GetDeclContextContainingUIDFromDWARF(const DWARFDIE &die) override {
    return lldb_private::CompilerDeclContext();
  }

  lldb_private::CompilerDecl
  GetDeclForUIDFromDWARF(const DWARFDIE &die) override {
    return lldb_private::CompilerDecl();
  }

  std::vector<DWARFDIE> GetDIEForDeclContext(
      lldb_private::CompilerDeclContext decl_context) override {
    return std::vector<DWARFDIE>();
  }

  void ParseChildMembers(const DWARFDIE &parent_die,
                         lldb_private::CompilerType &class_compiler_type);

private:
  lldb_private::JavaASTContext &m_ast;

  lldb::TypeSP ParseBaseTypeFromDIE(const DWARFDIE &die);

  lldb::TypeSP ParseArrayTypeFromDIE(const DWARFDIE &die);

  lldb::TypeSP ParseReferenceTypeFromDIE(const DWARFDIE &die);

  lldb::TypeSP ParseClassTypeFromDIE(const DWARFDIE &die, bool &is_new_type);
};

#endif // SymbolFileDWARF_DWARFASTParserJava_h_
