//===-- DWARFASTParserGo.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFASTParserGo_h_
#define SymbolFileDWARF_DWARFASTParserGo_h_

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
#include "lldb/Symbol/GoASTContext.h"

class DWARFDebugInfoEntry;
class DWARFDIECollection;

class DWARFASTParserGo : public DWARFASTParser {
public:
  DWARFASTParserGo(lldb_private::GoASTContext &ast);

  ~DWARFASTParserGo() override;

  lldb::TypeSP ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                                  const DWARFDIE &die, lldb_private::Log *log,
                                  bool *type_is_new_ptr) override;

  lldb_private::Function *
  ParseFunctionFromDWARF(const lldb_private::SymbolContext &sc,
                         const DWARFDIE &die) override;

  bool CompleteTypeFromDWARF(const DWARFDIE &die, lldb_private::Type *type,
                             lldb_private::CompilerType &go_type) override;

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

private:
  size_t ParseChildParameters(
      const lldb_private::SymbolContext &sc, const DWARFDIE &parent_die,
      bool &is_variadic,
      std::vector<lldb_private::CompilerType> &function_param_types);
  void ParseChildArrayInfo(const lldb_private::SymbolContext &sc,
                           const DWARFDIE &parent_die, int64_t &first_index,
                           std::vector<uint64_t> &element_orders,
                           uint32_t &byte_stride, uint32_t &bit_stride);

  size_t ParseChildMembers(const lldb_private::SymbolContext &sc,
                           const DWARFDIE &die,
                           lldb_private::CompilerType &class_compiler_type);

  lldb_private::GoASTContext &m_ast;
};

#endif // SymbolFileDWARF_DWARFASTParserGo_h_
