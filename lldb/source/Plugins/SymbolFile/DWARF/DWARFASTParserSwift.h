//===-- DWARFASTParserSwift.h -----------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFASTParserSwift_h_
#define SymbolFileDWARF_DWARFASTParserSwift_h_

#include "DWARFASTParser.h"
#include "DWARFDIE.h"

class DWARFDebugInfoEntry;
class DWARFDIECollection;

class DWARFASTParserSwift : public DWARFASTParser {
public:
  DWARFASTParserSwift(lldb_private::SwiftASTContext &ast);

  virtual ~DWARFASTParserSwift();

  lldb::TypeSP ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                                  const DWARFDIE &die, lldb_private::Log *log,
                                  bool *type_is_new_ptr) override;

  lldb_private::Function *
  ParseFunctionFromDWARF(const lldb_private::SymbolContext &sc,
                         const DWARFDIE &die) override;

  bool
  CompleteTypeFromDWARF(const DWARFDIE &die, lldb_private::Type *type,
                        lldb_private::CompilerType &compiler_type) override {
    return false;
  }

  lldb_private::CompilerDecl
  GetDeclForUIDFromDWARF(const DWARFDIE &die) override {
    return lldb_private::CompilerDecl();
  }

  lldb_private::CompilerDeclContext
  GetDeclContextForUIDFromDWARF(const DWARFDIE &die) override;

  lldb_private::CompilerDeclContext
  GetDeclContextContainingUIDFromDWARF(const DWARFDIE &die) override;

  std::vector<DWARFDIE> GetDIEForDeclContext(
      lldb_private::CompilerDeclContext decl_context) override {
    return {};
  }

protected:
  lldb_private::SwiftASTContext &m_ast;
};

#endif // SymbolFileDWARF_DWARFASTParserSwift_h_
