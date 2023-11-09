//===-- DWARFASTParserSwift.h -----------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFASTParserSwift_h_
#define SymbolFileDWARF_DWARFASTParserSwift_h_

#include "DWARFASTParser.h"
#include "DWARFDIE.h"
#include "swift/RemoteInspection/DescriptorFinder.h"
#include "swift/RemoteInspection/TypeRef.h"

namespace swift {
namespace reflection {
class TypeInfo;
} // namespace reflection
namespace remote {
struct TypeInfoProvider;
} // namespace remote
} // namespace swift

namespace lldb_private { class TypeSystemSwiftTypeRef; }

class DWARFASTParserSwift : public lldb_private::plugin::dwarf::DWARFASTParser,
                            public swift::reflection::DescriptorFinder {
public:
  using DWARFDIE = lldb_private::plugin::dwarf::DWARFDIE;
  DWARFASTParserSwift(lldb_private::TypeSystemSwiftTypeRef &swift_typesystem);

  virtual ~DWARFASTParserSwift();

  lldb::TypeSP ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                                  const DWARFDIE &die,
                                  bool *type_is_new_ptr) override;

  lldb_private::ConstString
  ConstructDemangledNameFromDWARF(const DWARFDIE &die) override;

  lldb_private::Function *
  ParseFunctionFromDWARF(lldb_private::CompileUnit &comp_unit,
                         const DWARFDIE &die,
                         const lldb_private::AddressRange &func_range) override;

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

  void EnsureAllDIEsInDeclContextHaveBeenParsed(
      lldb_private::CompilerDeclContext decl_context) override {}

  // FIXME: What should this do?
  lldb_private::ConstString
  GetDIEClassTemplateParams(const DWARFDIE &die) override {
    assert(false && "DWARFASTParserSwift::GetDIEClassTemplateParams has not "
                    "yet been implemented");
    return lldb_private::ConstString();
  }

  static bool classof(const DWARFASTParser *Parser) {
    return Parser->GetKind() == Kind::DWARFASTParserSwift;
  }

  /// Returns a field descriptor constructed from DWARF info.
  std::unique_ptr<swift::reflection::FieldDescriptorBase>
  getFieldDescriptor(const swift::reflection::TypeRef *TR) override;

  /// Returns a builtin descriptor constructed from DWARF info.
  std::unique_ptr<swift::reflection::BuiltinTypeDescriptorBase>
  getBuiltinTypeDescriptor(const swift::reflection::TypeRef *TR) override;

protected:
  lldb_private::TypeSystemSwiftTypeRef &m_swift_typesystem;
};

#endif // SymbolFileDWARF_DWARFASTParserSwift_h_
