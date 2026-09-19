//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeSystemFortran.h"
#include "FortranTypes.h"

#include "Plugins/SymbolFile/DWARF/DWARFASTParserFortran.h"

#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace lldb_private::plugin::dwarf;
using namespace lldb_private::plugin::fortran;

LLDB_PLUGIN_DEFINE(TypeSystemFortran)

char TypeSystemFortran::ID;

TypeSystemFortran::~TypeSystemFortran() = default;
TypeSystemFortran::TypeSystemFortran() = default;

void TypeSystemFortran::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "fortran plug-in",
                                CreateInstance, GetSupportedLanguagesForTypes(),
                                GetSupportedLanguagesForExpressions());
}

void TypeSystemFortran::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

plugin::dwarf::DWARFASTParser *TypeSystemFortran::GetDWARFParser() {
  if (!m_dwarf_ast_parser_up)
    m_dwarf_ast_parser_up = std::make_unique<DWARFASTParserFortran>(*this);
  return m_dwarf_ast_parser_up.get();
}

TypeSystemSP TypeSystemFortran::CreateInstance(LanguageType language,
                                               Module *module, Target *target) {
  if (Language::LanguageIsFortran(language))
    return std::make_shared<TypeSystemFortran>();

  return TypeSystemSP();
}

LanguageSet TypeSystemFortran::GetSupportedLanguagesForTypes() {
  LanguageSet languages;
  languages.Insert(eLanguageTypeFortran77);
  languages.Insert(eLanguageTypeFortran90);
  languages.Insert(eLanguageTypeFortran95);
  languages.Insert(eLanguageTypeFortran03);
  languages.Insert(eLanguageTypeFortran08);
  languages.Insert(eLanguageTypeFortran18);
  return languages;
}

LanguageSet TypeSystemFortran::GetSupportedLanguagesForExpressions() {
  return GetSupportedLanguagesForTypes();
}

#ifndef NDEBUG
bool TypeSystemFortran::Verify(lldb::opaque_compiler_type_t type) {
  return !type || llvm::isa<FortranType>(static_cast<FortranType *>(type));
}
#endif

bool TypeSystemFortran::IsFloatingPointType(opaque_compiler_type_t type) {
  if (!type)
    return false;

  if (static_cast<FortranType *>(type)->GetKind() == FortranType::KIND_REAL)
    return true;

  return false;
}

bool TypeSystemFortran::IsIntegerType(opaque_compiler_type_t type,
                                      bool &is_signed) {
  if (!type)
    return false;

  if (static_cast<FortranType *>(type)->GetKind() ==
      FortranType::KIND_INTEGER) {
    is_signed = true;
    return true;
  }
  return false;
}

bool TypeSystemFortran::SupportsLanguage(lldb::LanguageType language) {
  return Language::LanguageIsFortran(language);
}

/// Returns the type name upper-cased to follow Fortran's general style.
ConstString TypeSystemFortran::GetTypeName(opaque_compiler_type_t type,
                                           bool BaseOnly) {
  if (!type)
    return ConstString();

  FortranType *fortran_type = static_cast<FortranType *>(type);
  switch (fortran_type->GetKind()) {
  case FortranType::KIND_INTEGER:
  case FortranType::KIND_LOGICAL:
  case FortranType::KIND_REAL:
  case FortranType::KIND_COMPLEX:
    return fortran_type->GetName();
  case FortranType::KIND_UNKNOWN:
    return ConstString("Unsupported");
  }
  return ConstString("Unsupported");
}

CompilerType
TypeSystemFortran::CreateBaseType(llvm::dwarf::TypeKind dwarf_encoding,
                                  uint64_t bitsize, ConstString name) {
  int underlying_kind = FortranType::KIND_UNKNOWN;
  const char *default_name = nullptr;

  switch (dwarf_encoding) {
  case dwarf::DW_ATE_boolean:
    underlying_kind = FortranType::KIND_LOGICAL;
    default_name = "LOGICAL";
    break;
  case dwarf::DW_ATE_float:
    underlying_kind = FortranType::KIND_REAL;
    default_name = "REAL";
    break;
  case dwarf::DW_ATE_signed:
  case dwarf::DW_ATE_signed_char:
    underlying_kind = FortranType::KIND_INTEGER;
    default_name = "INTEGER";
    break;
  case dwarf::DW_ATE_complex_float:
    underlying_kind = FortranType::KIND_COMPLEX;
    default_name = "COMPLEX";
    break;
  default:
    return CompilerType();
  }
  /// DWARFASTParserFortran provides the names it gets from DWARF, but in case
  /// it doesn't we fallback to the default name.
  if (name.IsEmpty() && default_name)
    name.SetCString(default_name);

  return GetOrCreateFortranBaseType(underlying_kind, bitsize, name);
}

/// Creates the type if it has not been created before, otherwise it gets it
/// from the type map.
CompilerType TypeSystemFortran::GetOrCreateFortranBaseType(int kind,
                                                           uint64_t bitsize,
                                                           ConstString name) {
  auto new_type_up = std::make_unique<FortranType>(kind, bitsize, name);
  FortranType *fortran_type = m_basic_types.getOrInsert(new_type_up.get());
  if (fortran_type == new_type_up.get())
    m_types.push_back(std::move(new_type_up));
  return CompilerType(weak_from_this(), (void *)fortran_type);
}

lldb::TypeClass
TypeSystemFortran::GetTypeClass(lldb::opaque_compiler_type_t type) {
  return type ? lldb::eTypeClassBuiltin : lldb::eTypeClassInvalid;
}

CompilerType
TypeSystemFortran::GetCanonicalType(lldb::opaque_compiler_type_t type) {
  if (!type)
    return CompilerType();

  return CompilerType(weak_from_this(), type);
}

Expected<uint64_t>
TypeSystemFortran::GetBitSize(opaque_compiler_type_t type,
                              ExecutionContextScope *exe_scope) {
  if (!type)
    return 0;

  return static_cast<FortranType *>(type)->GetBitSize();
}

BasicType
TypeSystemFortran::GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) {
  if (!type)
    return eBasicTypeInvalid;

  FortranType *fortran_type = static_cast<FortranType *>(type);
  switch (fortran_type->GetKind()) {
  case FortranType::KIND_INTEGER:
    switch (fortran_type->GetBitSize()) {
    case 8:
      return eBasicTypeSignedChar;
    case 16:
      return eBasicTypeShort;
    case 32:
      return eBasicTypeInt;
    case 64:
      return eBasicTypeLongLong;
    case 128:
      return eBasicTypeInt128;
    default:
      return eBasicTypeInvalid;
    }
  case FortranType::KIND_LOGICAL:
    return eBasicTypeBool;
  case FortranType::KIND_COMPLEX:
    switch (fortran_type->GetBitSize()) {
    case 64:
      return eBasicTypeFloatComplex;
    case 128:
      return eBasicTypeDoubleComplex;
    case 256:
      return eBasicTypeLongDoubleComplex;
    default:
      return eBasicTypeInvalid;
    }
  case FortranType::KIND_REAL:
    switch (fortran_type->GetBitSize()) {
    case 16:
      return eBasicTypeHalf;
    case 32:
      return eBasicTypeFloat;
    case 64:
      return eBasicTypeDouble;
    case 128:
      return eBasicTypeFloat128;
    default:
      return eBasicTypeInvalid;
    }
  default:
    return eBasicTypeInvalid;
  }
}

CompilerType TypeSystemFortran::GetBasicTypeFromAST(BasicType basic_type) {
  switch (basic_type) {
  case eBasicTypeInt:
    return GetOrCreateFortranBaseType(FortranType::KIND_INTEGER, 32,
                                      ConstString("INTEGER"));
  case eBasicTypeFloat:
    return GetOrCreateFortranBaseType(FortranType::KIND_REAL, 32,
                                      ConstString("REAL"));
  case eBasicTypeDouble:
    return GetOrCreateFortranBaseType(FortranType::KIND_REAL, 64,
                                      ConstString("REAL(KIND=8)"));
  case eBasicTypeBool:
    return GetOrCreateFortranBaseType(FortranType::KIND_LOGICAL, 32,
                                      ConstString("LOGICAL"));
  case eBasicTypeFloatComplex:
    return GetOrCreateFortranBaseType(FortranType::KIND_COMPLEX, 64,
                                      ConstString("COMPLEX"));
  case eBasicTypeDoubleComplex:
    return GetOrCreateFortranBaseType(FortranType::KIND_COMPLEX, 128,
                                      ConstString("COMPLEX(KIND=8)"));
  case eBasicTypeLongDoubleComplex:
    return GetOrCreateFortranBaseType(FortranType::KIND_COMPLEX, 256,
                                      ConstString("COMPLEX(KIND=16)"));
  default:
    return CompilerType();
  }
}

CompilerType
TypeSystemFortran::GetBuiltinTypeForEncodingAndBitSize(Encoding encoding,
                                                       size_t bit_size) {
  switch (encoding) {
  case eEncodingSint:
    return GetOrCreateFortranBaseType(FortranType::KIND_INTEGER, bit_size,
                                      ConstString("INTEGER"));
  case eEncodingIEEE754:
    return GetOrCreateFortranBaseType(FortranType::KIND_REAL, bit_size,
                                      ConstString("REAL"));
  default:
    return CompilerType();
  }
}
