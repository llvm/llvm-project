//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the Fortran type system.
///
//===----------------------------------------------------------------------===//
#include "TypeSystemFortran.h"
#include "FortranTypes.h"

#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"

#include "llvm/Support/raw_ostream.h"

#include "Plugins/SymbolFile/DWARF/DWARFASTParserFortran.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace lldb_private::plugin::fortran;
using namespace lldb_private::plugin::dwarf;

LLDB_PLUGIN_DEFINE(TypeSystemFortran)

/// Used to determine if TypeSystem supports the language passed in
/// CreateInstance
static bool IsLanguageSupported(LanguageType language) {
  if (language == LanguageType::eLanguageTypeFortran77 ||
      language == LanguageType::eLanguageTypeFortran90 ||
      language == LanguageType::eLanguageTypeFortran95 ||
      language == LanguageType::eLanguageTypeFortran03 ||
      language == LanguageType::eLanguageTypeFortran08 ||
      language == LanguageType::eLanguageTypeFortran18)
    return true;

  return false;
}

/// Unlike C/C++, Fortran formats COMPLEX numbers as a parenthesised tuple
/// (real, imaginary).
static bool DumpComplex(Stream &s, const lldb_private::DataExtractor &data,
                        lldb::offset_t &offset, size_t data_byte_size) {
  if (sizeof(float) * 2 == data_byte_size) {
    float f32_1 = data.GetFloat(&offset);
    float f32_2 = data.GetFloat(&offset);

    s.Printf("(%g, %g)", f32_1, f32_2);
    return true;
  } else if (sizeof(double) * 2 == data_byte_size) {
    double d64_1 = data.GetDouble(&offset);
    double d64_2 = data.GetDouble(&offset);

    s.Printf("(%lg, %lg)", d64_1, d64_2);
    return true;
  } else if (sizeof(long double) * 2 == data_byte_size) {
    long double ld64_1 = data.GetLongDouble(&offset);
    long double ld64_2 = data.GetLongDouble(&offset);
    s.Printf("(%Lg, %Lg)", ld64_1, ld64_2);
    return true;
  } else {
    s.Printf("error: unsupported byte size (%" PRIu64
             ") for complex float format",
             (uint64_t)data_byte_size);
    return false;
  }
}

char TypeSystemFortran::ID;

TypeSystemFortran::~TypeSystemFortran() = default;
TypeSystemFortran::TypeSystemFortran() = default;

void TypeSystemFortran::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "fortran AST context plug-in", CreateInstance,
      GetSupportedLanguagesForTypes(), GetSupportedLanguagesForExpressions());
}

void TypeSystemFortran::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

plugin::dwarf::DWARFASTParser *TypeSystemFortran::GetDWARFParser() {
  if (!m_dwarf_ast_parser_up)
    m_dwarf_ast_parser_up = std::make_unique<DWARFASTParserFortran>(*this);
  return m_dwarf_ast_parser_up.get();
}

// TODO: Process Target and architecture for pointers and Expression Evaluation,
// if module and target have different typesystems like clang, we would have to
// account for that here.
TypeSystemSP TypeSystemFortran::CreateInstance(LanguageType language,
                                               Module *module, Target *target) {

  if (IsLanguageSupported(language)) {
    auto type_system_sp = std::make_shared<TypeSystemFortran>();

    // Get the byte order from the target or module and store it
    if (target) {
      type_system_sp->SetByteOrder(target->GetArchitecture().GetByteOrder());
    } else if (module) {
      type_system_sp->SetByteOrder(module->GetArchitecture().GetByteOrder());
    }

    return type_system_sp;
  }
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

// FIXME: Currently returns all Fortran languages to satisfy plugin
// requirements, but expression evaluation is not yet implemented.
LanguageSet TypeSystemFortran::GetSupportedLanguagesForExpressions() {
  LanguageSet languages;
  languages.Insert(eLanguageTypeFortran77);
  languages.Insert(eLanguageTypeFortran90);
  languages.Insert(eLanguageTypeFortran95);
  languages.Insert(eLanguageTypeFortran03);
  languages.Insert(eLanguageTypeFortran08);
  languages.Insert(eLanguageTypeFortran18);
  return languages;
}

bool TypeSystemFortran::SupportsLanguage(lldb::LanguageType language) {
  if (language == lldb::LanguageType::eLanguageTypeFortran77 ||
      language == lldb::LanguageType::eLanguageTypeFortran90 ||
      language == lldb::LanguageType::eLanguageTypeFortran95 ||
      language == lldb::LanguageType::eLanguageTypeFortran03 ||
      language == lldb::LanguageType::eLanguageTypeFortran08 ||
      language == lldb::LanguageType::eLanguageTypeFortran18) {
    return true;
  }
  return false;
}

CompilerType TypeSystemFortran::GetOrCreateFortranType(int kind,
                                                       uint64_t bitsize,
                                                       ConstString name) {
  llvm::FoldingSetNodeID id;
  FortranType::Profile(id, kind, bitsize);
  void *insert_pos = nullptr;
  FortranType *fortran_type = m_basic_types.FindNodeOrInsertPos(id, insert_pos);
  if (fortran_type)
    return CompilerType(weak_from_this(), (void *)fortran_type);
  auto new_type_up = std::make_unique<FortranType>(kind, bitsize, name);
  fortran_type = new_type_up.get();

  m_types.push_back(std::move(new_type_up));
  m_basic_types.InsertNode(fortran_type, insert_pos);
  return CompilerType(weak_from_this(), (void *)fortran_type);
}

CompilerType TypeSystemFortran::GetOrCreateFortranFunction(
    ConstString name, const SmallVectorImpl<CompilerType> &parameters) {
  llvm::FoldingSetNodeID id;
  FortranFunction::Profile(id, name, parameters);
  void *insert_pos = nullptr;
  FortranFunction *fortran_function =
      m_functions.FindNodeOrInsertPos(id, insert_pos);
  if (fortran_function)
    return CompilerType(weak_from_this(), (void *)fortran_function);
  auto new_type_up = std::make_unique<FortranFunction>(name, parameters);
  fortran_function = new_type_up.get();

  m_functions.InsertNode(fortran_function, insert_pos);
  m_types.push_back(std::move(new_type_up));

  return CompilerType(weak_from_this(), (void *)fortran_function);
}

CompilerType TypeSystemFortran::CreateType(uint32_t kind, uint64_t bitsize,
                                           ConstString name) {
  int underlying_kind;
  switch (kind) {
  case dwarf::DW_ATE_boolean:
    if (bitsize == 32)
      name.SetCString("LOGICAL");
    underlying_kind = FortranType::KIND_LOGICAL;
    break;
  case dwarf::DW_ATE_float:
    if (bitsize == 32)
      name.SetCString("REAL");
    underlying_kind = FortranType::KIND_REAL;
    break;
  case dwarf::DW_ATE_signed:
    if (bitsize == 32)
      name.SetCString("INTEGER");
    underlying_kind = FortranType::KIND_INTEGER;
    break;
  case dwarf::DW_ATE_complex_float:
    if (bitsize == 64)
      name.SetCString("COMPLEX");
    underlying_kind = FortranType::KIND_COMPLEX;
    break;
  default:
    return CompilerType();
  }
  return GetOrCreateFortranType(underlying_kind, bitsize, name);
}

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
  default:
    return ConstString("Unsupported");
  }
}

CompilerType TypeSystemFortran::GetBasicTypeFromAST(BasicType basic_type) {
  switch (basic_type) {
  case eBasicTypeInt:
    return GetOrCreateFortranType(FortranType::KIND_INTEGER, 32,
                                  ConstString("INTEGER"));
  case eBasicTypeFloat:
    return GetOrCreateFortranType(FortranType::KIND_REAL, 32,
                                  ConstString("REAL"));
  case eBasicTypeDouble:
    return GetOrCreateFortranType(FortranType::KIND_REAL, 64,
                                  ConstString("REAL(KIND=8)"));
  case eBasicTypeBool:
    return GetOrCreateFortranType(FortranType::KIND_LOGICAL, 32,
                                  ConstString("LOGICAL"));
  case eBasicTypeFloatComplex:
    return GetOrCreateFortranType(FortranType::KIND_COMPLEX, 64,
                                  ConstString("COMPLEX"));
  case eBasicTypeDoubleComplex:
    return GetOrCreateFortranType(FortranType::KIND_COMPLEX, 128,
                                  ConstString("COMPLEX(KIND=8)"));
  case eBasicTypeLongDoubleComplex:
    return GetOrCreateFortranType(FortranType::KIND_COMPLEX, 256,
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
    return GetOrCreateFortranType(FortranType::KIND_INTEGER, bit_size,
                                  ConstString("INTEGER"));
  case eEncodingIEEE754:
    return GetOrCreateFortranType(FortranType::KIND_REAL, bit_size,
                                  ConstString("REAL"));
  default:
    return CompilerType();
  }
}

uint32_t
TypeSystemFortran::GetTypeInfo(opaque_compiler_type_t type,
                               CompilerType *pointee_or_element_compiler_type) {
  if (!type)
    return 0;
  FortranType *fortran_type = static_cast<FortranType *>(type);
  uint32_t builtin_type_flags = eTypeIsBuiltIn | eTypeHasValue;
  int type_kind = fortran_type->GetKind();
  switch (type_kind) {
  case FortranType::KIND_REAL:
  case FortranType::KIND_INTEGER:
  case FortranType::KIND_LOGICAL:
    builtin_type_flags |= eTypeIsScalar;
    if (type_kind == FortranType::KIND_INTEGER)
      builtin_type_flags |= eTypeIsInteger | eTypeIsSigned;
    if (type_kind == FortranType::KIND_REAL)
      builtin_type_flags |= eTypeIsFloat;
    break;
  case FortranType::KIND_COMPLEX:
    builtin_type_flags |= eTypeIsComplex;
    break;
  default:
    break;
  }
  return builtin_type_flags;
}

Expected<uint64_t>
TypeSystemFortran::GetBitSize(opaque_compiler_type_t type,
                              ExecutionContextScope *exe_scope) {
  if (!type)
    return 0;
  FortranType *fortran_type = static_cast<FortranType *>(type);
  return fortran_type->GetBitSize();
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

Encoding TypeSystemFortran::GetEncoding(opaque_compiler_type_t type) {
  if (!type)
    return eEncodingInvalid;
  FortranType *fortran_type = static_cast<FortranType *>(type);
  switch (fortran_type->GetKind()) {
  case FortranType::KIND_COMPLEX:
  case FortranType::KIND_REAL:
    return eEncodingIEEE754;
  case FortranType::KIND_INTEGER:
    return eEncodingSint;
  case FortranType::KIND_LOGICAL:
    return eEncodingUint;
  default:
    return eEncodingInvalid;
  }
}

Format TypeSystemFortran::GetFormat(opaque_compiler_type_t type) {
  if (!type)
    return eFormatDefault;
  FortranType *fortran_type = static_cast<FortranType *>(type);
  switch (fortran_type->GetKind()) {
  case FortranType::KIND_INTEGER:
    return eFormatDecimal;
  case FortranType::KIND_REAL:
    return eFormatFloat;
  case FortranType::KIND_LOGICAL:
    return eFormatBoolean;
  case FortranType::KIND_COMPLEX:
    return eFormatComplex;
  default:
    return eFormatDefault;
  }
}

bool TypeSystemFortran::IsIntegerType(opaque_compiler_type_t type,
                                      bool &is_signed) {
  if (!type)
    return false;
  FortranType *fortran_type = static_cast<FortranType *>(type);
  if (fortran_type->GetKind() == FortranType::KIND_INTEGER) {
    is_signed = true;
    return true;
  }
  return false;
}

bool TypeSystemFortran::IsFloatingPointType(opaque_compiler_type_t type) {
  int kind = static_cast<FortranType *>(type)->GetKind();
  if (kind == FortranType::KIND_REAL)
    return true;
  return false;
}

bool TypeSystemFortran::DumpTypeValue(
    lldb::opaque_compiler_type_t type, Stream &s, lldb::Format format,
    const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, ExecutionContextScope *exe_scope) {
  if (!type)
    return false;

  FortranType *fortran_type = static_cast<FortranType *>(type);
  int type_kind = fortran_type->GetKind();
  DataExtractor format_data;
  switch (type_kind) {
  case FortranType::KIND_INTEGER:
  case FortranType::KIND_REAL:
  case FortranType::KIND_LOGICAL:
    format_data.SetData(data, 0, data.GetByteSize());
    format_data.SetAddressByteSize(data.GetAddressByteSize());
    format_data.SetByteOrder(m_byte_order);
    return DumpDataExtractor(format_data, &s, data_offset, format,
                             data_byte_size, 1 /*item_count*/, UINT32_MAX,
                             LLDB_INVALID_ADDRESS, bitfield_bit_size,
                             bitfield_bit_offset, exe_scope);
  case FortranType::KIND_COMPLEX:
    // For Complex we print the value exactly how Fortran prints it
    format_data.SetData(data, 0, data.GetByteSize());
    format_data.SetAddressByteSize(data.GetAddressByteSize());
    format_data.SetByteOrder(m_byte_order);
    return DumpComplex(s, data, data_offset, data_byte_size);
  default:
    Host::SystemLog(lldb::eSeverityError,
                    "Error: DumpTypeValue not handled yet.\n");
    return false;
  }
}