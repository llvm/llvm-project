//===-- TypeSystemFortran.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

// TODO: Should this be a lambda?
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
// account for that here
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

// FIXME: Is support for all Fortran ISO the goal?
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

/// Returns the type assosciated with the kind and bitsize, or creates it
/// if it is not in the map
CompilerType TypeSystemFortran::GetOrCreateFortranType(int kind,
                                                       uint64_t bitsize,
                                                       ConstString name) {
  FortranType *type = m_basic_type_map[{kind, bitsize}].get();
  if (type)
    return CompilerType(weak_from_this(), (void *)type);
  auto new_type_up = std::make_unique<FortranType>(kind, name, bitsize);
  FortranType *raw_ptr = new_type_up.get();

  m_basic_type_map[{kind, bitsize}] = std::move(new_type_up);

  return CompilerType(weak_from_this(), (void *)raw_ptr);
}

/// Returns the type assosciated with the name, or creates it
/// if it is not in the map
CompilerType TypeSystemFortran::GetOrCreateFortranFunction(
    ConstString name, const SmallVectorImpl<CompilerType> &parameters) {
  FortranType *type = m_function_map[name].get();
  if (type)
    return CompilerType(weak_from_this(), (void *)type);
  auto new_type_up = std::make_unique<FortranFunction>(name, parameters);
  FortranType *raw_ptr = new_type_up.get();

  m_function_map[name] = std::move(new_type_up);

  return CompilerType(weak_from_this(), (void *)raw_ptr);
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

CompilerType TypeSystemFortran::CreateArrayType(FortranArrayMetadata array_info,
                                                uint64_t total_array_size,
                                                uint64_t total_elements) {

  size_t rank = array_info.dimensions.size();
  llvm::SmallVector<ArrayShape, 2> array_shapes;
  ConstString type_name;
  for (int idx = 0; idx < rank; ++idx) {
    ArrayShape shape;
    ArrayBound lb;
    ArrayBound ub;
    ArrayBound::Category bound_category;
    uint64_t byte_stride;
    int64_t dim_elements = -1;

    if (std::holds_alternative<std::monostate>(
            array_info.dimensions[idx].byte_stride)) {
      auto byte_stride_or_err = array_info.element_type.GetByteSize(nullptr);
      if (!byte_stride_or_err) {
        LLDB_LOG_ERROR(GetLog(LLDBLog::Types), byte_stride_or_err.takeError(),
                       "{0}");
        return CompilerType();
      }
      shape.SetByteStride(*byte_stride_or_err);
    }

    else if (std::holds_alternative<uint64_t>(
                 array_info.dimensions[idx].byte_stride))
      shape.SetByteStride(
          std::get<uint64_t>(array_info.dimensions[idx].byte_stride));
    // If the elements for this dimension are unknown it is either colon or star
    // Star can only appear as the last bound

    if (!std::holds_alternative<uint64_t>(
            array_info.dimensions[idx].element_count)) {
      bound_category = ArrayBound::Category::Colon;

      if (std::holds_alternative<std::monostate>(
              array_info.dimensions[idx].element_count) &&
          idx == rank - 1)
        bound_category = ArrayBound::Category::Star;
    }

    else {
      bound_category = ArrayBound::Category::Explicit;
      dim_elements =
          std::get<uint64_t>(array_info.dimensions[idx].element_count);
      shape.SetElementCount(dim_elements);
    }

    lb.SetCategory(bound_category);
    ub.SetCategory(bound_category);

    if (std::holds_alternative<int64_t>(
            array_info.dimensions[idx].lower_bound)) {
      int64_t lbound =
          std::get<int64_t>(array_info.dimensions[idx].lower_bound);
      lb.SetBound(lbound);
      if (dim_elements != -1)
        ub.SetBound(lbound + dim_elements - 1);
    }

    else if (std::holds_alternative<std::monostate>(
                 array_info.dimensions[idx].lower_bound)) {
      lb.SetBound(1);
      ub.SetBound(dim_elements);
    }

    if (std::holds_alternative<int64_t>(array_info.dimensions[idx].upper_bound))
      ub.SetBound(std::get<int64_t>(array_info.dimensions[idx].upper_bound));

    shape.SetLowerBound(lb);
    shape.SetUpperBound(ub);

    if (std::holds_alternative<DWARFExpressionList>(
            array_info.dimensions[idx].upper_bound))
      shape.SetUpperBoundExpression(std::get<DWARFExpressionList>(
          array_info.dimensions[idx].upper_bound));
    if (std::holds_alternative<DWARFExpressionList>(
            array_info.dimensions[idx].lower_bound))
      shape.SetLowerBoundExpression(std::get<DWARFExpressionList>(
          array_info.dimensions[idx].lower_bound));
    if (std::holds_alternative<DWARFExpressionList>(
            array_info.dimensions[idx].element_count))
      shape.SetElementCountExpression(std::get<DWARFExpressionList>(
          array_info.dimensions[idx].element_count));
    if (std::holds_alternative<DWARFExpressionList>(
            array_info.dimensions[idx].byte_stride))
      shape.SetByteStrideExpression(std::get<DWARFExpressionList>(
          array_info.dimensions[idx].byte_stride));

    array_shapes.push_back(shape);
  }
  ConstString array_type_name =
      CreateArrayTypeName(array_info.element_type, array_shapes,
                          array_info.is_allocatable, array_info.is_star);

  FortranArray *child_type = new FortranArray(
      array_info.element_type, array_shapes, array_type_name, total_array_size,
      array_info.is_allocatable, array_info.is_dynamic, total_elements,
      array_info.allocated_exp, array_info.data_location_exp);

  return CompilerType(weak_from_this(), (void *)child_type);
}

llvm::Expected<CompilerType> TypeSystemFortran::GetChildCompilerTypeAtIndex(
    lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
    bool transparent_pointers, bool omit_empty_base_classes,
    bool ignore_array_bounds, std::string &child_name,
    uint32_t &child_byte_size, int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class, bool &child_is_deref_of_parent,
    ValueObject *valobj, uint64_t &language_flags) {
  if (!type)
    return createStringError(
        inconvertibleErrorCode(),
        "Failed to craft intermediate Fortran array type: type is null.");

  FortranType *super_type = static_cast<FortranType *>(type);

  if (super_type->GetKind() != FortranType::KIND_ARRAY)
    return CompilerType();

  FortranArray *fortran_type = static_cast<FortranArray *>(super_type);

  if (!fortran_type)
    return CompilerType();

  if (fortran_type->IsDynamic())
    return CompilerType();
  llvm::ArrayRef<ArrayShape> old_dimensions = fortran_type->GetDimensions();
  int64_t ub = old_dimensions.front().GetUpperBound().GetBound();
  int64_t lb = old_dimensions.front().GetLowerBound().GetBound();
  uint64_t num_elements = old_dimensions.front().GetNumberOfElements();

  if (idx >= num_elements)
    return CompilerType();

  if (old_dimensions.size() > 1) {

    llvm::SmallVector<ArrayShape, 2> new_dimensions(old_dimensions.begin() + 1,
                                                    old_dimensions.end());

    ArrayShape old_first_dimension = old_dimensions.front();
    uint64_t new_byte_stride;

    if (old_first_dimension.GetByteStride() != 0)
      new_byte_stride = old_first_dimension.GetNumberOfElements() *
                        old_first_dimension.GetByteStride();
    else
      new_byte_stride = old_first_dimension.GetNumberOfElements() *
                        fortran_type->GetElementByteSize();

    new_dimensions.front().SetByteStride(new_byte_stride);
    bool is_star = new_dimensions.back().GetLowerBound().IsStar();
    bool is_allocatable = fortran_type->IsAllocatable();
    uint64_t new_total_elements = fortran_type->GetTotalElements() /
                                  old_first_dimension.GetNumberOfElements();
    if (old_dimensions.front().GetByteStride() != 0)
      child_byte_offset = idx * old_dimensions.front().GetByteStride();
    else
      child_byte_offset = idx * fortran_type->GetElementByteSize();

    uint64_t last_dim_stride = new_dimensions.back().GetByteStride();

    if (last_dim_stride != 0) {
      child_byte_size =
          new_dimensions.back().GetNumberOfElements() * last_dim_stride;
    } else {
      child_byte_size = new_total_elements * fortran_type->GetElementByteSize();
    }
    child_name = llvm::formatv("[{0}]", lb + idx);
    ConstString type_name =
        CreateArrayTypeName(fortran_type->GetElementType(), new_dimensions,
                            is_allocatable, is_star);
    FortranArray *array_type = new FortranArray(
        fortran_type->GetElementType(), new_dimensions, type_name,
        child_byte_size, false, false, new_total_elements,
        DWARFExpressionList(), DWARFExpressionList());
    return CompilerType(weak_from_this(), (void *)array_type);
  } else {
    child_byte_offset = idx * old_dimensions.front().GetByteStride();
    child_byte_size = fortran_type->GetElementByteSize();
    return fortran_type->GetElementType();
  }
}

/// Returns the type name upper-cased to follow Fortran's general style
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
  case FortranType::KIND_ARRAY:
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

Expected<uint32_t>
TypeSystemFortran::GetNumChildren(opaque_compiler_type_t type,
                                  bool omit_empty_base_classes,
                                  const ExecutionContext *exe_ctx) {
  if (!type)
    return createStringError(
        inconvertibleErrorCode(),
        "Couldn't get number of children, bad Fortran type.");
  FortranType *super_type = static_cast<FortranType *>(type);

  switch (super_type->GetKind()) {
  case FortranType::KIND_ARRAY: {
    FortranArray *fortran_array = static_cast<FortranArray *>(super_type);

    if (!fortran_array)
      return createStringError(
          inconvertibleErrorCode(),
          "Couldn't get number of children, bad Fortran type.");

    // Fetch the number of elements
    if (!fortran_array->IsDynamic())
      return fortran_array->GetDimensions().front().GetNumberOfElements();
    return 0;
  }
  default:
    return 0;
  }
}

CompilerType
TypeSystemFortran::GetPointeeType(lldb::opaque_compiler_type_t type) {
  if (!type)
    return CompilerType();
  FortranType *fortran_type = static_cast<FortranType *>(type);
  if (fortran_type->GetKind() != FortranType::KIND_POINTER)
    return CompilerType();
  FortranPointer *pointer_type = static_cast<FortranPointer *>(fortran_type);
  FortranType *pointee_type = pointer_type->GetPointee();
  if (!pointee_type)
    return CompilerType();

  return CompilerType(weak_from_this(), (void *)pointee_type);
}

CompilerType
TypeSystemFortran::GetPointerType(lldb::opaque_compiler_type_t type) {
  if (!type)
    return CompilerType();

  FortranType *pointee_type = static_cast<FortranType *>(type);
  if (!pointee_type)
    return CompilerType();
  std::string ptr_name = pointee_type->GetName().GetStringRef().str() + " *";
  FortranPointer *pointer_type =
      new FortranPointer(pointee_type, ConstString(ptr_name));
  return CompilerType(weak_from_this(), (void *)pointer_type);
}

llvm::Expected<CompilerType> TypeSystemFortran::GetDereferencedType(
    lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx,
    std::string &deref_name, uint32_t &deref_byte_size,
    int32_t &deref_byte_offset, ValueObject *valobj, uint64_t &language_flags) {
  if (!type)
    return createStringError(inconvertibleErrorCode(),
                             "Couldn't get dereferenced type, type is null.");

  FortranType *fortran_type = static_cast<FortranType *>(type);

  if (!fortran_type)
    return createStringError(
        inconvertibleErrorCode(),
        "Couldn't get dereferenced type, type is not a fortran type.");

  if (fortran_type->GetKind() != FortranType::KIND_POINTER)
    return CompilerType();

  CompilerType pointee_type = GetPointeeType(type);

  deref_byte_offset = 0;

  if (exe_ctx && pointee_type.IsValid()) {
    llvm::Expected<uint64_t> size_or_err =
        pointee_type.GetByteSize(exe_ctx->GetBestExecutionContextScope());
    if (size_or_err)
      deref_byte_size = *size_or_err;
  }

  return pointee_type;
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

bool TypeSystemFortran::IsArrayType(lldb::opaque_compiler_type_t type,
                                    CompilerType *element_type, uint64_t *size,
                                    bool *is_incomplete) {
  if (element_type)
    element_type->Clear();
  if (size)
    *size = 0;
  if (is_incomplete)
    *is_incomplete = false;

  FortranType *super_type = static_cast<FortranType *>(type);
  if (!super_type)
    return false;

  if (super_type->GetKind() != FortranType::KIND_ARRAY)
    return false;

  FortranArray *array_type = static_cast<FortranArray *>(super_type);

  if (!array_type)
    return false;

  if (element_type)
    *element_type = array_type->GetElementType();
  // TODO: If it isn't we have to evaluate the DWARFExpressionList
  if (!array_type->IsDynamic() && size)
    *size = array_type->GetTotalElements();

  return true;
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
