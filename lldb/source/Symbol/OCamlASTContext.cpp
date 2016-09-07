//===-- OCamlASTContext.cpp ----------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/OCamlASTContext.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"

#include "Plugins/SymbolFile/DWARF/DWARFASTParserOCaml.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
class OCamlASTContext::OCamlType {
public:
  enum LLVMCastKind {
    eKindPrimitive,
    eKindObject,
    eKindReference,
    eKindArray,
    kNumKinds
  };

  OCamlType(LLVMCastKind kind) : m_kind(kind) {}

  virtual ~OCamlType() = default;

  virtual ConstString GetName() = 0;

  virtual void Dump(Stream *s) = 0;

  virtual bool IsCompleteType() = 0;

  LLVMCastKind getKind() const { return m_kind; }

private:
  LLVMCastKind m_kind;
};

} // end of namespace lldb_private

namespace {

class OCamlPrimitiveType : public OCamlASTContext::OCamlType {
public:
  enum TypeKind {
    eTypeInt,
  };

  OCamlPrimitiveType(TypeKind type_kind, uint32_t byte_size)
      : OCamlType(OCamlType::eKindPrimitive), m_type_kind(type_kind),
        m_type(ConstString()), m_byte_size(byte_size) {}

  OCamlPrimitiveType(TypeKind type_kind, ConstString s, uint32_t byte_size)
      : OCamlType(OCamlType::eKindPrimitive), m_type_kind(type_kind), m_type(s),
        m_byte_size(byte_size) {}

  ConstString GetName() override {
    switch (m_type_kind) {
    case eTypeInt:
      return m_type;
    }
    return ConstString();
  }

  TypeKind GetTypeKind() { return m_type_kind; }

  void Dump(Stream *s) override { s->Printf("%s\n", GetName().GetCString()); }

  bool IsCompleteType() override { return true; }

  static bool classof(const OCamlType *ot) {
    return ot->getKind() == OCamlType::eKindPrimitive;
  }

  uint64_t GetByteSize() const { return m_byte_size; }

private:
  const TypeKind m_type_kind;
  const ConstString m_type;
  uint64_t m_byte_size;
};
}

OCamlASTContext::OCamlASTContext()
    : TypeSystem(eKindOCaml), m_pointer_byte_size(0) {}

OCamlASTContext::~OCamlASTContext() {}

ConstString OCamlASTContext::GetPluginNameStatic() {
  return ConstString("ocaml");
}

ConstString OCamlASTContext::GetPluginName() {
  return OCamlASTContext::GetPluginNameStatic();
}

uint32_t OCamlASTContext::GetPluginVersion() { return 1; }

lldb::TypeSystemSP
OCamlASTContext::CreateInstance(lldb::LanguageType language, Module *module,
                                Target *target, const char *compiler_options) {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_LANGUAGE));

  if (language == lldb::eLanguageTypeOCaml) {
    std::shared_ptr<OCamlASTContext> ocaml_ast_sp;
    ArchSpec arch;

    if (module) {
      arch = module->GetArchitecture();

      ObjectFile *objfile = module->GetObjectFile();
      ArchSpec object_arch;

      if (!objfile || !objfile->GetArchitecture(object_arch))
        return lldb::TypeSystemSP();

      ocaml_ast_sp = std::shared_ptr<OCamlASTContext>(new OCamlASTContext);

      if (log) {
        log->Printf(
            "((Module*)%p) [%s]->GetOCamlASTContext() = %p", (void *)module,
            module->GetFileSpec().GetFilename().AsCString("<anonymous>"),
            (void *)ocaml_ast_sp.get());
      }

    } else if (target) {
      arch = target->GetArchitecture();
      ocaml_ast_sp = std::shared_ptr<OCamlASTContextForExpr>(
          new OCamlASTContextForExpr(target->shared_from_this()));

      if (log) {
        log->Printf("((Target*)%p)->GetOCamlASTContext() = %p", (void *)target,
                    (void *)ocaml_ast_sp.get());
      }
    }

    if (arch.IsValid()) {
      ocaml_ast_sp->SetAddressByteSize(arch.GetAddressByteSize());
      return ocaml_ast_sp;
    }
  }

  return lldb::TypeSystemSP();
}

void OCamlASTContext::EnumerateSupportedLanguages(
    std::set<lldb::LanguageType> &languages_for_types,
    std::set<lldb::LanguageType> &languages_for_expressions) {
  static std::vector<lldb::LanguageType> s_supported_languages_for_types(
      {lldb::eLanguageTypeOCaml});
  static std::vector<lldb::LanguageType> s_supported_languages_for_expressions(
      {});

  languages_for_types.insert(s_supported_languages_for_types.begin(),
                             s_supported_languages_for_types.end());
  languages_for_expressions.insert(
      s_supported_languages_for_expressions.begin(),
      s_supported_languages_for_expressions.end());
}

void OCamlASTContext::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "OCaml AST context plug-in", CreateInstance,
                                EnumerateSupportedLanguages);
}

void OCamlASTContext::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

DWARFASTParser *OCamlASTContext::GetDWARFParser() {
  if (!m_dwarf_ast_parser_ap) {
    m_dwarf_ast_parser_ap.reset(new DWARFASTParserOCaml(*this));
  }

  return m_dwarf_ast_parser_ap.get();
}

bool OCamlASTContext::IsArrayType(lldb::opaque_compiler_type_t type,
                                  CompilerType *element_type, uint64_t *size,
                                  bool *is_incomplete) {
  return false;
}

bool OCamlASTContext::IsVectorType(lldb::opaque_compiler_type_t type,
                                   CompilerType *element_type, uint64_t *size) {
  return false;
}

bool OCamlASTContext::IsAggregateType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool OCamlASTContext::IsBeingDefined(lldb::opaque_compiler_type_t type) {
  return false;
}

bool OCamlASTContext::IsCharType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool OCamlASTContext::IsCompleteType(lldb::opaque_compiler_type_t type) {
  return static_cast<OCamlPrimitiveType *>(type)->IsCompleteType();
}

bool OCamlASTContext::IsConst(lldb::opaque_compiler_type_t type) {
  return false;
}

bool OCamlASTContext::IsCStringType(lldb::opaque_compiler_type_t type,
                                    uint32_t &length) {
  return false;
}

bool OCamlASTContext::IsDefined(lldb::opaque_compiler_type_t type) {
  return type != nullptr;
}

bool OCamlASTContext::IsFloatingPointType(lldb::opaque_compiler_type_t type,
                                          uint32_t &count, bool &is_complex) {
  return false;
}

bool OCamlASTContext::IsFunctionType(lldb::opaque_compiler_type_t type,
                                     bool *is_variadic_ptr) {
  return false;
}

uint32_t
OCamlASTContext::IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                                        CompilerType *base_type_ptr) {
  return false;
}

size_t OCamlASTContext::GetNumberOfFunctionArguments(
    lldb::opaque_compiler_type_t type) {
  return 0;
}

CompilerType
OCamlASTContext::GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                                            const size_t index) {
  return CompilerType();
}

bool OCamlASTContext::IsFunctionPointerType(lldb::opaque_compiler_type_t type) {
  return IsFunctionType(type);
}

bool OCamlASTContext::IsBlockPointerType(
    lldb::opaque_compiler_type_t type,
    CompilerType *function_pointer_type_ptr) {
  return false;
}

bool OCamlASTContext::IsIntegerType(lldb::opaque_compiler_type_t type,
                                    bool &is_signed) {
  if (OCamlPrimitiveType *ptype =
          llvm::dyn_cast<OCamlPrimitiveType>(static_cast<OCamlType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case OCamlPrimitiveType::eTypeInt:
      is_signed = true;
      return true;
    }
  }

  is_signed = false;
  return false;
}

bool OCamlASTContext::IsPolymorphicClass(lldb::opaque_compiler_type_t type) {
  return false;
}

bool OCamlASTContext::IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                                            CompilerType *target_type,
                                            bool check_cplusplus,
                                            bool check_objc, bool check_swift) {
  return false;
}

bool OCamlASTContext::IsRuntimeGeneratedType(
    lldb::opaque_compiler_type_t type) {
  return false;
}

bool OCamlASTContext::IsPointerType(lldb::opaque_compiler_type_t type,
                                    CompilerType *pointee_type) {
  if (pointee_type)
    pointee_type->Clear();
  return false;
}

bool OCamlASTContext::IsPointerOrReferenceType(
    lldb::opaque_compiler_type_t type, CompilerType *pointee_type) {
  return IsPointerType(type, pointee_type);
}

bool OCamlASTContext::IsReferenceType(lldb::opaque_compiler_type_t type,
                                      CompilerType *pointee_type,
                                      bool *is_rvalue) {
  return false;
}

bool OCamlASTContext::IsScalarType(lldb::opaque_compiler_type_t type) {
  return llvm::isa<OCamlPrimitiveType>(static_cast<OCamlType *>(type));
}

bool OCamlASTContext::IsTypedefType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool OCamlASTContext::IsVoidType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool OCamlASTContext::SupportsLanguage(lldb::LanguageType language) {
  return language == lldb::eLanguageTypeOCaml;
}

bool OCamlASTContext::GetCompleteType(lldb::opaque_compiler_type_t type) {
  if (IsCompleteType(type))
    return true;

  return false;
}

uint32_t OCamlASTContext::GetPointerByteSize() { return m_pointer_byte_size; }

ConstString OCamlASTContext::GetTypeName(lldb::opaque_compiler_type_t type) {
  if (type)
    return static_cast<OCamlPrimitiveType *>(type)->GetName();

  return ConstString();
}

uint32_t
OCamlASTContext::GetTypeInfo(lldb::opaque_compiler_type_t type,
                             CompilerType *pointee_or_element_compiler_type) {
  if (pointee_or_element_compiler_type)
    pointee_or_element_compiler_type->Clear();
  if (!type)
    return 0;

  if (OCamlPrimitiveType *ptype =
          llvm::dyn_cast<OCamlPrimitiveType>(static_cast<OCamlType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case OCamlPrimitiveType::eTypeInt:
      return eTypeHasValue | eTypeIsBuiltIn | eTypeIsScalar | eTypeIsInteger |
             eTypeIsSigned;
    }
  }

  return 0;
}

lldb::TypeClass
OCamlASTContext::GetTypeClass(lldb::opaque_compiler_type_t type) {
  if (llvm::isa<OCamlPrimitiveType>(static_cast<OCamlType *>(type)))
    return eTypeClassBuiltin;

  return lldb::eTypeClassInvalid;
}

lldb::BasicType
OCamlASTContext::GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) {
  return lldb::eBasicTypeInvalid;
}

lldb::LanguageType
OCamlASTContext::GetMinimumLanguage(lldb::opaque_compiler_type_t type) {
  return lldb::eLanguageTypeOCaml;
}

unsigned OCamlASTContext::GetTypeQualifiers(lldb::opaque_compiler_type_t type) {
  return 0;
}

//----------------------------------------------------------------------
// Creating related types
//----------------------------------------------------------------------

CompilerType
OCamlASTContext::GetArrayElementType(lldb::opaque_compiler_type_t type,
                                     uint64_t *stride) {
  return CompilerType();
}

CompilerType
OCamlASTContext::GetCanonicalType(lldb::opaque_compiler_type_t type) {
  return CompilerType(this, type);
}

CompilerType
OCamlASTContext::GetInstanceType(lldb::opaque_compiler_type_t type) {
  return CompilerType(this, type);
}

CompilerType
OCamlASTContext::GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) {
  return CompilerType(this, type);
}

int OCamlASTContext::GetFunctionArgumentCount(
    lldb::opaque_compiler_type_t type) {
  return GetNumberOfFunctionArguments(type);
}

CompilerType OCamlASTContext::GetFunctionArgumentTypeAtIndex(
    lldb::opaque_compiler_type_t type, size_t idx) {
  return GetFunctionArgumentAtIndex(type, idx);
}

CompilerType
OCamlASTContext::GetFunctionReturnType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

size_t
OCamlASTContext::GetNumMemberFunctions(lldb::opaque_compiler_type_t type) {
  return 0;
}

TypeMemberFunctionImpl
OCamlASTContext::GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                                          size_t idx) {
  return TypeMemberFunctionImpl();
}

CompilerType
OCamlASTContext::GetNonReferenceType(lldb::opaque_compiler_type_t type) {
  return CompilerType(this, type);
}

CompilerType
OCamlASTContext::GetPointeeType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType
OCamlASTContext::GetPointerType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType
OCamlASTContext::GetTypedefedType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType
OCamlASTContext::GetUnboundType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType GetUnboundType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType OCamlASTContext::GetBasicTypeFromAST(lldb::BasicType basic_type) {
  return CompilerType();
}

CompilerType
OCamlASTContext::GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                     size_t bit_size) {
  return CompilerType();
}

uint64_t OCamlASTContext::GetBitSize(lldb::opaque_compiler_type_t type,
                                     ExecutionContextScope *exe_scope) {
  if (OCamlPrimitiveType *ptype =
          llvm::dyn_cast<OCamlPrimitiveType>(static_cast<OCamlType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case OCamlPrimitiveType::eTypeInt:
      return ptype->GetByteSize() * 8;
    }
  }
  return 0;
}

uint64_t OCamlASTContext::GetByteStride(lldb::opaque_compiler_type_t type) {
  if (OCamlPrimitiveType *ptype =
          llvm::dyn_cast<OCamlPrimitiveType>(static_cast<OCamlType *>(type))) {
    // This is very likely insufficient as it doesn't take any kind of
    // post-structure packing into account.
    return ptype->GetByteSize();
  }
  return 0;
}

lldb::Encoding OCamlASTContext::GetEncoding(lldb::opaque_compiler_type_t type,
                                            uint64_t &count) {
  count = 1;
  bool is_signed;
  if (IsIntegerType(type, is_signed))
    return is_signed ? lldb::eEncodingSint : lldb::eEncodingUint;
  bool is_complex;
  uint32_t complex_count;
  if (IsFloatingPointType(type, complex_count, is_complex)) {
    count = complex_count;
    return lldb::eEncodingIEEE754;
  }
  if (IsPointerType(type))
    return lldb::eEncodingUint;
  return lldb::eEncodingInvalid;
}

lldb::Format OCamlASTContext::GetFormat(lldb::opaque_compiler_type_t type) {
  if (!type)
    return lldb::eFormatDefault;
  return lldb::eFormatBytes;
}

size_t OCamlASTContext::GetTypeBitAlign(lldb::opaque_compiler_type_t type) {
  return 0;
}

uint32_t OCamlASTContext::GetNumChildren(lldb::opaque_compiler_type_t type,
                                         bool omit_empty_base_classes) {
  if (!type || !GetCompleteType(type))
    return 0;

  return GetNumFields(type);
}

uint32_t OCamlASTContext::GetNumFields(lldb::opaque_compiler_type_t type) {
  if (!type || !GetCompleteType(type))
    return 0;
  return 0;
}

CompilerType OCamlASTContext::GetFieldAtIndex(lldb::opaque_compiler_type_t type,
                                              size_t idx, std::string &name,
                                              uint64_t *bit_offset_ptr,
                                              uint32_t *bitfield_bit_size_ptr,
                                              bool *is_bitfield_ptr) {
  if (bit_offset_ptr)
    *bit_offset_ptr = 0;
  if (bitfield_bit_size_ptr)
    *bitfield_bit_size_ptr = 0;
  if (is_bitfield_ptr)
    *is_bitfield_ptr = false;

  if (!type || !GetCompleteType(type))
    return CompilerType();

  return CompilerType();
}

CompilerType OCamlASTContext::GetChildCompilerTypeAtIndex(
    lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
    bool transparent_pointers, bool omit_empty_base_classes,
    bool ignore_array_bounds, std::string &child_name,
    uint32_t &child_byte_size, int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class, bool &child_is_deref_of_parent,
    ValueObject *valobj, uint64_t &language_flags) {
  child_name.clear();
  child_byte_size = 0;
  child_byte_offset = 0;
  child_bitfield_bit_size = 0;
  child_bitfield_bit_offset = 0;
  child_is_base_class = false;
  child_is_deref_of_parent = false;
  language_flags = 0;

  if (!type || !GetCompleteType(type))
    return CompilerType();

  return CompilerType();
}

uint32_t
OCamlASTContext::GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                                         const char *name,
                                         bool omit_empty_base_classes) {
  if (!type || !GetCompleteType(type))
    return UINT_MAX;

  return UINT_MAX;
}

size_t OCamlASTContext::GetIndexOfChildMemberWithName(
    lldb::opaque_compiler_type_t type, const char *name,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  uint32_t index = GetIndexOfChildWithName(type, name, omit_empty_base_classes);
  if (index == UINT_MAX)
    return 0;
  child_indexes.push_back(index);
  return 1;
}

size_t
OCamlASTContext::ConvertStringToFloatValue(lldb::opaque_compiler_type_t type,
                                           const char *s, uint8_t *dst,
                                           size_t dst_size) {
  assert(false);
  return 0;
}
//----------------------------------------------------------------------
// Dumping types
//----------------------------------------------------------------------

void OCamlASTContext::DumpValue(
    lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, Stream *s,
    lldb::Format format, const DataExtractor &data,
    lldb::offset_t data_byte_offset, size_t data_byte_size,
    uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset, bool show_types,
    bool show_summary, bool verbose, uint32_t depth) {
  if (!type) {
    s->Printf("no type\n");
    return;
  }

  s->Printf("no value\n");

  if (show_summary)
    DumpSummary(type, exe_ctx, s, data, data_byte_offset, data_byte_size);
}

bool OCamlASTContext::DumpTypeValue(
    lldb::opaque_compiler_type_t type, Stream *s, lldb::Format format,
    const DataExtractor &data, lldb::offset_t byte_offset, size_t byte_size,
    uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
    ExecutionContextScope *exe_scope, bool is_base_class) {
  if (!type) {
    s->Printf("no type value\n");
    return false;
  }

  if (IsScalarType(type)) {
    return data.Dump(s, byte_offset, format, byte_size, 1, UINT64_MAX,
                     LLDB_INVALID_ADDRESS, bitfield_bit_size,
                     bitfield_bit_offset, exe_scope);
  }

  return false;
}

void OCamlASTContext::DumpSummary(lldb::opaque_compiler_type_t type,
                                  ExecutionContext *exe_ctx, Stream *s,
                                  const DataExtractor &data,
                                  lldb::offset_t data_offset,
                                  size_t data_byte_size) {
  s->Printf("no summary\n");
}

void OCamlASTContext::DumpTypeDescription(lldb::opaque_compiler_type_t type) {
  StreamFile s(stdout, false);
  DumpTypeDescription(type, &s);
}

void OCamlASTContext::DumpTypeDescription(lldb::opaque_compiler_type_t type,
                                          Stream *s) {
  static_cast<OCamlType *>(type)->Dump(s);
}

CompilerType OCamlASTContext::CreateBaseType(const ConstString &name,
                                             uint64_t byte_size) {
  if (m_base_type_map.empty()) {
    OCamlPrimitiveType *type = new OCamlPrimitiveType(
        OCamlPrimitiveType::eTypeInt, ConstString("ocaml_int"), byte_size);
    m_base_type_map.emplace(type->GetName(),
                            std::unique_ptr<OCamlASTContext::OCamlType>(type));
  }

  auto it = m_base_type_map.find(name);
  if (it == m_base_type_map.end()) {
    OCamlPrimitiveType *type =
        new OCamlPrimitiveType(OCamlPrimitiveType::eTypeInt, name, byte_size);
    it = m_base_type_map
             .emplace(name, std::unique_ptr<OCamlASTContext::OCamlType>(type))
             .first;
  }

  return CompilerType(this, it->second.get());
}
