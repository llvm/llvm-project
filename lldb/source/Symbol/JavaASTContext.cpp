//===-- JavaASTContext.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <sstream>

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/JavaASTContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/Target.h"

#include "Plugins/SymbolFile/DWARF/DWARFASTParserJava.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {

class JavaASTContext::JavaType {
public:
  enum LLVMCastKind {
    eKindPrimitive,
    eKindObject,
    eKindReference,
    eKindArray,
    kNumKinds
  };

  JavaType(LLVMCastKind kind) : m_kind(kind) {}

  virtual ~JavaType() = default;

  virtual ConstString GetName() = 0;

  virtual void Dump(Stream *s) = 0;

  virtual bool IsCompleteType() = 0;

  LLVMCastKind getKind() const { return m_kind; }

private:
  LLVMCastKind m_kind;
};

} // end of namespace lldb_private

namespace {

class JavaPrimitiveType : public JavaASTContext::JavaType {
public:
  enum TypeKind {
    eTypeByte,
    eTypeShort,
    eTypeInt,
    eTypeLong,
    eTypeFloat,
    eTypeDouble,
    eTypeBoolean,
    eTypeChar,
  };

  JavaPrimitiveType(TypeKind type_kind)
      : JavaType(JavaType::eKindPrimitive), m_type_kind(type_kind) {}

  ConstString GetName() override {
    switch (m_type_kind) {
    case eTypeByte:
      return ConstString("byte");
    case eTypeShort:
      return ConstString("short");
    case eTypeInt:
      return ConstString("int");
    case eTypeLong:
      return ConstString("long");
    case eTypeFloat:
      return ConstString("float");
    case eTypeDouble:
      return ConstString("double");
    case eTypeBoolean:
      return ConstString("boolean");
    case eTypeChar:
      return ConstString("char");
    }
    return ConstString();
  }

  TypeKind GetTypeKind() { return m_type_kind; }

  void Dump(Stream *s) override { s->Printf("%s\n", GetName().GetCString()); }

  bool IsCompleteType() override { return true; }

  static bool classof(const JavaType *jt) {
    return jt->getKind() == JavaType::eKindPrimitive;
  }

private:
  const TypeKind m_type_kind;
};

class JavaDynamicType : public JavaASTContext::JavaType {
public:
  JavaDynamicType(LLVMCastKind kind, const ConstString &linkage_name)
      : JavaType(kind), m_linkage_name(linkage_name),
        m_dynamic_type_id(nullptr) {}

  ConstString GetLinkageName() const { return m_linkage_name; }

  void SetDynamicTypeId(const DWARFExpression &type_id) {
    m_dynamic_type_id = type_id;
  }

  uint64_t CalculateDynamicTypeId(ExecutionContext *exe_ctx,
                                  ValueObject &value_obj) {
    if (!m_dynamic_type_id.IsValid())
      return UINT64_MAX;

    Value obj_load_address = value_obj.GetValue();
    obj_load_address.ResolveValue(exe_ctx);
    obj_load_address.SetValueType(Value::eValueTypeLoadAddress);

    Value result;
    if (m_dynamic_type_id.Evaluate(exe_ctx->GetBestExecutionContextScope(),
                                   nullptr, nullptr, 0, &obj_load_address,
                                   nullptr, result, nullptr)) {
      Error error;

      lldb::addr_t type_id_addr = result.GetScalar().UInt();
      lldb::ProcessSP process_sp = exe_ctx->GetProcessSP();
      if (process_sp)
        return process_sp->ReadUnsignedIntegerFromMemory(
            type_id_addr, process_sp->GetAddressByteSize(), UINT64_MAX, error);
    }

    return UINT64_MAX;
  }

public:
  ConstString m_linkage_name;
  DWARFExpression m_dynamic_type_id;
};

class JavaObjectType : public JavaDynamicType {
public:
  struct Field {
    ConstString m_name;
    CompilerType m_type;
    uint32_t m_offset;
  };

  JavaObjectType(const ConstString &name, const ConstString &linkage_name,
                 uint32_t byte_size)
      : JavaDynamicType(JavaType::eKindObject, linkage_name), m_name(name),
        m_byte_size(byte_size), m_base_class_offset(0), m_is_complete(false) {}

  ConstString GetName() override { return m_name; }

  uint32_t GetByteSize() const { return m_byte_size; }

  uint32_t GetNumFields() { return m_fields.size(); }

  void Dump(Stream *s) override {
    if (m_base_class.IsValid())
      s->Printf("%s : %s\n", GetName().GetCString(),
                m_base_class.GetTypeName().GetCString());
    else
      s->Printf("%s\n", GetName().GetCString());

    s->IndentMore();
    for (const Field &f : m_fields)
      s->Printf("%s %s\n", f.m_type.GetTypeName().GetCString(),
                f.m_name.GetCString());
    s->IndentLess();
  }

  Field *GetFieldAtIndex(size_t idx) {
    if (idx < m_fields.size())
      return &m_fields[idx];
    return nullptr;
  }

  CompilerType GetBaseClass() { return m_base_class; }

  uint32_t GetBaseClassOffset() { return m_base_class_offset; }

  uint32_t GetNumInterfaces() { return m_interfaces.size(); }

  CompilerType GetInterfaceAtIndex(uint32_t idx) {
    if (m_interfaces.size() < idx)
      return m_interfaces[idx];
    return CompilerType();
  }

  bool IsCompleteType() override { return m_is_complete; }

  void SetCompleteType(bool is_complete) {
    m_is_complete = is_complete;
    if (m_byte_size == 0) {
      // Try to calcualte the size of the object based on it's values
      for (const Field &field : m_fields) {
        uint32_t field_end = field.m_offset + field.m_type.GetByteSize(nullptr);
        if (field_end > m_byte_size)
          m_byte_size = field_end;
      }
    }
  }

  void AddBaseClass(const CompilerType &type, uint32_t offset) {
    // TODO: Check if type is an interface and add it to the interface list in
    // that case
    m_base_class = type;
    m_base_class_offset = offset;
  }

  void AddField(const ConstString &name, const CompilerType &type,
                uint32_t offset) {
    m_fields.push_back({name, type, offset});
  }

  static bool classof(const JavaType *jt) {
    return jt->getKind() == JavaType::eKindObject;
  }

private:
  ConstString m_name;
  uint32_t m_byte_size;
  CompilerType m_base_class;
  uint32_t m_base_class_offset;
  std::vector<CompilerType> m_interfaces;
  std::vector<Field> m_fields;
  bool m_is_complete;
};

class JavaReferenceType : public JavaASTContext::JavaType {
public:
  JavaReferenceType(CompilerType pointee_type)
      : JavaType(JavaType::eKindReference), m_pointee_type(pointee_type) {}

  static bool classof(const JavaType *jt) {
    return jt->getKind() == JavaType::eKindReference;
  }

  CompilerType GetPointeeType() { return m_pointee_type; }

  ConstString GetName() override {
    ConstString pointee_type_name =
        static_cast<JavaType *>(GetPointeeType().GetOpaqueQualType())
            ->GetName();
    return ConstString(std::string(pointee_type_name.AsCString()) + "&");
  }

  void Dump(Stream *s) override {
    static_cast<JavaType *>(m_pointee_type.GetOpaqueQualType())->Dump(s);
  }

  bool IsCompleteType() override { return m_pointee_type.IsCompleteType(); }

private:
  CompilerType m_pointee_type;
};

class JavaArrayType : public JavaDynamicType {
public:
  JavaArrayType(const ConstString &linkage_name, CompilerType element_type,
                const DWARFExpression &length_expression,
                lldb::addr_t data_offset)
      : JavaDynamicType(JavaType::eKindArray, linkage_name),
        m_element_type(element_type), m_length_expression(length_expression),
        m_data_offset(data_offset) {}

  static bool classof(const JavaType *jt) {
    return jt->getKind() == JavaType::eKindArray;
  }

  CompilerType GetElementType() { return m_element_type; }

  ConstString GetName() override {
    ConstString element_type_name =
        static_cast<JavaType *>(GetElementType().GetOpaqueQualType())
            ->GetName();
    return ConstString(std::string(element_type_name.AsCString()) + "[]");
  }

  void Dump(Stream *s) override { s->Printf("%s\n", GetName().GetCString()); }

  bool IsCompleteType() override { return m_length_expression.IsValid(); }

  uint32_t GetNumElements(ValueObject *value_obj) {
    if (!m_length_expression.IsValid())
      return UINT32_MAX;

    Error error;
    ValueObjectSP address_obj = value_obj->AddressOf(error);
    if (error.Fail())
      return UINT32_MAX;

    Value obj_load_address = address_obj->GetValue();
    obj_load_address.SetValueType(Value::eValueTypeLoadAddress);

    Value result;
    ExecutionContextScope *exec_ctx_scope = value_obj->GetExecutionContextRef()
                                                .Lock(true)
                                                .GetBestExecutionContextScope();
    if (m_length_expression.Evaluate(exec_ctx_scope, nullptr, nullptr, 0,
                                     nullptr, &obj_load_address, result,
                                     nullptr))
      return result.GetScalar().UInt();

    return UINT32_MAX;
  }

  uint64_t GetElementOffset(size_t idx) {
    return m_data_offset + idx * m_element_type.GetByteSize(nullptr);
  }

private:
  CompilerType m_element_type;
  DWARFExpression m_length_expression;
  lldb::addr_t m_data_offset;
};

} // end of anonymous namespace

ConstString JavaASTContext::GetPluginNameStatic() {
  return ConstString("java");
}

ConstString JavaASTContext::GetPluginName() {
  return JavaASTContext::GetPluginNameStatic();
}

uint32_t JavaASTContext::GetPluginVersion() { return 1; }

lldb::TypeSystemSP JavaASTContext::CreateInstance(lldb::LanguageType language,
                                                  Module *module,
                                                  Target *target,
                                                  const char *extra_options) {
  if (language == eLanguageTypeJava) {
    if (module)
      return std::make_shared<JavaASTContext>(module->GetArchitecture());
    if (target)
      return std::make_shared<JavaASTContext>(target->GetArchitecture());
    assert(false && "Either a module or a target has to be specifed to create "
                    "a JavaASTContext");
  }
  return lldb::TypeSystemSP();
}

void JavaASTContext::EnumerateSupportedLanguages(
    std::set<lldb::LanguageType> &languages_for_types,
    std::set<lldb::LanguageType> &languages_for_expressions) {
  static std::vector<lldb::LanguageType> s_languages_for_types(
      {lldb::eLanguageTypeJava});
  static std::vector<lldb::LanguageType> s_languages_for_expressions({});

  languages_for_types.insert(s_languages_for_types.begin(),
                             s_languages_for_types.end());
  languages_for_expressions.insert(s_languages_for_expressions.begin(),
                                   s_languages_for_expressions.end());
}

void JavaASTContext::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "AST context plug-in",
                                CreateInstance, EnumerateSupportedLanguages);
}

void JavaASTContext::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

JavaASTContext::JavaASTContext(const ArchSpec &arch)
    : TypeSystem(eKindJava), m_pointer_byte_size(arch.GetAddressByteSize()) {}

JavaASTContext::~JavaASTContext() {}

uint32_t JavaASTContext::GetPointerByteSize() { return m_pointer_byte_size; }

DWARFASTParser *JavaASTContext::GetDWARFParser() {
  if (!m_dwarf_ast_parser_ap)
    m_dwarf_ast_parser_ap.reset(new DWARFASTParserJava(*this));
  return m_dwarf_ast_parser_ap.get();
}

ConstString JavaASTContext::DeclGetName(void *opaque_decl) {
  return ConstString();
}

std::vector<CompilerDecl> JavaASTContext::DeclContextFindDeclByName(
    void *opaque_decl_ctx, ConstString name, const bool ignore_imported_decls) {
  return std::vector<CompilerDecl>();
}

bool JavaASTContext::DeclContextIsStructUnionOrClass(void *opaque_decl_ctx) {
  return false;
}

ConstString JavaASTContext::DeclContextGetName(void *opaque_decl_ctx) {
  return ConstString();
}

bool JavaASTContext::DeclContextIsClassMethod(
    void *opaque_decl_ctx, lldb::LanguageType *language_ptr,
    bool *is_instance_method_ptr, ConstString *language_object_name_ptr) {
  return false;
}

bool JavaASTContext::IsArrayType(lldb::opaque_compiler_type_t type,
                                 CompilerType *element_type, uint64_t *size,
                                 bool *is_incomplete) {
  if (element_type)
    element_type->Clear();
  if (size)
    *size = 0;
  if (is_incomplete)
    *is_incomplete = false;

  if (JavaArrayType *array =
          llvm::dyn_cast<JavaArrayType>(static_cast<JavaType *>(type))) {
    if (element_type)
      *element_type = array->GetElementType();
    return true;
  }
  return false;
}

bool JavaASTContext::IsAggregateType(lldb::opaque_compiler_type_t type) {
  return llvm::isa<JavaObjectType>(static_cast<JavaType *>(type));
}

bool JavaASTContext::IsCharType(lldb::opaque_compiler_type_t type) {
  if (JavaPrimitiveType *ptype =
          llvm::dyn_cast<JavaPrimitiveType>(static_cast<JavaType *>(type)))
    return ptype->GetTypeKind() == JavaPrimitiveType::eTypeChar;
  return false;
}

bool JavaASTContext::IsFloatingPointType(lldb::opaque_compiler_type_t type,
                                         uint32_t &count, bool &is_complex) {
  is_complex = true;

  if (JavaPrimitiveType *ptype =
          llvm::dyn_cast<JavaPrimitiveType>(static_cast<JavaType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case JavaPrimitiveType::eTypeFloat:
    case JavaPrimitiveType::eTypeDouble:
      count = 1;
      return true;
    default:
      break;
    }
  }

  count = 0;
  return false;
}

bool JavaASTContext::IsFunctionType(lldb::opaque_compiler_type_t type,
                                    bool *is_variadic_ptr) {
  if (is_variadic_ptr)
    *is_variadic_ptr = false;
  return false;
}

size_t JavaASTContext::GetNumberOfFunctionArguments(
    lldb::opaque_compiler_type_t type) {
  return 0;
}

CompilerType
JavaASTContext::GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                                           const size_t index) {
  return CompilerType();
}

bool JavaASTContext::IsFunctionPointerType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool JavaASTContext::IsBlockPointerType(
    lldb::opaque_compiler_type_t type,
    CompilerType *function_pointer_type_ptr) {
  return false;
}

bool JavaASTContext::IsIntegerType(lldb::opaque_compiler_type_t type,
                                   bool &is_signed) {
  if (JavaPrimitiveType *ptype =
          llvm::dyn_cast<JavaPrimitiveType>(static_cast<JavaType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case JavaPrimitiveType::eTypeByte:
    case JavaPrimitiveType::eTypeShort:
    case JavaPrimitiveType::eTypeInt:
    case JavaPrimitiveType::eTypeLong:
      is_signed = true;
      return true;
    default:
      break;
    }
  }

  is_signed = false;
  return false;
}

bool JavaASTContext::IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                                           CompilerType *target_type,
                                           bool check_cplusplus,
                                           bool check_objc, bool check_swift) {
  return llvm::isa<JavaReferenceType>(static_cast<JavaType *>(type));
}

bool JavaASTContext::IsPointerType(lldb::opaque_compiler_type_t type,
                                   CompilerType *pointee_type) {
  if (pointee_type)
    pointee_type->Clear();
  return false;
}

bool JavaASTContext::IsReferenceType(lldb::opaque_compiler_type_t type,
                                     CompilerType *pointee_type,
                                     bool *is_rvalue) {
  if (is_rvalue)
    *is_rvalue = false;

  if (JavaReferenceType *ref =
          llvm::dyn_cast<JavaReferenceType>(static_cast<JavaType *>(type))) {
    if (pointee_type)
      *pointee_type = ref->GetPointeeType();
    return true;
  }

  if (pointee_type)
    pointee_type->Clear();
  return false;
}

bool JavaASTContext::IsScalarType(lldb::opaque_compiler_type_t type) {
  return llvm::isa<JavaReferenceType>(static_cast<JavaType *>(type)) ||
         llvm::isa<JavaPrimitiveType>(static_cast<JavaType *>(type));
}

bool JavaASTContext::IsVoidType(lldb::opaque_compiler_type_t type) {
  return false; // TODO: Implement if we introduce the void type
}

bool JavaASTContext::SupportsLanguage(lldb::LanguageType language) {
  return language == lldb::eLanguageTypeJava;
}

bool JavaASTContext::IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) {
  return true;
}

bool JavaASTContext::IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                              CompilerType *pointee_type) {
  return IsPointerType(type, pointee_type) ||
         IsReferenceType(type, pointee_type);
}

bool JavaASTContext::IsCStringType(lldb::opaque_compiler_type_t type,
                                   uint32_t &length) {
  return false; // TODO: Implement it if we need it for string literals
}

bool JavaASTContext::IsTypedefType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool JavaASTContext::IsVectorType(lldb::opaque_compiler_type_t type,
                                  CompilerType *element_type, uint64_t *size) {
  if (element_type)
    element_type->Clear();
  if (size)
    *size = 0;
  return false;
}

bool JavaASTContext::IsPolymorphicClass(lldb::opaque_compiler_type_t type) {
  return llvm::isa<JavaObjectType>(static_cast<JavaType *>(type));
}

uint32_t
JavaASTContext::IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                                       CompilerType *base_type_ptr) {
  return false;
}

bool JavaASTContext::IsCompleteType(lldb::opaque_compiler_type_t type) {
  return static_cast<JavaType *>(type)->IsCompleteType();
}

bool JavaASTContext::IsConst(lldb::opaque_compiler_type_t type) {
  return false;
}

bool JavaASTContext::IsBeingDefined(lldb::opaque_compiler_type_t type) {
  return false;
}

bool JavaASTContext::IsDefined(lldb::opaque_compiler_type_t type) {
  return type != nullptr;
}

bool JavaASTContext::GetCompleteType(lldb::opaque_compiler_type_t type) {
  if (IsCompleteType(type))
    return true;

  if (JavaArrayType *array =
          llvm::dyn_cast<JavaArrayType>(static_cast<JavaType *>(type)))
    return GetCompleteType(array->GetElementType().GetOpaqueQualType());

  if (JavaReferenceType *reference =
          llvm::dyn_cast<JavaReferenceType>(static_cast<JavaType *>(type)))
    return GetCompleteType(reference->GetPointeeType().GetOpaqueQualType());

  if (llvm::isa<JavaObjectType>(static_cast<JavaType *>(type))) {
    SymbolFile *symbol_file = GetSymbolFile();
    if (!symbol_file)
      return false;

    CompilerType object_type(this, type);
    return symbol_file->CompleteType(object_type);
  }
  return false;
}

ConstString JavaASTContext::GetTypeName(lldb::opaque_compiler_type_t type) {
  if (type)
    return static_cast<JavaType *>(type)->GetName();
  return ConstString();
}

uint32_t
JavaASTContext::GetTypeInfo(lldb::opaque_compiler_type_t type,
                            CompilerType *pointee_or_element_compiler_type) {
  if (pointee_or_element_compiler_type)
    pointee_or_element_compiler_type->Clear();
  if (!type)
    return 0;

  if (IsReferenceType(type, pointee_or_element_compiler_type))
    return eTypeHasChildren | eTypeHasValue | eTypeIsReference;
  if (IsArrayType(type, pointee_or_element_compiler_type, nullptr, nullptr))
    return eTypeHasChildren | eTypeIsArray;
  if (llvm::isa<JavaObjectType>(static_cast<JavaType *>(type)))
    return eTypeHasChildren | eTypeIsClass;

  if (JavaPrimitiveType *ptype =
          llvm::dyn_cast<JavaPrimitiveType>(static_cast<JavaType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case JavaPrimitiveType::eTypeByte:
    case JavaPrimitiveType::eTypeShort:
    case JavaPrimitiveType::eTypeInt:
    case JavaPrimitiveType::eTypeLong:
      return eTypeHasValue | eTypeIsBuiltIn | eTypeIsScalar | eTypeIsInteger |
             eTypeIsSigned;
    case JavaPrimitiveType::eTypeFloat:
    case JavaPrimitiveType::eTypeDouble:
      return eTypeHasValue | eTypeIsBuiltIn | eTypeIsScalar | eTypeIsFloat |
             eTypeIsSigned;
    case JavaPrimitiveType::eTypeBoolean:
      return eTypeHasValue | eTypeIsBuiltIn | eTypeIsScalar;
    case JavaPrimitiveType::eTypeChar:
      return eTypeHasValue | eTypeIsBuiltIn | eTypeIsScalar;
    }
  }
  return 0;
}

lldb::TypeClass
JavaASTContext::GetTypeClass(lldb::opaque_compiler_type_t type) {
  if (!type)
    return eTypeClassInvalid;
  if (llvm::isa<JavaReferenceType>(static_cast<JavaType *>(type)))
    return eTypeClassReference;
  if (llvm::isa<JavaArrayType>(static_cast<JavaType *>(type)))
    return eTypeClassArray;
  if (llvm::isa<JavaObjectType>(static_cast<JavaType *>(type)))
    return eTypeClassClass;
  if (llvm::isa<JavaPrimitiveType>(static_cast<JavaType *>(type)))
    return eTypeClassBuiltin;
  assert(false && "Java type with unhandled type class");
  return eTypeClassInvalid;
}

lldb::LanguageType
JavaASTContext::GetMinimumLanguage(lldb::opaque_compiler_type_t type) {
  return lldb::eLanguageTypeJava;
}

CompilerType
JavaASTContext::GetArrayElementType(lldb::opaque_compiler_type_t type,
                                    uint64_t *stride) {
  if (stride)
    *stride = 0;

  CompilerType element_type;
  if (IsArrayType(type, &element_type, nullptr, nullptr))
    return element_type;
  return CompilerType();
}

CompilerType JavaASTContext::GetPointeeType(lldb::opaque_compiler_type_t type) {
  CompilerType pointee_type;
  if (IsPointerType(type, &pointee_type))
    return pointee_type;
  return CompilerType();
}

CompilerType JavaASTContext::GetPointerType(lldb::opaque_compiler_type_t type) {
  return CompilerType(); // No pointer types in java
}

CompilerType
JavaASTContext::GetCanonicalType(lldb::opaque_compiler_type_t type) {
  return CompilerType(this, type);
}

CompilerType
JavaASTContext::GetInstanceType(lldb::opaque_compiler_type_t type) {
  return CompilerType(this, type);
}

CompilerType
JavaASTContext::GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) {
  return CompilerType(this, type);
}

CompilerType
JavaASTContext::GetNonReferenceType(lldb::opaque_compiler_type_t type) {
  CompilerType pointee_type;
  if (IsReferenceType(type, &pointee_type))
    return pointee_type;
  return CompilerType(this, type);
}

CompilerType
JavaASTContext::GetTypedefedType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType JavaASTContext::GetUnboundType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType JavaASTContext::GetBasicTypeFromAST(lldb::BasicType basic_type) {
  return CompilerType();
}

CompilerType
JavaASTContext::GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                    size_t bit_size) {
  return CompilerType();
}

size_t JavaASTContext::GetTypeBitAlign(lldb::opaque_compiler_type_t type) {
  return 0;
}

lldb::BasicType
JavaASTContext::GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) {
  if (JavaPrimitiveType *ptype =
          llvm::dyn_cast<JavaPrimitiveType>(static_cast<JavaType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case JavaPrimitiveType::eTypeByte:
      return eBasicTypeOther;
    case JavaPrimitiveType::eTypeShort:
      return eBasicTypeShort;
    case JavaPrimitiveType::eTypeInt:
      return eBasicTypeInt;
    case JavaPrimitiveType::eTypeLong:
      return eBasicTypeLong;
    case JavaPrimitiveType::eTypeFloat:
      return eBasicTypeFloat;
    case JavaPrimitiveType::eTypeDouble:
      return eBasicTypeDouble;
    case JavaPrimitiveType::eTypeBoolean:
      return eBasicTypeBool;
    case JavaPrimitiveType::eTypeChar:
      return eBasicTypeChar;
    }
  }
  return eBasicTypeInvalid;
}

uint64_t JavaASTContext::GetBitSize(lldb::opaque_compiler_type_t type,
                                    ExecutionContextScope *exe_scope) {
  if (JavaPrimitiveType *ptype =
          llvm::dyn_cast<JavaPrimitiveType>(static_cast<JavaType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case JavaPrimitiveType::eTypeByte:
      return 8;
    case JavaPrimitiveType::eTypeShort:
      return 16;
    case JavaPrimitiveType::eTypeInt:
      return 32;
    case JavaPrimitiveType::eTypeLong:
      return 64;
    case JavaPrimitiveType::eTypeFloat:
      return 32;
    case JavaPrimitiveType::eTypeDouble:
      return 64;
    case JavaPrimitiveType::eTypeBoolean:
      return 1;
    case JavaPrimitiveType::eTypeChar:
      return 16;
    }
  } else if (llvm::isa<JavaReferenceType>(static_cast<JavaType *>(type))) {
    return 32; // References are always 4 byte long in java
  } else if (llvm::isa<JavaArrayType>(static_cast<JavaType *>(type))) {
    return 64;
  } else if (JavaObjectType *obj = llvm::dyn_cast<JavaObjectType>(
                 static_cast<JavaType *>(type))) {
    return obj->GetByteSize() * 8;
  }
  return 0;
}

uint64_t JavaASTContext::GetByteStride(void *type) {
  return (GetBitSize(type, nullptr) + 7) / 8;
}

lldb::Encoding JavaASTContext::GetEncoding(lldb::opaque_compiler_type_t type,
                                           uint64_t &count) {
  count = 1;

  if (JavaPrimitiveType *ptype =
          llvm::dyn_cast<JavaPrimitiveType>(static_cast<JavaType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case JavaPrimitiveType::eTypeByte:
    case JavaPrimitiveType::eTypeShort:
    case JavaPrimitiveType::eTypeInt:
    case JavaPrimitiveType::eTypeLong:
      return eEncodingSint;
    case JavaPrimitiveType::eTypeFloat:
    case JavaPrimitiveType::eTypeDouble:
      return eEncodingIEEE754;
    case JavaPrimitiveType::eTypeBoolean:
    case JavaPrimitiveType::eTypeChar:
      return eEncodingUint;
    }
  }
  if (IsReferenceType(type))
    return eEncodingUint;
  return eEncodingInvalid;
}

lldb::Format JavaASTContext::GetFormat(lldb::opaque_compiler_type_t type) {
  if (JavaPrimitiveType *ptype =
          llvm::dyn_cast<JavaPrimitiveType>(static_cast<JavaType *>(type))) {
    switch (ptype->GetTypeKind()) {
    case JavaPrimitiveType::eTypeByte:
    case JavaPrimitiveType::eTypeShort:
    case JavaPrimitiveType::eTypeInt:
    case JavaPrimitiveType::eTypeLong:
      return eFormatDecimal;
    case JavaPrimitiveType::eTypeFloat:
    case JavaPrimitiveType::eTypeDouble:
      return eFormatFloat;
    case JavaPrimitiveType::eTypeBoolean:
      return eFormatBoolean;
    case JavaPrimitiveType::eTypeChar:
      return eFormatUnicode16;
    }
  }
  if (IsReferenceType(type))
    return eFormatHex;
  return eFormatDefault;
}

unsigned JavaASTContext::GetTypeQualifiers(lldb::opaque_compiler_type_t type) {
  return 0;
}

size_t
JavaASTContext::ConvertStringToFloatValue(lldb::opaque_compiler_type_t type,
                                          const char *s, uint8_t *dst,
                                          size_t dst_size) {
  assert(false && "Not implemented");
  return 0;
}

size_t
JavaASTContext::GetNumTemplateArguments(lldb::opaque_compiler_type_t type) {
  return 0;
}

CompilerType
JavaASTContext::GetTemplateArgument(lldb::opaque_compiler_type_t type,
                                    size_t idx,
                                    lldb::TemplateArgumentKind &kind) {
  return CompilerType();
}

uint32_t JavaASTContext::GetNumFields(lldb::opaque_compiler_type_t type) {
  if (JavaObjectType *obj =
          llvm::dyn_cast<JavaObjectType>(static_cast<JavaType *>(type))) {
    GetCompleteType(type);
    return obj->GetNumFields();
  }
  return 0;
}

CompilerType JavaASTContext::GetFieldAtIndex(lldb::opaque_compiler_type_t type,
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

  if (JavaObjectType *obj =
          llvm::dyn_cast<JavaObjectType>(static_cast<JavaType *>(type))) {
    GetCompleteType(type);

    JavaObjectType::Field *field = obj->GetFieldAtIndex(idx);
    if (!field)
      return CompilerType();
    name = field->m_name.AsCString();
    if (bit_offset_ptr)
      *bit_offset_ptr = field->m_offset * 8;
    return field->m_type;
  }
  return CompilerType();
}

uint32_t JavaASTContext::GetNumChildren(lldb::opaque_compiler_type_t type,
                                        bool omit_empty_base_classes) {
  GetCompleteType(type);

  if (JavaReferenceType *ref =
          llvm::dyn_cast<JavaReferenceType>(static_cast<JavaType *>(type)))
    return ref->GetPointeeType().GetNumChildren(omit_empty_base_classes);

  if (llvm::isa<JavaObjectType>(static_cast<JavaType *>(type)))
    return GetNumFields(type) + GetNumDirectBaseClasses(type);

  return 0;
}

uint32_t
JavaASTContext::GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) {
  if (JavaObjectType *obj =
          llvm::dyn_cast<JavaObjectType>(static_cast<JavaType *>(type))) {
    GetCompleteType(type);
    return obj->GetNumInterfaces() + (obj->GetBaseClass() ? 1 : 0);
  }
  return 0;
}

uint32_t
JavaASTContext::GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) {
  if (JavaObjectType *obj =
          llvm::dyn_cast<JavaObjectType>(static_cast<JavaType *>(type))) {
    GetCompleteType(type);
    return obj->GetNumInterfaces();
  }
  return 0;
}

CompilerType JavaASTContext::GetDirectBaseClassAtIndex(
    lldb::opaque_compiler_type_t type, size_t idx, uint32_t *bit_offset_ptr) {
  if (JavaObjectType *obj =
          llvm::dyn_cast<JavaObjectType>(static_cast<JavaType *>(type))) {
    GetCompleteType(type);

    if (CompilerType base_class = obj->GetBaseClass()) {
      if (idx == 0)
        return base_class;
      else
        --idx;
    }
    return obj->GetInterfaceAtIndex(idx);
  }
  return CompilerType();
}

CompilerType JavaASTContext::GetVirtualBaseClassAtIndex(
    lldb::opaque_compiler_type_t type, size_t idx, uint32_t *bit_offset_ptr) {
  if (JavaObjectType *obj =
          llvm::dyn_cast<JavaObjectType>(static_cast<JavaType *>(type))) {
    GetCompleteType(type);
    return obj->GetInterfaceAtIndex(idx);
  }
  return CompilerType();
}

void JavaASTContext::DumpValue(
    lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, Stream *s,
    lldb::Format format, const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, bool show_types, bool show_summary,
    bool verbose, uint32_t depth) {
  assert(false && "Not implemented");
}

bool JavaASTContext::DumpTypeValue(
    lldb::opaque_compiler_type_t type, Stream *s, lldb::Format format,
    const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, ExecutionContextScope *exe_scope,
    bool is_base_class) {
  if (IsScalarType(type)) {
    return data.Dump(s, data_offset, format, data_byte_size,
                     1, // count
                     UINT32_MAX, LLDB_INVALID_ADDRESS, bitfield_bit_size,
                     bitfield_bit_offset, exe_scope);
  }
  return false;
}

void JavaASTContext::DumpTypeDescription(lldb::opaque_compiler_type_t type) {
  StreamFile s(stdout, false);
  DumpTypeDescription(type, &s);
}

void JavaASTContext::DumpTypeDescription(lldb::opaque_compiler_type_t type,
                                         Stream *s) {
  static_cast<JavaType *>(type)->Dump(s);
}

void JavaASTContext::DumpSummary(lldb::opaque_compiler_type_t type,
                                 ExecutionContext *exe_ctx, Stream *s,
                                 const DataExtractor &data,
                                 lldb::offset_t data_offset,
                                 size_t data_byte_size) {
  assert(false && "Not implemented");
}

int JavaASTContext::GetFunctionArgumentCount(
    lldb::opaque_compiler_type_t type) {
  return 0;
}

CompilerType JavaASTContext::GetFunctionArgumentTypeAtIndex(
    lldb::opaque_compiler_type_t type, size_t idx) {
  return CompilerType();
}

CompilerType
JavaASTContext::GetFunctionReturnType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

size_t
JavaASTContext::GetNumMemberFunctions(lldb::opaque_compiler_type_t type) {
  return 0;
}

TypeMemberFunctionImpl
JavaASTContext::GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                                         size_t idx) {
  return TypeMemberFunctionImpl();
}

CompilerType JavaASTContext::GetChildCompilerTypeAtIndex(
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

  ExecutionContextScope *exec_ctx_scope =
      exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr;

  if (JavaObjectType *obj =
          llvm::dyn_cast<JavaObjectType>(static_cast<JavaType *>(type))) {
    GetCompleteType(type);

    if (CompilerType base_class = obj->GetBaseClass()) {
      if (idx == 0) {
        JavaType *base_class_type =
            static_cast<JavaType *>(base_class.GetOpaqueQualType());
        child_name = base_class_type->GetName().GetCString();
        child_byte_size = base_class.GetByteSize(
            exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr);
        child_byte_offset = obj->GetBaseClassOffset();
        child_is_base_class = true;
        return base_class;
      }
      idx -= 1;
    }

    JavaObjectType::Field *field = obj->GetFieldAtIndex(idx);
    if (!field)
      return CompilerType();

    child_name = field->m_name.AsCString();
    child_byte_size = field->m_type.GetByteSize(exec_ctx_scope);
    child_byte_offset = field->m_offset;
    return field->m_type;
  } else if (JavaReferenceType *ref = llvm::dyn_cast<JavaReferenceType>(
                 static_cast<JavaType *>(type))) {
    CompilerType pointee_type = ref->GetPointeeType();

    if (transparent_pointers)
      return pointee_type.GetChildCompilerTypeAtIndex(
          exe_ctx, idx, transparent_pointers, omit_empty_base_classes,
          ignore_array_bounds, child_name, child_byte_size, child_byte_offset,
          child_bitfield_bit_size, child_bitfield_bit_offset,
          child_is_base_class, child_is_deref_of_parent, valobj,
          language_flags);

    if (idx != 0)
      return CompilerType();

    if (valobj && valobj->GetName())
      child_name = valobj->GetName().GetCString();
    child_is_deref_of_parent = true;
    child_byte_offset = 0;
    child_byte_size = pointee_type.GetByteSize(exec_ctx_scope);
    return pointee_type;
  }
  return CompilerType();
}

uint32_t
JavaASTContext::GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                                        const char *name,
                                        bool omit_empty_base_classes) {
  if (JavaObjectType *obj =
          llvm::dyn_cast<JavaObjectType>(static_cast<JavaType *>(type))) {
    GetCompleteType(type);

    uint32_t index_offset = 0;
    if (CompilerType base_class = obj->GetBaseClass()) {
      if (base_class.GetTypeName() == ConstString(name))
        return 0;
      index_offset = 1;
    }
    for (uint32_t i = 0; i < obj->GetNumFields(); ++i) {
      if (obj->GetFieldAtIndex(i)->m_name == ConstString(name))
        return i + index_offset;
    }
  } else if (JavaReferenceType *ref = llvm::dyn_cast<JavaReferenceType>(
                 static_cast<JavaType *>(type))) {
    return GetIndexOfChildWithName(ref->GetPointeeType().GetOpaqueQualType(),
                                   name, omit_empty_base_classes);
  }
  return UINT_MAX;
}

size_t JavaASTContext::GetIndexOfChildMemberWithName(
    lldb::opaque_compiler_type_t type, const char *name,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  child_indexes.clear();

  if (JavaObjectType *obj =
          llvm::dyn_cast<JavaObjectType>(static_cast<JavaType *>(type))) {
    GetCompleteType(type);

    uint32_t index_offset = 0;
    if (CompilerType base_class = obj->GetBaseClass()) {
      if (GetIndexOfChildMemberWithName(base_class.GetOpaqueQualType(), name,
                                        omit_empty_base_classes,
                                        child_indexes) != 0) {
        child_indexes.insert(child_indexes.begin(), 0);
        return child_indexes.size();
      }
      index_offset = 1;
    }

    for (uint32_t i = 0; i < obj->GetNumFields(); ++i) {
      if (obj->GetFieldAtIndex(i)->m_name == ConstString(name)) {
        child_indexes.push_back(i + index_offset);
        return child_indexes.size();
      }
    }
  } else if (JavaReferenceType *ref = llvm::dyn_cast<JavaReferenceType>(
                 static_cast<JavaType *>(type))) {
    return GetIndexOfChildMemberWithName(
        ref->GetPointeeType().GetOpaqueQualType(), name,
        omit_empty_base_classes, child_indexes);
  }
  return 0;
}

CompilerType
JavaASTContext::GetLValueReferenceType(lldb::opaque_compiler_type_t type) {
  return CreateReferenceType(CompilerType(this, type));
}

ConstString JavaASTContext::DeclContextGetScopeQualifiedName(
    lldb::opaque_compiler_type_t opaque_decl_ctx) {
  return GetTypeName(opaque_decl_ctx);
}

static void AddPrimitiveType(JavaASTContext::JavaTypeMap &type_map,
                             JavaPrimitiveType::TypeKind type_kind) {
  JavaPrimitiveType *type = new JavaPrimitiveType(type_kind);
  type_map.emplace(type->GetName(),
                   std::unique_ptr<JavaASTContext::JavaType>(type));
}

CompilerType JavaASTContext::CreateBaseType(const ConstString &name) {
  if (m_base_type_map.empty()) {
    AddPrimitiveType(m_base_type_map, JavaPrimitiveType::eTypeByte);
    AddPrimitiveType(m_base_type_map, JavaPrimitiveType::eTypeShort);
    AddPrimitiveType(m_base_type_map, JavaPrimitiveType::eTypeInt);
    AddPrimitiveType(m_base_type_map, JavaPrimitiveType::eTypeLong);
    AddPrimitiveType(m_base_type_map, JavaPrimitiveType::eTypeFloat);
    AddPrimitiveType(m_base_type_map, JavaPrimitiveType::eTypeDouble);
    AddPrimitiveType(m_base_type_map, JavaPrimitiveType::eTypeBoolean);
    AddPrimitiveType(m_base_type_map, JavaPrimitiveType::eTypeChar);
  }
  auto it = m_base_type_map.find(name);
  if (it != m_base_type_map.end())
    return CompilerType(this, it->second.get());
  return CompilerType();
}

CompilerType JavaASTContext::CreateObjectType(const ConstString &name,
                                              const ConstString &linkage_name,
                                              uint32_t byte_size) {
  auto it = m_object_type_map.find(name);
  if (it == m_object_type_map.end()) {
    std::unique_ptr<JavaType> object_type(
        new JavaObjectType(name, linkage_name, byte_size));
    it = m_object_type_map.emplace(name, std::move(object_type)).first;
  }
  return CompilerType(this, it->second.get());
}

CompilerType JavaASTContext::CreateArrayType(
    const ConstString &linkage_name, const CompilerType &element_type,
    const DWARFExpression &length_expression, const lldb::addr_t data_offset) {
  ConstString name = element_type.GetTypeName();
  auto it = m_array_type_map.find(name);
  if (it == m_array_type_map.end()) {
    std::unique_ptr<JavaType> array_type(new JavaArrayType(
        linkage_name, element_type, length_expression, data_offset));
    it = m_array_type_map.emplace(name, std::move(array_type)).first;
  }
  return CompilerType(this, it->second.get());
}

CompilerType
JavaASTContext::CreateReferenceType(const CompilerType &pointee_type) {
  ConstString name = pointee_type.GetTypeName();
  auto it = m_reference_type_map.find(name);
  if (it == m_reference_type_map.end())
    it = m_reference_type_map
             .emplace(name, std::unique_ptr<JavaType>(
                                new JavaReferenceType(pointee_type)))
             .first;
  return CompilerType(this, it->second.get());
}

void JavaASTContext::CompleteObjectType(const CompilerType &object_type) {
  JavaObjectType *obj = llvm::dyn_cast<JavaObjectType>(
      static_cast<JavaType *>(object_type.GetOpaqueQualType()));
  assert(obj &&
         "JavaASTContext::CompleteObjectType called with not a JavaObjectType");
  obj->SetCompleteType(true);
}

void JavaASTContext::AddBaseClassToObject(const CompilerType &object_type,
                                          const CompilerType &member_type,
                                          uint32_t member_offset) {
  JavaObjectType *obj = llvm::dyn_cast<JavaObjectType>(
      static_cast<JavaType *>(object_type.GetOpaqueQualType()));
  assert(obj &&
         "JavaASTContext::AddMemberToObject called with not a JavaObjectType");
  obj->AddBaseClass(member_type, member_offset);
}

void JavaASTContext::AddMemberToObject(const CompilerType &object_type,
                                       const ConstString &name,
                                       const CompilerType &member_type,
                                       uint32_t member_offset) {
  JavaObjectType *obj = llvm::dyn_cast<JavaObjectType>(
      static_cast<JavaType *>(object_type.GetOpaqueQualType()));
  assert(obj &&
         "JavaASTContext::AddMemberToObject called with not a JavaObjectType");
  obj->AddField(name, member_type, member_offset);
}

void JavaASTContext::SetDynamicTypeId(const CompilerType &type,
                                      const DWARFExpression &type_id) {
  JavaObjectType *obj = llvm::dyn_cast<JavaObjectType>(
      static_cast<JavaType *>(type.GetOpaqueQualType()));
  assert(obj &&
         "JavaASTContext::SetDynamicTypeId called with not a JavaObjectType");
  obj->SetDynamicTypeId(type_id);
}

uint64_t JavaASTContext::CalculateDynamicTypeId(ExecutionContext *exe_ctx,
                                                const CompilerType &type,
                                                ValueObject &in_value) {
  if (JavaObjectType *obj = llvm::dyn_cast<JavaObjectType>(
          static_cast<JavaType *>(type.GetOpaqueQualType())))
    return obj->CalculateDynamicTypeId(exe_ctx, in_value);
  if (JavaArrayType *arr = llvm::dyn_cast<JavaArrayType>(
          static_cast<JavaType *>(type.GetOpaqueQualType())))
    return arr->CalculateDynamicTypeId(exe_ctx, in_value);
  return UINT64_MAX;
}

uint32_t JavaASTContext::CalculateArraySize(const CompilerType &type,
                                            ValueObject &in_value) {
  if (JavaArrayType *arr = llvm::dyn_cast<JavaArrayType>(
          static_cast<JavaType *>(type.GetOpaqueQualType())))
    return arr->GetNumElements(&in_value);
  return UINT32_MAX;
}

uint64_t JavaASTContext::CalculateArrayElementOffset(const CompilerType &type,
                                                     size_t index) {
  if (JavaArrayType *arr = llvm::dyn_cast<JavaArrayType>(
          static_cast<JavaType *>(type.GetOpaqueQualType())))
    return arr->GetElementOffset(index);
  return UINT64_MAX;
}

ConstString JavaASTContext::GetLinkageName(const CompilerType &type) {
  if (JavaObjectType *obj = llvm::dyn_cast<JavaObjectType>(
          static_cast<JavaType *>(type.GetOpaqueQualType())))
    return obj->GetLinkageName();
  return ConstString();
}
