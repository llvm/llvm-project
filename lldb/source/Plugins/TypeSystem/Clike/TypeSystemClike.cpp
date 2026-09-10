//===-- TypeSystemClike.cpp
//-------------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeSystemClike.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Type.h"
#include "llvm/ADT/APFloat.h"

#include <optional>

using namespace lldb_private;

LLDB_PLUGIN_DEFINE(TypeSystemClike)

char TypeSystemClike::ID;

TypeSystemClike::TypeSystemClike() = default;

TypeSystemClike::~TypeSystemClike() = default;

lldb::TypeSystemSP TypeSystemClike::CreateInstance(lldb::LanguageType language,
                                                   Module *module,
                                                   Target *target) {
  return lldb::TypeSystemSP();
}

LanguageSet TypeSystemClike::GetSupportedLanguagesForTypes() {
  return LanguageSet();
}

LanguageSet TypeSystemClike::GetSupportedLanguagesForExpressions() {
  return LanguageSet();
}

void TypeSystemClike::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "C/C++/Objective-C++ TypeSystem plug-in",
                                CreateInstance, GetSupportedLanguagesForTypes(),
                                GetSupportedLanguagesForExpressions());
}

void TypeSystemClike::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ConstString TypeSystemClike::DeclGetName(void *opaque_decl) {
  return ConstString();
}

CompilerType TypeSystemClike::GetTypeForDecl(void *opaque_decl) {
  return CompilerType();
}

ConstString TypeSystemClike::DeclContextGetName(void *opaque_decl_ctx) {
  return ConstString();
}

ConstString
TypeSystemClike::DeclContextGetScopeQualifiedName(void *opaque_decl_ctx) {
  return ConstString();
}

bool TypeSystemClike::DeclContextIsClassMethod(void *opaque_decl_ctx) {
  return false;
}

bool TypeSystemClike::DeclContextIsContainedInLookup(
    void *opaque_decl_ctx, void *other_opaque_decl_ctx) {
  return false;
}

lldb::LanguageType
TypeSystemClike::DeclContextGetLanguage(void *opaque_decl_ctx) {
  return lldb::eLanguageTypeUnknown;
}

bool TypeSystemClike::Verify(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsArrayType(lldb::opaque_compiler_type_t type,
                                  CompilerType *element_type, uint64_t *size,
                                  bool *is_incomplete) {
  return false;
}

bool TypeSystemClike::IsAggregateType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsCharType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsCompleteType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsDefined(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsFloatingPointType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsFunctionType(lldb::opaque_compiler_type_t type) {
  return false;
}

size_t TypeSystemClike::GetNumberOfFunctionArguments(
    lldb::opaque_compiler_type_t type) {
  return 0;
}

CompilerType
TypeSystemClike::GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                                            const size_t index) {
  return CompilerType();
}

bool TypeSystemClike::IsFunctionPointerType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsMemberFunctionPointerType(
    lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsMemberDataPointerType(
    lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsBlockPointerType(
    lldb::opaque_compiler_type_t type,
    CompilerType *function_pointer_type_ptr) {
  return false;
}

bool TypeSystemClike::IsIntegerType(lldb::opaque_compiler_type_t type,
                                    bool &is_signed) {
  return false;
}

bool TypeSystemClike::IsScopedEnumerationType(
    lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                                            CompilerType *target_type,
                                            bool check_cplusplus,
                                            bool check_objc) {
  return false;
}

bool TypeSystemClike::IsPointerType(lldb::opaque_compiler_type_t type,
                                    CompilerType *pointee_type) {
  return false;
}

bool TypeSystemClike::IsScalarType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsVoidType(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::CanPassInRegisters(const CompilerType &type) {
  return false;
}

bool TypeSystemClike::SupportsLanguage(lldb::LanguageType language) {
  return false;
}

bool TypeSystemClike::GetCompleteType(lldb::opaque_compiler_type_t type) {
  return false;
}

uint32_t TypeSystemClike::GetPointerByteSize() { return 0; }

CompilerType TypeSystemClike::GetPointerDiffType(bool is_signed) {
  return CompilerType();
}

CompilerType TypeSystemClike::GetSizeType() { return CompilerType(); }

unsigned TypeSystemClike::GetPtrAuthKey(lldb::opaque_compiler_type_t type) {
  return 0;
}

unsigned
TypeSystemClike::GetPtrAuthDiscriminator(lldb::opaque_compiler_type_t type) {
  return 0;
}

bool TypeSystemClike::GetPtrAuthAddressDiversity(
    lldb::opaque_compiler_type_t type) {
  return false;
}

ConstString TypeSystemClike::GetTypeName(lldb::opaque_compiler_type_t type,
                                         bool BaseOnly) {
  return ConstString();
}

ConstString
TypeSystemClike::GetDisplayTypeName(lldb::opaque_compiler_type_t type) {
  return ConstString();
}

uint32_t
TypeSystemClike::GetTypeInfo(lldb::opaque_compiler_type_t type,
                             CompilerType *pointee_or_element_compiler_type) {
  return 0;
}

lldb::LanguageType
TypeSystemClike::GetMinimumLanguage(lldb::opaque_compiler_type_t type) {
  return lldb::eLanguageTypeUnknown;
}

lldb::TypeClass
TypeSystemClike::GetTypeClass(lldb::opaque_compiler_type_t type) {
  return lldb::eTypeClassInvalid;
}

CompilerType
TypeSystemClike::GetArrayElementType(lldb::opaque_compiler_type_t type,
                                     ExecutionContextScope *exe_scope) {
  return CompilerType();
}

CompilerType
TypeSystemClike::GetCanonicalType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType
TypeSystemClike::GetEnumerationIntegerType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

int TypeSystemClike::GetFunctionArgumentCount(
    lldb::opaque_compiler_type_t type) {
  return 0;
}

CompilerType TypeSystemClike::GetFunctionArgumentTypeAtIndex(
    lldb::opaque_compiler_type_t type, size_t idx) {
  return CompilerType();
}

CompilerType
TypeSystemClike::GetFunctionReturnType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

size_t
TypeSystemClike::GetNumMemberFunctions(lldb::opaque_compiler_type_t type) {
  return 0;
}

TypeMemberFunctionImpl
TypeSystemClike::GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                                          size_t idx) {
  return TypeMemberFunctionImpl();
}

CompilerType
TypeSystemClike::GetPointeeType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType
TypeSystemClike::GetPointerType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

const llvm::fltSemantics &
TypeSystemClike::GetFloatTypeSemantics(size_t byte_size, lldb::Format format) {
  return llvm::APFloat::Bogus();
}

llvm::Expected<uint64_t>
TypeSystemClike::GetBitSize(lldb::opaque_compiler_type_t type,
                            ExecutionContextScope *exe_scope) {
  return 0;
}

lldb::Encoding TypeSystemClike::GetEncoding(lldb::opaque_compiler_type_t type) {
  return lldb::eEncodingInvalid;
}

lldb::Format TypeSystemClike::GetFormat(lldb::opaque_compiler_type_t type) {
  return lldb::eFormatDefault;
}

llvm::Expected<uint32_t>
TypeSystemClike::GetNumChildren(lldb::opaque_compiler_type_t type,
                                bool omit_empty_base_classes,
                                const ExecutionContext *exe_ctx) {
  return 0;
}

lldb::BasicType
TypeSystemClike::GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) {
  return lldb::eBasicTypeInvalid;
}

uint32_t TypeSystemClike::GetNumFields(lldb::opaque_compiler_type_t type) {
  return 0;
}

CompilerType TypeSystemClike::GetFieldAtIndex(lldb::opaque_compiler_type_t type,
                                              size_t idx, std::string &name,
                                              uint64_t *bit_offset_ptr,
                                              uint32_t *bitfield_bit_size_ptr,
                                              bool *is_bitfield_ptr) {
  return CompilerType();
}

uint32_t
TypeSystemClike::GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) {
  return 0;
}

uint32_t
TypeSystemClike::GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) {
  return 0;
}

CompilerType TypeSystemClike::GetDirectBaseClassAtIndex(
    lldb::opaque_compiler_type_t type, size_t idx, uint32_t *bit_offset_ptr) {
  return CompilerType();
}

CompilerType TypeSystemClike::GetVirtualBaseClassAtIndex(
    lldb::opaque_compiler_type_t type, size_t idx, uint32_t *bit_offset_ptr) {
  return CompilerType();
}

llvm::Expected<CompilerType> TypeSystemClike::GetDereferencedType(
    lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx,
    std::string &deref_name, uint32_t &deref_byte_size,
    int32_t &deref_byte_offset, ValueObject *valobj, uint64_t &language_flags) {
  return CompilerType();
}

llvm::Expected<CompilerType> TypeSystemClike::GetChildCompilerTypeAtIndex(
    lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
    bool transparent_pointers, bool omit_empty_base_classes,
    bool ignore_array_bounds, std::string &child_name,
    uint32_t &child_byte_size, int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class, bool &child_is_deref_of_parent,
    ValueObject *valobj, uint64_t &language_flags) {
  return CompilerType();
}

llvm::Expected<uint32_t>
TypeSystemClike::GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                                         llvm::StringRef name,
                                         bool omit_empty_base_classes) {
  return 0;
}

size_t TypeSystemClike::GetIndexOfChildMemberWithName(
    lldb::opaque_compiler_type_t type, llvm::StringRef name,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  return 0;
}

bool TypeSystemClike::DumpTypeValue(
    lldb::opaque_compiler_type_t type, Stream &s, lldb::Format format,
    const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, ExecutionContextScope *exe_scope) {
  return false;
}

void TypeSystemClike::DumpTypeDescription(lldb::opaque_compiler_type_t type,
                                          lldb::DescriptionLevel level) {}

void TypeSystemClike::DumpTypeDescription(lldb::opaque_compiler_type_t type,
                                          Stream &s,
                                          lldb::DescriptionLevel level) {}

void TypeSystemClike::Dump(llvm::raw_ostream &output, llvm::StringRef filter,
                           bool show_color) {}

bool TypeSystemClike::IsRuntimeGeneratedType(
    lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsPointerOrReferenceType(
    lldb::opaque_compiler_type_t type, CompilerType *pointee_type) {
  return false;
}

unsigned TypeSystemClike::GetTypeQualifiers(lldb::opaque_compiler_type_t type) {
  return 0;
}

std::optional<size_t>
TypeSystemClike::GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                                 ExecutionContextScope *exe_scope) {
  return std::nullopt;
}

CompilerType TypeSystemClike::GetBasicTypeFromAST(lldb::BasicType basic_type) {
  return CompilerType();
}

CompilerType
TypeSystemClike::GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                     size_t bit_size) {
  return CompilerType();
}

bool TypeSystemClike::IsBeingDefined(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsConst(lldb::opaque_compiler_type_t type) {
  return false;
}

uint32_t
TypeSystemClike::IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                                        CompilerType *base_type_ptr) {
  return 0;
}

bool TypeSystemClike::IsPolymorphicClass(lldb::opaque_compiler_type_t type) {
  return false;
}

bool TypeSystemClike::IsTypedefType(lldb::opaque_compiler_type_t type) {
  return false;
}

CompilerType
TypeSystemClike::GetTypedefedType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

bool TypeSystemClike::IsVectorType(lldb::opaque_compiler_type_t type,
                                   CompilerType *element_type, uint64_t *size) {
  return false;
}

CompilerType
TypeSystemClike::GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

CompilerType
TypeSystemClike::GetNonReferenceType(lldb::opaque_compiler_type_t type) {
  return CompilerType();
}

bool TypeSystemClike::IsReferenceType(lldb::opaque_compiler_type_t type,
                                      CompilerType *pointee_type,
                                      bool *is_rvalue) {
  return false;
}
