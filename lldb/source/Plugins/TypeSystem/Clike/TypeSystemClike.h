//===-- TypeSystemClike.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_TYPESYSTEMCLIKE_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_TYPESYSTEMCLIKE_H

#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/TypeSystem.h"

namespace lldb_private {

/// A TypeSystem for Clike languages such as C, C++ and Objective-C.
class TypeSystemClike : public TypeSystem {
  // LLVM RTTI support
  static char ID;

public:
  TypeSystemClike();
  ~TypeSystemClike() override;

  // PluginInterface
  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() { return "cpp"; }
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  static lldb::TypeSystemSP CreateInstance(lldb::LanguageType language,
                                           Module *module, Target *target);
  static LanguageSet GetSupportedLanguagesForTypes();
  static LanguageSet GetSupportedLanguagesForExpressions();

  // LLVM RTTI support
  bool isA(const void *ClassID) const override { return ClassID == &ID; }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }

  ConstString DeclGetName(void *opaque_decl) override;
  CompilerType GetTypeForDecl(void *opaque_decl) override;
  ConstString DeclContextGetName(void *opaque_decl_ctx) override;
  ConstString DeclContextGetScopeQualifiedName(void *opaque_decl_ctx) override;
  bool DeclContextIsClassMethod(void *opaque_decl_ctx) override;
  bool DeclContextIsContainedInLookup(void *opaque_decl_ctx,
                                      void *other_opaque_decl_ctx) override;
  lldb::LanguageType DeclContextGetLanguage(void *opaque_decl_ctx) override;
  bool Verify(lldb::opaque_compiler_type_t type) override;
  bool IsArrayType(lldb::opaque_compiler_type_t type,
                   CompilerType *element_type, uint64_t *size,
                   bool *is_incomplete) override;
  bool IsAggregateType(lldb::opaque_compiler_type_t type) override;
  bool IsCharType(lldb::opaque_compiler_type_t type) override;
  bool IsCompleteType(lldb::opaque_compiler_type_t type) override;
  bool IsDefined(lldb::opaque_compiler_type_t type) override;
  bool IsFloatingPointType(lldb::opaque_compiler_type_t type) override;
  bool IsFunctionType(lldb::opaque_compiler_type_t type) override;
  size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override;
  CompilerType GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                                          const size_t index) override;
  bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override;
  bool IsMemberFunctionPointerType(lldb::opaque_compiler_type_t type) override;
  bool IsMemberDataPointerType(lldb::opaque_compiler_type_t type) override;
  bool IsBlockPointerType(lldb::opaque_compiler_type_t type,
                          CompilerType *function_pointer_type_ptr) override;
  bool IsIntegerType(lldb::opaque_compiler_type_t type,
                     bool &is_signed) override;
  bool IsScopedEnumerationType(lldb::opaque_compiler_type_t type) override;
  bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                             CompilerType *target_type, bool check_cplusplus,
                             bool check_objc) override;
  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     CompilerType *pointee_type) override;
  bool IsScalarType(lldb::opaque_compiler_type_t type) override;
  bool IsVoidType(lldb::opaque_compiler_type_t type) override;
  bool CanPassInRegisters(const CompilerType &type) override;
  bool SupportsLanguage(lldb::LanguageType language) override;
  bool GetCompleteType(lldb::opaque_compiler_type_t type) override;
  uint32_t GetPointerByteSize() override;
  CompilerType GetPointerDiffType(bool is_signed) override;
  CompilerType GetSizeType() override;
  unsigned GetPtrAuthKey(lldb::opaque_compiler_type_t type) override;
  unsigned GetPtrAuthDiscriminator(lldb::opaque_compiler_type_t type) override;
  bool GetPtrAuthAddressDiversity(lldb::opaque_compiler_type_t type) override;
  ConstString GetTypeName(lldb::opaque_compiler_type_t type,
                          bool BaseOnly) override;
  ConstString GetDisplayTypeName(lldb::opaque_compiler_type_t type) override;
  uint32_t GetTypeInfo(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_or_element_compiler_type) override;
  lldb::LanguageType
  GetMinimumLanguage(lldb::opaque_compiler_type_t type) override;
  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override;
  CompilerType GetArrayElementType(lldb::opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) override;
  CompilerType GetCanonicalType(lldb::opaque_compiler_type_t type) override;
  CompilerType
  GetEnumerationIntegerType(lldb::opaque_compiler_type_t type) override;
  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override;
  CompilerType GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                              size_t idx) override;
  CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override;
  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override;
  TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override;
  CompilerType GetPointeeType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetPointerType(lldb::opaque_compiler_type_t type) override;
  const llvm::fltSemantics &GetFloatTypeSemantics(size_t byte_size,
                                                  lldb::Format format) override;
  llvm::Expected<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             ExecutionContextScope *exe_scope) override;
  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type) override;
  lldb::Format GetFormat(lldb::opaque_compiler_type_t type) override;
  llvm::Expected<uint32_t>
  GetNumChildren(lldb::opaque_compiler_type_t type,
                 bool omit_empty_base_classes,
                 const ExecutionContext *exe_ctx) override;
  lldb::BasicType
  GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) override;
  uint32_t GetNumFields(lldb::opaque_compiler_type_t type) override;
  CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                               std::string &name, uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) override;
  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override;
  uint32_t GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) override;
  CompilerType GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                         size_t idx,
                                         uint32_t *bit_offset_ptr) override;
  CompilerType GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                          size_t idx,
                                          uint32_t *bit_offset_ptr) override;
  llvm::Expected<CompilerType>
  GetDereferencedType(lldb::opaque_compiler_type_t type,
                      ExecutionContext *exe_ctx, std::string &deref_name,
                      uint32_t &deref_byte_size, int32_t &deref_byte_offset,
                      ValueObject *valobj, uint64_t &language_flags) override;
  llvm::Expected<CompilerType> GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
      bool transparent_pointers, bool omit_empty_base_classes,
      bool ignore_array_bounds, std::string &child_name,
      uint32_t &child_byte_size, int32_t &child_byte_offset,
      uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
      bool &child_is_base_class, bool &child_is_deref_of_parent,
      ValueObject *valobj, uint64_t &language_flags) override;
  llvm::Expected<uint32_t>
  GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                          llvm::StringRef name,
                          bool omit_empty_base_classes) override;
  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                llvm::StringRef name,
                                bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) override;
  bool DumpTypeValue(lldb::opaque_compiler_type_t type, Stream &s,
                     lldb::Format format, const DataExtractor &data,
                     lldb::offset_t data_offset, size_t data_byte_size,
                     uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                     ExecutionContextScope *exe_scope) override;
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override;
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream &s,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override;
  void Dump(llvm::raw_ostream &output, llvm::StringRef filter,
            bool show_color) override;
  bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) override;
  bool IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                CompilerType *pointee_type) override;
  unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) override;
  std::optional<size_t>
  GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                  ExecutionContextScope *exe_scope) override;
  CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) override;
  CompilerType GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                   size_t bit_size) override;
  bool IsBeingDefined(lldb::opaque_compiler_type_t type) override;
  bool IsConst(lldb::opaque_compiler_type_t type) override;
  uint32_t IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                                  CompilerType *base_type_ptr) override;
  bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) override;
  bool IsTypedefType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetTypedefedType(lldb::opaque_compiler_type_t type) override;
  bool IsVectorType(lldb::opaque_compiler_type_t type,
                    CompilerType *element_type, uint64_t *size) override;
  CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetNonReferenceType(lldb::opaque_compiler_type_t type) override;
  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_type, bool *is_rvalue) override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_TYPESYSTEMCLIKE_H
