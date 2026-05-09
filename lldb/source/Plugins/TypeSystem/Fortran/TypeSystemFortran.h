//===-- TypeSystemFortran.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_TYPESYSTEMFORTRAN_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_TYPESYSTEMFORTRAN_H
#include "lldb/Symbol/TypeSystem.h"
#include "llvm/Support/ErrorHandling.h"
namespace lldb_private {

class TypeSystemFortran : public TypeSystem {

  // llvm casting support
  bool isA(const void *ClassID) const override { return ClassID == &ID; }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }

  TypeSystemFortran();
  ~TypeSystemFortran();

  // CompilerDecl functions
  ConstString DeclGetName(void *opaque_decl) override { return ConstString(); }

  CompilerType GetTypeForDecl(void *opaque_decl) override {
    return CompilerType();
  }

  // CompilerDeclContext functions

  ConstString DeclContextGetName(void *opaque_decl_ctx) override {
    return ConstString();
  }

  ConstString DeclContextGetScopeQualifiedName(void *opaque_decl_ctx) override {
    return ConstString();
  }

  bool DeclContextIsClassMethod(void *opaque_decl_ctx) override {
    return false;
  }

  bool DeclContextIsContainedInLookup(void *opaque_decl_ctx,
                                      void *other_opaque_decl_ctx) override {
    return false;
  }

  lldb::LanguageType DeclContextGetLanguage(void *opaque_decl_ctx) override {
    return lldb::LanguageType::eLanguageTypeUnknown;
  }

// Tests
#ifndef NDEBUG
  /// Verify the integrity of the type to catch CompilerTypes that mix
  /// and match invalid TypeSystem/Opaque type pairs.
  bool Verify(lldb::opaque_compiler_type_t type) { return false; };
#endif

  bool IsArrayType(lldb::opaque_compiler_type_t type,
                   CompilerType *element_type, uint64_t *size,
                   bool *is_incomplete) override {
    return false;
  };

  bool IsAggregateType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsCharType(lldb::opaque_compiler_type_t type) override { return false; }

  bool IsCompleteType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsDefined(lldb::opaque_compiler_type_t type) override { return false; }

  bool IsFloatingPointType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsFunctionType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  CompilerType GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                                          const size_t index) override {
    return CompilerType();
  }

  bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsMemberFunctionPointerType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsMemberDataPointerType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsBlockPointerType(lldb::opaque_compiler_type_t type,
                          CompilerType *function_pointer_type_ptr) override {
    return false;
  }

  bool IsIntegerType(lldb::opaque_compiler_type_t type,
                     bool &is_signed) override {
    return false;
  };

  bool IsScopedEnumerationType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                             CompilerType *target_type, // Can pass NULL
                             bool check_cplusplus, bool check_objc) override {
    return false;
  }

  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     CompilerType *pointee_type) override {
    return false;
  }

  bool IsScalarType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsVoidType(lldb::opaque_compiler_type_t type) override { return false; }

  bool CanPassInRegisters(const CompilerType &type) override { return false; }

  // TypeSystems can support more than one language
  bool SupportsLanguage(lldb::LanguageType language) override {
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

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  static llvm::StringRef GetPluginNameStatic() { return "fortran"; }

  // Type Completion

  bool GetCompleteType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  // AST related queries

  uint32_t GetPointerByteSize() override { return 0; }

  CompilerType GetPointerDiffType(bool is_signed) override {
    return CompilerType();
  }

  unsigned GetPtrAuthKey(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  unsigned GetPtrAuthDiscriminator(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  bool GetPtrAuthAddressDiversity(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  // Accessors

  ConstString GetTypeName(lldb::opaque_compiler_type_t type,
                          bool BaseOnly) override {
    return ConstString();
  }

  ConstString GetDisplayTypeName(lldb::opaque_compiler_type_t type) override {
    return ConstString();
  }

  uint32_t
  GetTypeInfo(lldb::opaque_compiler_type_t type,
              CompilerType *pointee_or_element_compiler_type) override {
    return 0;
  }

  lldb::LanguageType
  GetMinimumLanguage(lldb::opaque_compiler_type_t type) override {
    return lldb::LanguageType::eLanguageTypeUnknown;
  }

  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override {
    return lldb::TypeClass::eTypeClassInvalid;
  }

  // Creating related types

  CompilerType GetArrayElementType(lldb::opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) override {
    return CompilerType();
  }

  CompilerType GetCanonicalType(lldb::opaque_compiler_type_t type) override {
    return CompilerType();
  }

  CompilerType
  GetEnumerationIntegerType(lldb::opaque_compiler_type_t type) override {
    return CompilerType();
  }

  // Returns -1 if this isn't a function of if the function doesn't have a
  // prototype Returns a value >= 0 if there is a prototype.
  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override {
    return -1;
  }

  CompilerType GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                              size_t idx) override {
    return CompilerType();
  }

  CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override {
    return CompilerType();
  }

  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override {
    return TypeMemberFunctionImpl();
  }

  CompilerType GetPointeeType(lldb::opaque_compiler_type_t type) override {
    return CompilerType();
  }

  CompilerType GetPointerType(lldb::opaque_compiler_type_t type) override {
    return CompilerType();
  }

  // Exploring the type

  const llvm::fltSemantics &
  GetFloatTypeSemantics(size_t byte_size, lldb::Format format) override {
    return llvm::APFloatBase::Bogus();
  }

  llvm::Expected<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             ExecutionContextScope *exe_scope) override {
    return 0;
  }

  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type) override {
    return lldb::eEncodingInvalid;
  }

  lldb::Format GetFormat(lldb::opaque_compiler_type_t type) override {
    return lldb::eFormatDefault;
  }

  llvm::Expected<uint32_t>
  GetNumChildren(lldb::opaque_compiler_type_t type,
                 bool omit_empty_base_classes,
                 const ExecutionContext *exe_ctx) override {
    return 0;
  }

  lldb::BasicType
  GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) override {
    return lldb::eBasicTypeUnsignedInt;
  }

  uint32_t GetNumFields(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                               std::string &name, uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) override {
    return CompilerType();
  }

  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  uint32_t
  GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  CompilerType GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                         size_t idx,
                                         uint32_t *bit_offset_ptr) override {
    return CompilerType();
  }

  CompilerType GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                          size_t idx,
                                          uint32_t *bit_offset_ptr) override {
    return CompilerType();
  }

  llvm::Expected<CompilerType>
  GetDereferencedType(lldb::opaque_compiler_type_t type,
                      ExecutionContext *exe_ctx, std::string &deref_name,
                      uint32_t &deref_byte_size, int32_t &deref_byte_offset,
                      ValueObject *valobj, uint64_t &language_flags) override {
    return CompilerType();
  }

  llvm::Expected<CompilerType> GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
      bool transparent_pointers, bool omit_empty_base_classes,
      bool ignore_array_bounds, std::string &child_name,
      uint32_t &child_byte_size, int32_t &child_byte_offset,
      uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
      bool &child_is_base_class, bool &child_is_deref_of_parent,
      ValueObject *valobj, uint64_t &language_flags) override {
    return CompilerType();
  }

  // Lookup a child given a name. This function will match base class names and
  // member member names in "clang_type" only, not descendants.
  llvm::Expected<uint32_t>
  GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                          llvm::StringRef name,
                          bool omit_empty_base_classes) override {
    return 0;
  }

  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                llvm::StringRef name,
                                bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) override {
    return 0;
  }

#ifndef NDEBUG
  /// Convenience LLVM-style dump method for use in the debugger only.
  LLVM_DUMP_METHOD void dump(lldb::opaque_compiler_type_t type) const override {
  }
#endif

  bool DumpTypeValue(lldb::opaque_compiler_type_t type, Stream &s,
                     lldb::Format format, const DataExtractor &data,
                     lldb::offset_t data_offset, size_t data_byte_size,
                     uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                     ExecutionContextScope *exe_scope) override {
    return false;
  }

  /// Dump the type to stdout.
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override {}

  /// Print a description of the type to a stream. The exact implementation
  /// varies, but the expectation is that eDescriptionLevelFull returns a
  /// source-like representation of the type, whereas eDescriptionLevelVerbose
  /// does a dump of the underlying AST if applicable.
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream &s,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override {}

  /// Dump a textual representation of the internal TypeSystem state to the
  /// given stream.
  ///
  /// This should not modify the state of the TypeSystem if possible.
  ///
  /// \param[out] output Stream to dup the AST into.
  /// \param[in] filter If empty, dump whole AST. If non-empty, will only
  /// dump decls whose names contain \c filter.
  /// \param[in] show_color If true, prints the AST color-highlighted.
  void Dump(llvm::raw_ostream &output, llvm::StringRef filter,
            bool show_color) override {}

  /// This is used by swift.
  bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  // TODO: Determine if these methods should move to TypeSystemClang.

  bool IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                CompilerType *pointee_type) override {
    return false;
  }

  unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  std::optional<size_t>
  GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                  ExecutionContextScope *exe_scope) override {
    return 0;
  }

  CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) override {
    return CompilerType();
  }

  CompilerType GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                   size_t bit_size) override {
    return CompilerType();
  }

  bool IsBeingDefined(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsConst(lldb::opaque_compiler_type_t type) override { return false; }

  uint32_t IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                                  CompilerType *base_type_ptr) override {
    return 0;
  }

  bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsTypedefType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  // If the current object represents a typedef type, get the underlying type
  CompilerType GetTypedefedType(lldb::opaque_compiler_type_t type) override {
    return CompilerType();
  }

  bool IsVectorType(lldb::opaque_compiler_type_t type,
                    CompilerType *element_type, uint64_t *size) override {
    return false;
  }

  CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override {
    return CompilerType();
  }

  CompilerType GetNonReferenceType(lldb::opaque_compiler_type_t type) override {
    return CompilerType();
  }
  // TODO
  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_type, bool *is_rvalue) override {
    return false;
  }

private:
  // LLVM RTTI support
  static char ID;
};
} // namespace lldb_private
#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_TYPESYSTEMFORTRAN_H
