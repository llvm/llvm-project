//===-- TypeSystemSwiftTypeRef.cpp ----------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Utility/Log.h"

#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"
#include "swift/Strings.h"

using namespace lldb;
using namespace lldb_private;

char TypeSystemSwift::ID;
char TypeSystemSwiftTypeRef::ID;

TypeSystemSwift::TypeSystemSwift() : TypeSystem() {}

CompilerType TypeSystemSwift::GetInstanceType(CompilerType compiler_type) {
  auto *ts = compiler_type.GetTypeSystem();
  if (auto *tr = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(ts))
    return tr->GetInstanceType(compiler_type.GetOpaqueQualType());
  if (auto *ast = llvm::dyn_cast_or_null<SwiftASTContext>(ts))
    return ast->GetInstanceType(compiler_type.GetOpaqueQualType());
  return {};
}

TypeSystemSwiftTypeRef::TypeSystemSwiftTypeRef(
    SwiftASTContext *swift_ast_context)
    : m_swift_ast_context(swift_ast_context) {
  m_description = "TypeSystemSwiftTypeRef";
}

const char *TypeSystemSwiftTypeRef::AsMangledName(void *type) {
  assert(type && *reinterpret_cast<const char *>(type) == '$' &&
         "wrong type system");
  return reinterpret_cast<const char *>(type);
}

ConstString TypeSystemSwiftTypeRef::GetMangledTypeName(void *type) {
  // FIXME: Suboptimal performance, because the ConstString is looked up again.
  return ConstString(AsMangledName(type));
}

void *TypeSystemSwiftTypeRef::ReconstructType(void *type) {
  Status error;
  return m_swift_ast_context->ReconstructType(GetMangledTypeName(type), error);
}

CompilerType TypeSystemSwiftTypeRef::ReconstructType(CompilerType type) {
  return {m_swift_ast_context, ReconstructType(type.GetOpaqueQualType())};
}

CompilerType TypeSystemSwiftTypeRef::GetTypeFromMangledTypename(
    ConstString mangled_typename) {
  return {this, (void *)mangled_typename.AsCString()};
}

lldb::TypeSP TypeSystemSwiftTypeRef::GetCachedType(ConstString mangled) {
  return m_swift_ast_context->GetCachedType(mangled);
}

void TypeSystemSwiftTypeRef::SetCachedType(ConstString mangled,
                                           const lldb::TypeSP &type_sp) {
  return m_swift_ast_context->SetCachedType(mangled, type_sp);
}

Module *TypeSystemSwiftTypeRef::GetModule() const {
  return m_swift_ast_context ? m_swift_ast_context->GetModule() : nullptr;
}

ConstString TypeSystemSwiftTypeRef::GetPluginName() {
  return ConstString("TypeSystemSwiftTypeRef");
}
uint32_t TypeSystemSwiftTypeRef::GetPluginVersion() { return 1; }

bool TypeSystemSwiftTypeRef::SupportsLanguage(lldb::LanguageType language) {
  return language == eLanguageTypeSwift;
}

Status TypeSystemSwiftTypeRef::IsCompatible() {
  return m_swift_ast_context->IsCompatible();
}

void TypeSystemSwiftTypeRef::DiagnoseWarnings(Process &process,
                                              Module &module) const {
  m_swift_ast_context->DiagnoseWarnings(process, module);
}
DWARFASTParser *TypeSystemSwiftTypeRef::GetDWARFParser() {
  return m_swift_ast_context->GetDWARFParser();
}
ConstString TypeSystemSwiftTypeRef::DeclContextGetName(void *opaque_decl_ctx) {
  return m_swift_ast_context->DeclContextGetName(opaque_decl_ctx);
}
ConstString TypeSystemSwiftTypeRef::DeclContextGetScopeQualifiedName(
    void *opaque_decl_ctx) {
  return m_swift_ast_context->DeclContextGetScopeQualifiedName(opaque_decl_ctx);
}
bool TypeSystemSwiftTypeRef::DeclContextIsClassMethod(
    void *opaque_decl_ctx, lldb::LanguageType *language_ptr,
    bool *is_instance_method_ptr, ConstString *language_object_name_ptr) {
  return m_swift_ast_context->DeclContextIsClassMethod(
      opaque_decl_ctx, language_ptr, is_instance_method_ptr,
      language_object_name_ptr);
}

// Tests

#ifndef NDEBUG
bool TypeSystemSwiftTypeRef::Verify(lldb::opaque_compiler_type_t type) {
  if (!type)
    return true;

  const char *str = reinterpret_cast<const char *>(type);
  return SwiftLanguageRuntime::IsSwiftMangledName(str);
}
#endif

bool TypeSystemSwiftTypeRef::IsArrayType(void *type, CompilerType *element_type,
                                         uint64_t *size, bool *is_incomplete) {
  return m_swift_ast_context->IsArrayType(ReconstructType(type), element_type,
                                          size, is_incomplete);
}
bool TypeSystemSwiftTypeRef::IsAggregateType(void *type) {
  return m_swift_ast_context->IsAggregateType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsCharType(void *type) {
  return m_swift_ast_context->IsCharType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsCompleteType(void *type) {
  return m_swift_ast_context->IsCompleteType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsDefined(void *type) {
  return m_swift_ast_context->IsDefined(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsFloatingPointType(void *type, uint32_t &count,
                                                 bool &is_complex) {
  return m_swift_ast_context->IsFloatingPointType(ReconstructType(type), count,
                                                  is_complex);
}
bool TypeSystemSwiftTypeRef::IsFunctionType(void *type, bool *is_variadic_ptr) {
  return m_swift_ast_context->IsFunctionType(ReconstructType(type),
                                             is_variadic_ptr);
}
size_t TypeSystemSwiftTypeRef::GetNumberOfFunctionArguments(void *type) {
  return m_swift_ast_context->GetNumberOfFunctionArguments(
      ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetFunctionArgumentAtIndex(void *type,
                                                   const size_t index) {
  return m_swift_ast_context->GetFunctionArgumentAtIndex(ReconstructType(type),
                                                         index);
}
bool TypeSystemSwiftTypeRef::IsFunctionPointerType(void *type) {
  return m_swift_ast_context->IsFunctionPointerType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsBlockPointerType(
    void *type, CompilerType *function_pointer_type_ptr) {
  return m_swift_ast_context->IsBlockPointerType(ReconstructType(type),
                                                 function_pointer_type_ptr);
}
bool TypeSystemSwiftTypeRef::IsIntegerType(void *type, bool &is_signed) {
  return m_swift_ast_context->IsIntegerType(ReconstructType(type), is_signed);
}
bool TypeSystemSwiftTypeRef::IsPossibleDynamicType(void *type,
                                                   CompilerType *target_type,
                                                   bool check_cplusplus,
                                                   bool check_objc) {
  return m_swift_ast_context->IsPossibleDynamicType(
      ReconstructType(type), target_type, check_cplusplus, check_objc);
}
bool TypeSystemSwiftTypeRef::IsPointerType(void *type,
                                           CompilerType *pointee_type) {
  return m_swift_ast_context->IsPointerType(ReconstructType(type),
                                            pointee_type);
}
bool TypeSystemSwiftTypeRef::IsScalarType(void *type) {
  return m_swift_ast_context->IsScalarType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsVoidType(void *type) {
  return m_swift_ast_context->IsVoidType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::CanPassInRegisters(const CompilerType &type) {
  return m_swift_ast_context->CanPassInRegisters(
      {m_swift_ast_context, ReconstructType(type.GetOpaqueQualType())});
}
// Type Completion
bool TypeSystemSwiftTypeRef::GetCompleteType(void *type) {
  return m_swift_ast_context->GetCompleteType(ReconstructType(type));
}
// AST related queries
uint32_t TypeSystemSwiftTypeRef::GetPointerByteSize() {
  return m_swift_ast_context->GetPointerByteSize();
}
// Accessors
ConstString TypeSystemSwiftTypeRef::GetTypeName(void *type) {
  return m_swift_ast_context->GetTypeName(ReconstructType(type));
}
ConstString
TypeSystemSwiftTypeRef::GetDisplayTypeName(void *type,
                                           const SymbolContext *sc) {
  return m_swift_ast_context->GetDisplayTypeName(ReconstructType(type), sc);
}
uint32_t TypeSystemSwiftTypeRef::GetTypeInfo(
    void *type, CompilerType *pointee_or_element_clang_type) {
  return m_swift_ast_context->GetTypeInfo(ReconstructType(type),
                                          pointee_or_element_clang_type);
}
lldb::LanguageType TypeSystemSwiftTypeRef::GetMinimumLanguage(void *type) {
  return m_swift_ast_context->GetMinimumLanguage(ReconstructType(type));
}
lldb::TypeClass TypeSystemSwiftTypeRef::GetTypeClass(void *type) {
  return m_swift_ast_context->GetTypeClass(ReconstructType(type));
}

// Creating related types
CompilerType TypeSystemSwiftTypeRef::GetArrayElementType(void *type,
                                                         uint64_t *stride) {
  return m_swift_ast_context->GetArrayElementType(ReconstructType(type),
                                                  stride);
}
CompilerType TypeSystemSwiftTypeRef::GetCanonicalType(void *type) {
  return m_swift_ast_context->GetCanonicalType(ReconstructType(type));
}
int TypeSystemSwiftTypeRef::GetFunctionArgumentCount(void *type) {
  return m_swift_ast_context->GetFunctionArgumentCount(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetFunctionArgumentTypeAtIndex(void *type, size_t idx) {
  return m_swift_ast_context->GetFunctionArgumentTypeAtIndex(
      ReconstructType(type), idx);
}
CompilerType TypeSystemSwiftTypeRef::GetFunctionReturnType(void *type) {
  return m_swift_ast_context->GetFunctionReturnType(ReconstructType(type));
}
size_t TypeSystemSwiftTypeRef::GetNumMemberFunctions(void *type) {
  return m_swift_ast_context->GetNumMemberFunctions(ReconstructType(type));
}
TypeMemberFunctionImpl
TypeSystemSwiftTypeRef::GetMemberFunctionAtIndex(void *type, size_t idx) {
  return m_swift_ast_context->GetMemberFunctionAtIndex(ReconstructType(type),
                                                       idx);
}
CompilerType TypeSystemSwiftTypeRef::GetPointeeType(void *type) {
  return m_swift_ast_context->GetPointeeType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetPointerType(void *type) {
  return m_swift_ast_context->GetPointerType(ReconstructType(type));
}

// Exploring the type
const llvm::fltSemantics &
TypeSystemSwiftTypeRef::GetFloatTypeSemantics(size_t byte_size) {
  return m_swift_ast_context->GetFloatTypeSemantics(byte_size);
}
llvm::Optional<uint64_t>
TypeSystemSwiftTypeRef::GetBitSize(lldb::opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) {
  return m_swift_ast_context->GetBitSize(ReconstructType(type), exe_scope);
}
llvm::Optional<uint64_t>
TypeSystemSwiftTypeRef::GetByteStride(lldb::opaque_compiler_type_t type,
                                      ExecutionContextScope *exe_scope) {
  return m_swift_ast_context->GetByteStride(ReconstructType(type), exe_scope);
}
lldb::Encoding TypeSystemSwiftTypeRef::GetEncoding(void *type,
                                                   uint64_t &count) {
  return m_swift_ast_context->GetEncoding(ReconstructType(type), count);
}
lldb::Format TypeSystemSwiftTypeRef::GetFormat(void *type) {
  return m_swift_ast_context->GetFormat(ReconstructType(type));
}
uint32_t
TypeSystemSwiftTypeRef::GetNumChildren(void *type, bool omit_empty_base_classes,
                                       const ExecutionContext *exe_ctx) {
  return m_swift_ast_context->GetNumChildren(ReconstructType(type),
                                             omit_empty_base_classes, exe_ctx);
}
lldb::BasicType TypeSystemSwiftTypeRef::GetBasicTypeEnumeration(void *type) {
  return m_swift_ast_context->GetBasicTypeEnumeration(ReconstructType(type));
}
uint32_t TypeSystemSwiftTypeRef::GetNumFields(void *type) {
  return m_swift_ast_context->GetNumFields(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetFieldAtIndex(
    void *type, size_t idx, std::string &name, uint64_t *bit_offset_ptr,
    uint32_t *bitfield_bit_size_ptr, bool *is_bitfield_ptr) {
  return m_swift_ast_context->GetFieldAtIndex(
      ReconstructType(type), idx, name, bit_offset_ptr, bitfield_bit_size_ptr,
      is_bitfield_ptr);
}
CompilerType TypeSystemSwiftTypeRef::GetChildCompilerTypeAtIndex(
    void *type, ExecutionContext *exe_ctx, size_t idx,
    bool transparent_pointers, bool omit_empty_base_classes,
    bool ignore_array_bounds, std::string &child_name,
    uint32_t &child_byte_size, int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class, bool &child_is_deref_of_parent,
    ValueObject *valobj, uint64_t &language_flags) {
  return m_swift_ast_context->GetChildCompilerTypeAtIndex(
      ReconstructType(type), exe_ctx, idx, transparent_pointers,
      omit_empty_base_classes, ignore_array_bounds, child_name, child_byte_size,
      child_byte_offset, child_bitfield_bit_size, child_bitfield_bit_offset,
      child_is_base_class, child_is_deref_of_parent, valobj, language_flags);
}
uint32_t
TypeSystemSwiftTypeRef::GetIndexOfChildWithName(void *type, const char *name,
                                                bool omit_empty_base_classes) {
  return m_swift_ast_context->GetIndexOfChildWithName(
      ReconstructType(type), name, omit_empty_base_classes);
}
size_t TypeSystemSwiftTypeRef::GetIndexOfChildMemberWithName(
    void *type, const char *name, bool omit_empty_base_classes,
    std::vector<uint32_t> &child_indexes) {
  return m_swift_ast_context->GetIndexOfChildMemberWithName(
      ReconstructType(type), name, omit_empty_base_classes, child_indexes);
}
size_t TypeSystemSwiftTypeRef::GetNumTemplateArguments(void *type) {
  return m_swift_ast_context->GetNumTemplateArguments(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetTypeForFormatters(void *type) {
  return m_swift_ast_context->GetTypeForFormatters(ReconstructType(type));
}
LazyBool TypeSystemSwiftTypeRef::ShouldPrintAsOneLiner(void *type,
                                                       ValueObject *valobj) {
  return m_swift_ast_context->ShouldPrintAsOneLiner(ReconstructType(type),
                                                    valobj);
}
bool TypeSystemSwiftTypeRef::IsMeaninglessWithoutDynamicResolution(void *type) {
  return m_swift_ast_context->IsMeaninglessWithoutDynamicResolution(
      ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsImportedType(CompilerType type,
                                            CompilerType *original_type) {
  return m_swift_ast_context->IsImportedType(
      {m_swift_ast_context, ReconstructType(type.GetOpaqueQualType())},
      original_type);
}
bool TypeSystemSwiftTypeRef::IsErrorType(CompilerType compiler_type) {
  return m_swift_ast_context->IsErrorType(
      {m_swift_ast_context,
       ReconstructType(compiler_type.GetOpaqueQualType())});
}
CompilerType TypeSystemSwiftTypeRef::GetErrorType() {
  return m_swift_ast_context->GetErrorType();
}

CompilerType
TypeSystemSwiftTypeRef::GetReferentType(CompilerType compiler_type) {
  return m_swift_ast_context->GetReferentType(
      {m_swift_ast_context,
       ReconstructType(compiler_type.GetOpaqueQualType())});
}

CompilerType TypeSystemSwiftTypeRef::GetInstanceType(void *type) {
  return m_swift_ast_context->GetInstanceType(ReconstructType(type));
}
TypeSystemSwift::TypeAllocationStrategy
TypeSystemSwiftTypeRef::GetAllocationStrategy(CompilerType type) {
  return m_swift_ast_context->GetAllocationStrategy(
      {m_swift_ast_context, ReconstructType(type.GetOpaqueQualType())});
}
CompilerType TypeSystemSwiftTypeRef::CreateTupleType(
    const std::vector<TupleElement> &elements) {
  return m_swift_ast_context->CreateTupleType(elements);
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(
    void *type, bool print_help_if_available,
    bool print_extensions_if_available) {
  return m_swift_ast_context->DumpTypeDescription(
      ReconstructType(type), print_help_if_available, print_help_if_available);
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(
    void *type, Stream *s, bool print_help_if_available,
    bool print_extensions_if_available) {
  return m_swift_ast_context->DumpTypeDescription(
      ReconstructType(type), s, print_help_if_available,
      print_extensions_if_available);
}

// Dumping types
#ifndef NDEBUG
/// Convenience LLVM-style dump method for use in the debugger only.
LLVM_DUMP_METHOD void
TypeSystemSwiftTypeRef::dump(lldb::opaque_compiler_type_t type) const {
  llvm::dbgs() << reinterpret_cast<const char *>(type) << "\n";
}
#endif

void TypeSystemSwiftTypeRef::DumpValue(
    void *type, ExecutionContext *exe_ctx, Stream *s, lldb::Format format,
    const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, bool show_types, bool show_summary,
    bool verbose, uint32_t depth) {
  return m_swift_ast_context->DumpValue(
      ReconstructType(type), exe_ctx, s, format, data, data_offset,
      data_byte_size, bitfield_bit_size, bitfield_bit_offset, show_types,
      show_summary, verbose, depth);
}

bool TypeSystemSwiftTypeRef::DumpTypeValue(
    void *type, Stream *s, lldb::Format format, const DataExtractor &data,
    lldb::offset_t data_offset, size_t data_byte_size,
    uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
    ExecutionContextScope *exe_scope, bool is_base_class) {
  return m_swift_ast_context->DumpTypeValue(
      ReconstructType(type), s, format, data, data_offset, data_byte_size,
      bitfield_bit_size, bitfield_bit_offset, exe_scope, is_base_class);
}

void TypeSystemSwiftTypeRef::DumpTypeDescription(void *type) {
  return m_swift_ast_context->DumpTypeDescription(ReconstructType(type));
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(void *type, Stream *s) {
  return m_swift_ast_context->DumpTypeDescription(ReconstructType(type), s);
}
bool TypeSystemSwiftTypeRef::IsRuntimeGeneratedType(void *type) {
  return m_swift_ast_context->IsRuntimeGeneratedType(ReconstructType(type));
}
void TypeSystemSwiftTypeRef::DumpSummary(void *type, ExecutionContext *exe_ctx,
                                         Stream *s, const DataExtractor &data,
                                         lldb::offset_t data_offset,
                                         size_t data_byte_size) {
  return m_swift_ast_context->DumpSummary(ReconstructType(type), exe_ctx, s,
                                          data, data_offset, data_byte_size);
}
bool TypeSystemSwiftTypeRef::IsPointerOrReferenceType(
    void *type, CompilerType *pointee_type) {
  return m_swift_ast_context->IsPointerOrReferenceType(ReconstructType(type),
                                                       pointee_type);
}
unsigned TypeSystemSwiftTypeRef::GetTypeQualifiers(void *type) {
  return m_swift_ast_context->GetTypeQualifiers(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsCStringType(void *type, uint32_t &length) {
  return m_swift_ast_context->IsCStringType(ReconstructType(type), length);
}
llvm::Optional<size_t>
TypeSystemSwiftTypeRef::GetTypeBitAlign(void *type,
                                        ExecutionContextScope *exe_scope) {
  return m_swift_ast_context->GetTypeBitAlign(ReconstructType(type), exe_scope);
}
CompilerType
TypeSystemSwiftTypeRef::GetBasicTypeFromAST(lldb::BasicType basic_type) {
  return m_swift_ast_context->GetBasicTypeFromAST(basic_type);
}
bool TypeSystemSwiftTypeRef::IsBeingDefined(void *type) {
  return m_swift_ast_context->IsBeingDefined(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsConst(void *type) {
  return m_swift_ast_context->IsConst(ReconstructType(type));
}
uint32_t
TypeSystemSwiftTypeRef::IsHomogeneousAggregate(void *type,
                                               CompilerType *base_type_ptr) {
  return m_swift_ast_context->IsHomogeneousAggregate(ReconstructType(type),
                                                     base_type_ptr);
}
bool TypeSystemSwiftTypeRef::IsPolymorphicClass(void *type) {
  return m_swift_ast_context->IsPolymorphicClass(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsTypedefType(void *type) {
  return m_swift_ast_context->IsTypedefType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetTypedefedType(void *type) {
  return m_swift_ast_context->GetTypedefedType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetTypeForDecl(void *opaque_decl) {
  return m_swift_ast_context->GetTypeForDecl(opaque_decl);
}
bool TypeSystemSwiftTypeRef::IsVectorType(void *type,
                                          CompilerType *element_type,
                                          uint64_t *size) {
  return m_swift_ast_context->IsVectorType(ReconstructType(type), element_type,
                                           size);
}
CompilerType TypeSystemSwiftTypeRef::GetFullyUnqualifiedType(void *type) {
  return m_swift_ast_context->GetFullyUnqualifiedType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetNonReferenceType(void *type) {
  return m_swift_ast_context->GetNonReferenceType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetLValueReferenceType(void *type) {
  return m_swift_ast_context->GetLValueReferenceType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetRValueReferenceType(void *type) {
  return m_swift_ast_context->GetRValueReferenceType(ReconstructType(type));
}
uint32_t TypeSystemSwiftTypeRef::GetNumDirectBaseClasses(void *type) {
  return m_swift_ast_context->GetNumDirectBaseClasses(ReconstructType(type));
}
uint32_t TypeSystemSwiftTypeRef::GetNumVirtualBaseClasses(void *type) {
  return m_swift_ast_context->GetNumVirtualBaseClasses(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetDirectBaseClassAtIndex(void *type, size_t idx,
                                                  uint32_t *bit_offset_ptr) {
  return m_swift_ast_context->GetDirectBaseClassAtIndex(ReconstructType(type),
                                                        idx, bit_offset_ptr);
}
CompilerType
TypeSystemSwiftTypeRef::GetVirtualBaseClassAtIndex(void *type, size_t idx,
                                                   uint32_t *bit_offset_ptr) {
  return m_swift_ast_context->GetVirtualBaseClassAtIndex(ReconstructType(type),
                                                         idx, bit_offset_ptr);
}
bool TypeSystemSwiftTypeRef::IsReferenceType(void *type,
                                             CompilerType *pointee_type,
                                             bool *is_rvalue) {
  return m_swift_ast_context->IsReferenceType(ReconstructType(type),
                                              pointee_type, is_rvalue);
}
bool TypeSystemSwiftTypeRef::ShouldTreatScalarValueAsAddress(
    lldb::opaque_compiler_type_t type) {
  return m_swift_ast_context->ShouldTreatScalarValueAsAddress(
      ReconstructType(type));
}
