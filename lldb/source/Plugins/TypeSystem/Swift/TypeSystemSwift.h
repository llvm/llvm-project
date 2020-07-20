//===-- TypeSystemSwift.h ---------------------------------------*- C++ -*-===//
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

#ifndef liblldb_TypeSystemSwift_h_
#define liblldb_TypeSystemSwift_h_

#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Flags.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

/// The implementation of lldb::Type's m_payload field for TypeSystemSwift.
class TypePayloadSwift {
  /// Layout: bit 1 ... IsFixedValueBuffer.
  Type::Payload m_payload = 0;

  static constexpr unsigned FixedValueBufferBit = 1;

public:
  TypePayloadSwift() = default;
  explicit TypePayloadSwift(bool is_fixed_value_buffer);
  explicit TypePayloadSwift(Type::Payload opaque_payload)
      : m_payload(opaque_payload) {}
  operator Type::Payload() { return m_payload; }

  /// \return whether this is a Swift fixed-size buffer. Resilient variables in
  /// fixed-size buffers may be indirect depending on the runtime size of the
  /// type. This is more a property of the value than of its type.
  bool IsFixedValueBuffer() {
    return Flags(m_payload).Test(FixedValueBufferBit);
  }
  void SetIsFixedValueBuffer(bool is_fixed_value_buffer) {
    m_payload = is_fixed_value_buffer
                    ? Flags(m_payload).Set(FixedValueBufferBit)
                    : Flags(m_payload).Clear(FixedValueBufferBit);
  }
};

/// Abstract base class for all Swift TypeSystems.
///
/// Swift CompilerTypes are either a mangled name or a Swift AST
/// type. If the typesystem is a TypeSystemSwiftTypeRef, they are
/// mangled names.
///
/// \verbatim
///                      TypeSystem (abstract)
///                            │
///                            ↓
///                      TypeSystemSwift (abstract)
///                        │         │
///                        ↓         ↓
///    TypeSystemSwiftTypeRef ⟷ SwiftASTContext (deprecated)
///                                  │
///                                  ↓
///                               SwiftASTContextForExpressions
///
/// \endverbatim
class TypeSystemSwift : public TypeSystem {
  /// LLVM RTTI support.
  static char ID;

public:
  /// LLVM RTTI support.
  /// \{
  bool isA(const void *ClassID) const override { return ClassID == &ID; }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }
  /// \}

  TypeSystemSwift();

  virtual lldb::TypeSP GetCachedType(ConstString mangled) = 0;
  virtual void SetCachedType(ConstString mangled,
                             const lldb::TypeSP &type_sp) = 0;
  virtual bool IsImportedType(lldb::opaque_compiler_type_t type,
                              CompilerType *original_type) = 0;
  virtual CompilerType GetErrorType() = 0;
  virtual CompilerType GetReferentType(lldb::opaque_compiler_type_t type) = 0;
  static CompilerType GetInstanceType(CompilerType ct);
  virtual CompilerType GetInstanceType(lldb::opaque_compiler_type_t type) = 0;
  enum class TypeAllocationStrategy { eInline, ePointer, eDynamic, eUnknown };
  virtual TypeAllocationStrategy
  GetAllocationStrategy(lldb::opaque_compiler_type_t type) = 0;
  struct TupleElement {
    ConstString element_name;
    CompilerType element_type;
  };
  virtual CompilerType
  CreateTupleType(const std::vector<TupleElement> &elements) = 0;
  virtual void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, bool print_help_if_available,
      bool print_extensions_if_available,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) = 0;
  virtual void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream *s,
      bool print_help_if_available, bool print_extensions_if_available,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) = 0;
  virtual CompilerType
  GetTypeFromMangledTypename(ConstString mangled_typename) = 0;

  /// Unavailable hardcoded functions that don't make sense for Swift.
  /// \{
  ConstString DeclContextGetName(void *opaque_decl_ctx) override { return {}; }
  ConstString DeclContextGetScopeQualifiedName(void *opaque_decl_ctx) override {
    return {};
  }
  bool
  DeclContextIsClassMethod(void *opaque_decl_ctx,
                           lldb::LanguageType *language_ptr,
                           bool *is_instance_method_ptr,
                           ConstString *language_object_name_ptr) override {
    return false;
  }
  bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsCharType(lldb::opaque_compiler_type_t type) override { return false; }
  bool IsCompleteType(lldb::opaque_compiler_type_t type) override {
    return true;
  }
  bool IsConst(lldb::opaque_compiler_type_t type) override { return false; }
  bool IsFloatingPointType(lldb::opaque_compiler_type_t type, uint32_t &count,
                           bool &is_complex) override;
  bool IsIntegerType(lldb::opaque_compiler_type_t type,
                     bool &is_signed) override;
  bool IsScalarType(lldb::opaque_compiler_type_t type) override;
  bool IsCStringType(lldb::opaque_compiler_type_t type,
                     uint32_t &length) override {
    return false;
  }
  bool IsVectorType(lldb::opaque_compiler_type_t type,
                    CompilerType *element_type, uint64_t *size) override {
    return false;
  }
  uint32_t IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                                  CompilerType *base_type_ptr) override {
    return 0;
  }
  bool IsBlockPointerType(lldb::opaque_compiler_type_t type,
                          CompilerType *function_pointer_type_ptr) override {
    return false;
  }
  bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsBeingDefined(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool GetCompleteType(lldb::opaque_compiler_type_t type) override {
    return true;
  }
  bool CanPassInRegisters(const CompilerType &type) override {
    // FIXME: Implement this. There was an abort() here to figure out which
    // tests where hitting this code. At least TestSwiftReturns and
    // TestSwiftStepping were failing because of this Darwin.
    return false;
  }
  lldb::LanguageType
  GetMinimumLanguage(lldb::opaque_compiler_type_t type) override {
    assert(type && "CompilerType::GetMinimumLanguage() is not supposed to "
                   "forward calls with NULL types ");
    return lldb::eLanguageTypeSwift;
  }
  unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) override {
    return 0;
  }
  CompilerType
  GetTypeForDecl(lldb::opaque_compiler_type_t opaque_decl) override {
    llvm_unreachable("GetTypeForDecl not implemented");
  }
  CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) override {
    return {};
  }
  const llvm::fltSemantics &GetFloatTypeSemantics(size_t byte_size) override {
    // See: https://reviews.llvm.org/D67239. At this time of writing this API
    // is only used by DumpDataExtractor for the C type system.
    llvm_unreachable("GetFloatTypeSemantics not implemented.");
  }
  lldb::BasicType
  GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) override {
    return lldb::eBasicTypeInvalid;
  }
  uint32_t
  GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t opaque_type) override {
    return 0;
  }
  CompilerType
  GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t opaque_type,
                             size_t idx, uint32_t *bit_offset_ptr) override {
    return {};
  }
  bool
  ShouldTreatScalarValueAsAddress(lldb::opaque_compiler_type_t type) override;
  /// \}
protected:
  /// Used in the logs.
  std::string m_description;
};

} // namespace lldb_private
#endif
