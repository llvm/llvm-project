//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the basic classes needed for the Type interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_TYPE_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_TYPE_H

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "lldb/lldb-enumerations.h"

#include "Identifier.h"

namespace lldb_private {
namespace clike {

class Context;
class Namespace;
class Type;

struct TemplateArgument;
struct MemberFunction;
struct StaticDataMember;
struct ObjCMethod;

/// \see Type::GetDynamicTypeCandidate
enum class DynamicTypeCheck : uint8_t {
  None = 0,
  /// Pointers and references to dynamic C++ classes.
  CPlusPlus = 1 << 0,
  /// Pointers to Objective-C objects.
  ObjC = 1 << 1,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/ObjC)
};
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// A reference to another Type.
///
/// The other type can belong to another context.
///
/// This is currently a wrapper around a Type pointer. In the future this could
/// also reference types that are no longer in memory (e.g., because they have
/// been deallocated to free up space).
class TypeRef {
public:
  explicit TypeRef(Type &type) : m_type(&type) {}

  /// The referenced type.
  Type &Get() const { return *m_type; }

private:
  Type *m_type;
};

static_assert(sizeof(TypeRef) == sizeof(void *),
              "TypeRef is expected to be pointer-sized!");

/// A single data member of a record type.
class Field {
public:
  /// \p bitfield_bit_size is the width of a bitfield member, or zero if this is
  /// not a bitfield (in which case \p bitfield_bit_offset must be zero too).
  Field(Identifier name, TypeRef type, uint64_t byte_offset,
        uint32_t bitfield_bit_size = 0, uint32_t bitfield_bit_offset = 0)
      : m_name(name), m_type(type), m_byte_offset(byte_offset),
        m_bitfield_bit_size(bitfield_bit_size),
        m_bitfield_bit_offset(bitfield_bit_offset) {
    assert((IsBitfield() || bitfield_bit_offset == 0) &&
           "bit offset given for a member that is not a bitfield");
  }

  Identifier GetName() const { return m_name; }

  /// The type of this member.
  TypeRef GetType() const { return m_type; }

  /// Offset of this member (or, for a bitfield, of its storage unit) from the
  /// start of the record, in bytes.
  uint64_t GetByteOffset() const { return m_byte_offset; }

  bool IsBitfield() const { return m_bitfield_bit_size != 0; }

  /// The width of this bitfield in bits.
  uint32_t GetBitfieldBitSize() const {
    assert(IsBitfield() && "not a bitfield");
    return m_bitfield_bit_size;
  }

  /// The offset of this bitfield's first bit within its storage unit.
  uint32_t GetBitfieldBitOffset() const {
    assert(IsBitfield() && "not a bitfield");
    return m_bitfield_bit_offset;
  }

private:
  Identifier m_name;
  TypeRef m_type;
  uint64_t m_byte_offset;
  /// Zero means this is not a bitfield.
  uint32_t m_bitfield_bit_size;
  uint32_t m_bitfield_bit_offset;
};

/// A direct base class of a C++ class or Objective-C interface.
class BaseClass {
public:
  /// \p vbase_offset_offset may only be given for a virtual base (see
  /// GetVBaseOffsetOffset).
  BaseClass(TypeRef type, uint64_t byte_offset, bool is_virtual = false,
            std::optional<uint64_t> vbase_offset_offset = std::nullopt)
      : m_type(type), m_byte_offset(byte_offset), m_is_virtual(is_virtual),
        m_vbase_offset_offset(vbase_offset_offset) {
    assert((is_virtual || !vbase_offset_offset) &&
           "vbase offset offset given for a non-virtual base");
  }

  /// The base class type.
  TypeRef GetType() const { return m_type; }

  /// Offset of the base class subobject from the start of the derived record.
  uint64_t GetByteOffset() const { return m_byte_offset; }

  /// True if this is a virtual base class.
  bool IsVirtual() const { return m_is_virtual; }

  /// The Itanium ABI vbase offset offset of a virtual base.
  ///
  /// The base subobject is at `obj + *(ptrdiff_t *)(*(char **)obj -
  /// vbase_offset_offset)`. Returns std::nullopt if this is not a virtual base
  /// or the offset is unknown.
  std::optional<uint64_t> GetVBaseOffsetOffset() const {
    return m_vbase_offset_offset;
  }

private:
  TypeRef m_type;
  uint64_t m_byte_offset;
  bool m_is_virtual;
  std::optional<uint64_t> m_vbase_offset_offset;
};

/// A single constant of an enumeration.
class Enumerator {
public:
  Enumerator(Identifier name, llvm::APInt value)
      : m_name(name), m_value(std::move(value)) {}

  Identifier GetName() const { return m_name; }

  const llvm::APInt &GetValue() const { return m_value; }

private:
  Identifier m_name;
  llvm::APInt m_value;
};

/// The ARMv8.3 pointer authentication schema of a
/// `T *__ptrauth(key, address_discriminated, extra_discriminator)` pointer.
class PtrAuthQualifier {
public:
  PtrAuthQualifier(unsigned key, bool address_discriminated,
                   unsigned extra_discriminator)
      : m_key(key), m_address_discriminated(address_discriminated),
        m_extra_discriminator(extra_discriminator) {
    assert(extra_discriminator <= 0xFFFF &&
           "extra discriminator must fit in 16 bits");
  }

  unsigned GetKey() const { return m_key; }

  /// True if the storage address is part of the discriminator.
  bool IsAddressDiscriminated() const { return m_address_discriminated; }

  /// The constant extra discriminator.
  unsigned GetExtraDiscriminator() const { return m_extra_discriminator; }

private:
  unsigned m_key;
  bool m_address_discriminated;
  unsigned m_extra_discriminator;
};

/// The enclosing namespace and unqualified name of a named type.
///
/// A default-constructed DeclScope represents the global namespace.
class DeclScope {
public:
  DeclScope() = default;
  /// \p unqualified_name must be interned in the IdentifierMap of the Context
  /// that owns the type.
  DeclScope(const Namespace *decl_context, Identifier unqualified_name)
      : m_decl_context(decl_context), m_unqualified_name(unqualified_name) {}

  /// The enclosing namespace, or null for the global namespace.
  const Namespace *GetDeclContext() const { return m_decl_context; }

  /// The unqualified name, or an empty identifier if unknown.
  Identifier GetUnqualifiedName() const { return m_unqualified_name; }

private:
  const Namespace *m_decl_context = nullptr;
  Identifier m_unqualified_name;
};

/// The base class of all Types in the Clike TypeSystem.
///
/// This implements the backend that the TypeSystem API implementation of
/// TypeSystemClike calls. This class is also used by TypeSystemClike as
/// its opaque_compiler_type_t type.
///
/// Types are usually immutable once created.
class Type : public llvm::RTTIExtends<Type, llvm::RTTIRoot> {
public:
  /// LLVM-style RTTI support.
  static char ID;

  virtual ~Type() = default;

  /// The Context that owns this type.
  Context &GetOwningContext() const {
    assert(m_context && "every type is owned by the Context that created it");
    return *m_context;
  }

  /// The type's qualified name.
  virtual Identifier GetName() const { return Identifier(); }

  /// The namespace this type is declared in. Null for the global namespace and
  /// for types without a declaration (e.g., pointers).
  virtual const Namespace *GetDeclContext() const { return nullptr; }

  /// The name of this type without namespace qualifiers (e.g., `string` or
  /// `vector<int, std::allocator<int>>`).
  virtual Identifier GetUnqualifiedName() const { return Identifier(); }

  /// The size of this type in bytes, or std::nullopt if unknown (e.g., for an
  /// incomplete record or a function type).
  virtual std::optional<uint64_t> GetByteSize() const { return std::nullopt; }

  /// The alignment of this type in bits, or std::nullopt if unknown.
  virtual std::optional<uint64_t> GetAlignmentInBits() const {
    return std::nullopt;
  }

  /// Removes all sugar (typedefs, qualifiers, elaborated types, ...).
  virtual Type *Desugar() { return this; }
  const Type *Desugar() const { return const_cast<Type *>(this)->Desugar(); }

  /// True if this type has members (a struct/class/union).
  virtual bool IsAggregate() const { return false; }

  /// True if this type is not a forward declaration.
  virtual bool IsComplete() const { return true; }

  virtual lldb::Encoding GetEncoding() const { return lldb::eEncodingInvalid; }
  virtual lldb::Format GetFormat() const { return lldb::eFormatDefault; }
  virtual lldb::TypeClass GetTypeClass() const { return lldb::eTypeClassOther; }
  virtual uint32_t GetTypeInfo() const { return 0; }

  /// The least specific language that can express this type.
  virtual lldb::LanguageType GetMinimumLanguage() const {
    return lldb::eLanguageTypeC;
  }

  /// Removes elaborated type sugar from the top of this type. Unlike Desugar()
  /// this preserves typedefs.
  virtual Type *StripElaborated() { return this; }
  const Type *StripElaborated() const {
    return const_cast<Type *>(this)->StripElaborated();
  }

  /// Removes all cv-qualifiers from the top of this type, but no other sugar.
  virtual Type *StripCVQualifiers() { return this; }
  const Type *StripCVQualifiers() const {
    return const_cast<Type *>(this)->StripCVQualifiers();
  }

  /// The combined cv-qualifiers at the top of this type.
  ///
  /// This reuses the, using the clang::Qualifiers values of Const = 0x1,
  /// Restrict = 0x2, Volatile = 0x4.
  virtual unsigned GetCVRQualifiers() const { return 0; }

  /// True if the outermost qualifier of this type is `const`.
  virtual bool IsConst() const { return false; }

  /// The pointer authentication qualifier of this type, or std::nullopt if
  /// there is none.
  virtual std::optional<PtrAuthQualifier> GetPtrAuthQualifier() const {
    return std::nullopt;
  }

  /// The aliased type if this is a typedef, otherwise null.
  virtual Type *GetTypedefedType() const { return nullptr; }

  /// True for `void`.
  virtual bool IsVoid() const { return false; }

  /// True for `char`, `signed char` and `unsigned char`.
  virtual bool IsChar() const { return false; }

  /// True for builtin integer types and enumerations with an integer encoding.
  /// Pointers and `std::nullptr_t` are not integers.
  virtual bool IsInteger() const { return false; }

  /// True for types to which the C++ integral promotions apply.
  virtual bool IsPromotableInteger() const { return false; }

  /// The lldb::BasicType of this type, or eBasicTypeInvalid if this is not a
  /// builtin type.
  virtual lldb::BasicType GetBasicTypeEnumeration() const {
    return lldb::eBasicTypeInvalid;
  }

  /// True for pointers, including block pointers.
  virtual bool IsPointer() const { return false; }

  /// True for block pointers (`int (^)(int)`).
  virtual bool IsBlockPointer() const { return false; }

  /// True for function pointers (`void (*)(int)`).
  virtual bool IsFunctionPointer() const { return false; }

  /// True for an lvalue (`T &`) or rvalue (`T &&`) reference.
  virtual bool IsReference() const { return false; }

  /// True for an rvalue reference (`T &&`).
  virtual bool IsRValueReference() const { return false; }

  /// The pointee type of a pointer or reference, otherwise null. Member
  /// pointers have no pointee type.
  virtual Type *GetPointeeType() const { return nullptr; }

  /// True for a pointer to a non-static member function (`R (C::*)(Args...)`).
  virtual bool IsMemberFunctionPointer() const { return false; }

  /// True for a pointer to a non-static data member (`T C::*`).
  virtual bool IsMemberDataPointer() const { return false; }

  /// If the children of this pointer or reference are the members of its
  /// pointee (instead of a single dereferenced child), returns the pointee.
  /// Otherwise returns null.
  ///
  /// This is primarily used for preserving existing behaviour for Objective-C
  /// interfaces.
  virtual Type *GetTransparentChildPointee() { return nullptr; }

  /// If members can be looked up by name through this pointer or reference
  /// (`ptr->member`, `ref.member`), returns the pointee. Otherwise returns
  /// null.
  virtual Type *GetNamedMemberPointee() { return nullptr; }

  /// If the dynamic type of the value of this pointer or reference may differ
  /// from its static pointee type, returns the pointee. Otherwise returns null.
  ///
  /// This is the case for pointers to Objective-C objects (if \p check
  /// contains DynamicTypeCheck::ObjC), pointers or references to dynamic C++
  /// classes (if \p check contains DynamicTypeCheck::CPlusPlus) and `void *`.
  ///
  /// \p complete is called to request completion of a record whose dynamic-ness
  /// is only known after completion.
  virtual Type *
  GetDynamicTypeCandidate(DynamicTypeCheck check,
                          llvm::function_ref<void(Type *)> complete) {
    return nullptr;
  }

  /// True if this is a C++ class with a vtable pointer. \p complete is called
  /// first to requestion completion of the class.
  virtual bool IsDynamicClass(llvm::function_ref<void(Type *)> complete) {
    return false;
  }

  /// The element type of an array or vector, otherwise null.
  virtual Type *GetArrayElementType() const { return nullptr; }

  /// The number of elements of an array, or std::nullopt if this is not an
  /// array or the bound is unknown.
  virtual std::optional<uint64_t> GetNumElements() const {
    return std::nullopt;
  }

  /// True for GCC/Clang vector types.
  virtual bool IsVector() const { return false; }

  /// The UID of the DIE that describes this type, or LLDB_INVALID_UID. Only
  /// arrays store this so that variable-length bounds can be resolved.
  virtual uint64_t GetDIEUID() const { return UINT64_MAX; }

  /// True for a function type.
  virtual bool IsFunction() const { return false; }

  /// The return type of a function, otherwise null.
  virtual Type *GetReturnType() const { return nullptr; }

  /// The parameters of a function, excluding the implicit `this` parameter.
  virtual uint32_t GetNumParameters() const { return 0; }
  virtual Type *GetParameterAtIndex(uint32_t idx) const { return nullptr; }

  /// The underlying integer type of an enumeration, otherwise null.
  virtual Type *GetEnumerationIntegerType() const { return nullptr; }

  /// True for scoped enumerations (`enum class`).
  virtual bool IsScopedEnumeration() const { return false; }
  virtual llvm::ArrayRef<Enumerator> GetEnumerators() const { return {}; }

  /// True for C structs/unions, C++ classes and Objective-C interfaces.
  virtual bool IsRecord() const { return false; }

  /// True for Objective-C interfaces.
  virtual bool IsObjCInterface() const { return false; }

  virtual bool IsUnion() const { return false; }

  /// True if this record was declared with the `class` keyword.
  virtual bool IsClassKeyword() const { return false; }

  /// True for anonymous structs and unions (`struct { int y; };`).
  virtual bool IsAnonymousStructOrUnion() const { return false; }

  virtual uint32_t GetNumFields() const { return 0; }
  virtual const Field *GetFieldAtIndex(uint32_t idx) const { return nullptr; }

  /// The direct base classes of a C++ class.
  virtual uint32_t GetNumBaseClasses() const { return 0; }
  virtual const BaseClass *GetBaseClassAtIndex(uint32_t idx) const {
    return nullptr;
  }

  /// True if this is a C++ class that has a vtable.
  virtual bool IsPolymorphic() const { return false; }

  /// True for class template specializations.
  virtual bool IsTemplateInstantiation() const { return false; }
  virtual uint32_t GetNumTemplateArguments() const { return 0; }
  virtual const TemplateArgument *
  GetTemplateArgumentAtIndex(uint32_t idx) const {
    return nullptr;
  }

  /// True if the member functions (or Objective-C methods) of this record have
  /// been parsed.
  ///
  /// These are parsed lazily, so a complete record might not have its member
  /// functions parsed yet.
  virtual bool AreMemberFunctionsParsed() const { return true; }

  virtual uint32_t GetNumMemberFunctions() const { return 0; }
  virtual const MemberFunction *GetMemberFunctionAtIndex(uint32_t idx) const {
    return nullptr;
  }

  /// The methods of an Objective-C interface.
  virtual uint32_t GetNumObjCMethods() const { return 0; }
  virtual const ObjCMethod *GetObjCMethodAtIndex(uint32_t idx) const {
    return nullptr;
  }

  virtual uint32_t GetNumStaticDataMembers() const { return 0; }
  virtual const StaticDataMember *
  GetStaticDataMemberAtIndex(uint32_t idx) const {
    return nullptr;
  }

  /// Returns the type with the given unqualified name that is declared
  /// directly inside this record, or null if there is no such type.
  virtual Type *GetNestedTypeWithName(llvm::StringRef name) const {
    return nullptr;
  }

  /// The number of types declared directly inside this record.
  virtual uint32_t GetNumNestedTypes() const { return 0; }

  /// If this record is a homogeneous floating-point or vector aggregate,
  /// returns the type of its fields and sets \p num_fields to the number of
  /// fields. Otherwise returns null and sets \p num_fields to 0. Expects a
  /// complete record.
  virtual Type *GetHomogeneousAggregateBase(uint32_t &num_fields) const {
    num_fields = 0;
    return nullptr;
  }

  /// True if this type is a record that has fields, either directly or in one
  /// of its base classes.
  ///
  /// Incomplete records are treated as having fields. \p complete is called on
  /// each record before it is inspected.
  bool HasFields(llvm::function_ref<void(Type *)> complete);

private:
  // Set once when the type is added to its Context.
  friend class Context;
  void SetOwningContext(Context &context) { m_context = &context; }

  Context *m_context = nullptr;
};

} // namespace clike
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_TYPE_H
