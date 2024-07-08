//===-- LibCxxUnorderedMap.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibCxx.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace lldb_private {
namespace formatters {
class LibcxxStdUnorderedMapSyntheticFrontEnd
    : public SyntheticChildrenFrontEnd {
public:
  LibcxxStdUnorderedMapSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~LibcxxStdUnorderedMapSyntheticFrontEnd() override = default;

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(ConstString name) override;

private:
  CompilerType m_element_type;
  CompilerType m_node_type;
  ValueObject *m_tree = nullptr;
  size_t m_num_elements = 0;
  ValueObject *m_next_element = nullptr;
  std::vector<std::pair<ValueObject *, uint64_t>> m_elements_cache;
};

/// Formats libcxx's std::unordered_map iterators
///
/// In raw form a std::unordered_map::iterator is represented as follows:
///
/// (lldb) var it --raw --ptr-depth 1
/// (std::__1::__hash_map_iterator<
///    std::__1::__hash_iterator<
///      std::__1::__hash_node<
///        std::__1::__hash_value_type<
///            std::__1::basic_string<char, std::__1::char_traits<char>,
///            std::__1::allocator<char> >, std::__1::basic_string<char,
///            std::__1::char_traits<char>, std::__1::allocator<char> > >,
///        void *> *> >)
///  it = {
///   __i_ = {
///     __node_ = 0x0000600001700040 {
///       __next_ = 0x0000600001704000
///     }
///   }
/// }
class LibCxxUnorderedMapIteratorSyntheticFrontEnd
    : public SyntheticChildrenFrontEnd {
public:
  LibCxxUnorderedMapIteratorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~LibCxxUnorderedMapIteratorSyntheticFrontEnd() override = default;

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(ConstString name) override;

private:
  ValueObject *m_iter_ptr = nullptr; ///< Held, not owned. Child of iterator
                                     ///< ValueObject supplied at construction.

  lldb::ValueObjectSP m_pair_sp; ///< ValueObject for the key/value pair
                                 ///< that the iterator currently points
                                 ///< to.
};

} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    LibcxxStdUnorderedMapSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type(),
      m_elements_cache() {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t> lldb_private::formatters::
    LibcxxStdUnorderedMapSyntheticFrontEnd::CalculateNumChildren() {
  return m_num_elements;
}

static void consumeInlineNamespace(llvm::StringRef &name) {
  // Delete past an inline namespace, if any: __[a-zA-Z0-9_]+::
  auto scratch = name;
  if (scratch.consume_front("__") && std::isalnum(scratch[0])) {
    scratch = scratch.drop_while([](char c) { return std::isalnum(c); });
    if (scratch.consume_front("::")) {
      // Successfully consumed a namespace.
      name = scratch;
    }
  }
}

static bool isStdTemplate(ConstString type_name, llvm::StringRef type) {
  llvm::StringRef name = type_name.GetStringRef();
  // The type name may be prefixed with `std::__<inline-namespace>::`.
  if (name.consume_front("std::"))
    consumeInlineNamespace(name);
  return name.consume_front(type) && name.starts_with("<");
}

static bool isUnorderedMap(ConstString type_name) {
  return isStdTemplate(type_name, "unordered_map") ||
         isStdTemplate(type_name, "unordered_multimap");
}

lldb::ValueObjectSP lldb_private::formatters::
    LibcxxStdUnorderedMapSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx >= CalculateNumChildrenIgnoringErrors())
    return lldb::ValueObjectSP();
  if (m_tree == nullptr)
    return lldb::ValueObjectSP();

  while (idx >= m_elements_cache.size()) {
    if (m_next_element == nullptr)
      return lldb::ValueObjectSP();

    Status error;
    ValueObjectSP node_sp = m_next_element->Dereference(error);
    if (!node_sp || error.Fail())
      return lldb::ValueObjectSP();

    ValueObjectSP value_sp = node_sp->GetChildMemberWithName("__value_");
    ValueObjectSP hash_sp = node_sp->GetChildMemberWithName("__hash_");
    if (!hash_sp || !value_sp) {
      if (!m_element_type) {
        auto p1_sp = m_backend.GetChildAtNamePath({"__table_", "__p1_"});
        if (!p1_sp)
          return nullptr;

        ValueObjectSP first_sp = GetFirstValueOfLibCXXCompressedPair(*p1_sp);
        if (!first_sp)
          return nullptr;

        m_element_type = first_sp->GetCompilerType();
        m_element_type = m_element_type.GetTypeTemplateArgument(0);
        m_element_type = m_element_type.GetPointeeType();
        m_node_type = m_element_type;
        m_element_type = m_element_type.GetTypeTemplateArgument(0);
        // This synthetic provider is used for both unordered_(multi)map and
        // unordered_(multi)set. For unordered_map, the element type has an
        // additional type layer, an internal struct (`__hash_value_type`)
        // that wraps a std::pair. Peel away the internal wrapper type - whose
        // structure is of no value to users, to expose the std::pair. This
        // matches the structure returned by the std::map synthetic provider.
        if (isUnorderedMap(m_backend.GetTypeName())) {
          std::string name;
          CompilerType field_type = m_element_type.GetFieldAtIndex(
              0, name, nullptr, nullptr, nullptr);
          CompilerType actual_type = field_type.GetTypedefedType();
          if (isStdTemplate(actual_type.GetTypeName(), "pair"))
            m_element_type = actual_type;
        }
      }
      if (!m_node_type)
        return nullptr;
      node_sp = m_next_element->Cast(m_node_type.GetPointerType())
              ->Dereference(error);
      if (!node_sp || error.Fail())
          return nullptr;

      hash_sp = node_sp->GetChildMemberWithName("__hash_");
      if (!hash_sp)
        return nullptr;

      value_sp = node_sp->GetChildMemberWithName("__value_");
      if (!value_sp) {
        // clang-format off
        // Since D101206 (ba79fb2e1f), libc++ wraps the `__value_` in an
        // anonymous union.
        // Child 0: __hash_node_base base class
        // Child 1: __hash_
        // Child 2: anonymous union
        // clang-format on
        auto anon_union_sp = node_sp->GetChildAtIndex(2);
        if (!anon_union_sp)
          return nullptr;

        value_sp = anon_union_sp->GetChildMemberWithName("__value_");
        if (!value_sp)
          return nullptr;
      }
    }
    m_elements_cache.push_back(
        {value_sp.get(), hash_sp->GetValueAsUnsigned(0)});
    m_next_element = node_sp->GetChildMemberWithName("__next_").get();
    if (!m_next_element || m_next_element->GetValueAsUnsigned(0) == 0)
      m_next_element = nullptr;
  }

  std::pair<ValueObject *, uint64_t> val_hash = m_elements_cache[idx];
  if (!val_hash.first)
    return lldb::ValueObjectSP();
  StreamString stream;
  stream.Printf("[%" PRIu64 "]", (uint64_t)idx);
  DataExtractor data;
  Status error;
  val_hash.first->GetData(data, error);
  if (error.Fail())
    return lldb::ValueObjectSP();
  const bool thread_and_frame_only_if_stopped = true;
  ExecutionContext exe_ctx = val_hash.first->GetExecutionContextRef().Lock(
      thread_and_frame_only_if_stopped);
  return CreateValueObjectFromData(stream.GetString(), data, exe_ctx,
                                   m_element_type);
}

lldb::ChildCacheState
lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::Update() {
  m_num_elements = 0;
  m_next_element = nullptr;
  m_elements_cache.clear();
  ValueObjectSP table_sp = m_backend.GetChildMemberWithName("__table_");
  if (!table_sp)
    return lldb::ChildCacheState::eRefetch;

  ValueObjectSP p2_sp = table_sp->GetChildMemberWithName("__p2_");
  if (!p2_sp)
    return lldb::ChildCacheState::eRefetch;

  ValueObjectSP num_elements_sp = GetFirstValueOfLibCXXCompressedPair(*p2_sp);
  if (!num_elements_sp)
    return lldb::ChildCacheState::eRefetch;

  ValueObjectSP p1_sp = table_sp->GetChildMemberWithName("__p1_");
  if (!p1_sp)
    return lldb::ChildCacheState::eRefetch;

  ValueObjectSP value_sp = GetFirstValueOfLibCXXCompressedPair(*p1_sp);
  if (!value_sp)
    return lldb::ChildCacheState::eRefetch;

  m_tree = value_sp->GetChildMemberWithName("__next_").get();
  if (m_tree == nullptr)
    return lldb::ChildCacheState::eRefetch;

  m_num_elements = num_elements_sp->GetValueAsUnsigned(0);

  if (m_num_elements > 0)
    m_next_element = m_tree;

  return lldb::ChildCacheState::eRefetch;
}

bool lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    MightHaveChildren() {
  return true;
}

size_t lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  return ExtractIndexFromString(name.GetCString());
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new LibcxxStdUnorderedMapSyntheticFrontEnd(valobj_sp)
                    : nullptr);
}

lldb_private::formatters::LibCxxUnorderedMapIteratorSyntheticFrontEnd::
    LibCxxUnorderedMapIteratorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  if (valobj_sp)
    Update();
}

lldb::ChildCacheState lldb_private::formatters::
    LibCxxUnorderedMapIteratorSyntheticFrontEnd::Update() {
  m_pair_sp.reset();
  m_iter_ptr = nullptr;

  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  TargetSP target_sp(valobj_sp->GetTargetSP());

  if (!target_sp)
    return lldb::ChildCacheState::eRefetch;

  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  auto exprPathOptions = ValueObject::GetValueForExpressionPathOptions()
                             .DontCheckDotVsArrowSyntax()
                             .SetSyntheticChildrenTraversal(
                                 ValueObject::GetValueForExpressionPathOptions::
                                     SyntheticChildrenTraversal::None);

  // This must be a ValueObject* because it is a child of the ValueObject we
  // are producing children for it if were a ValueObjectSP, we would end up
  // with a loop (iterator -> synthetic -> child -> parent == iterator) and
  // that would in turn leak memory by never allowing the ValueObjects to die
  // and free their memory.
  m_iter_ptr =
      valobj_sp
          ->GetValueForExpressionPath(".__i_.__node_", nullptr, nullptr,
                                      exprPathOptions, nullptr)
          .get();

  if (m_iter_ptr) {
    auto iter_child(valobj_sp->GetChildMemberWithName("__i_"));
    if (!iter_child) {
      m_iter_ptr = nullptr;
      return lldb::ChildCacheState::eRefetch;
    }

    CompilerType node_type(iter_child->GetCompilerType()
                               .GetTypeTemplateArgument(0)
                               .GetPointeeType());

    CompilerType pair_type(node_type.GetTypeTemplateArgument(0));

    std::string name;
    uint64_t bit_offset_ptr;
    uint32_t bitfield_bit_size_ptr;
    bool is_bitfield_ptr;

    pair_type = pair_type.GetFieldAtIndex(
        0, name, &bit_offset_ptr, &bitfield_bit_size_ptr, &is_bitfield_ptr);
    if (!pair_type) {
      m_iter_ptr = nullptr;
      return lldb::ChildCacheState::eRefetch;
    }

    uint64_t addr = m_iter_ptr->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    m_iter_ptr = nullptr;

    if (addr == 0 || addr == LLDB_INVALID_ADDRESS)
      return lldb::ChildCacheState::eRefetch;

    auto ts = pair_type.GetTypeSystem();
    auto ast_ctx = ts.dyn_cast_or_null<TypeSystemClang>();
    if (!ast_ctx)
      return lldb::ChildCacheState::eRefetch;

    // Mimick layout of std::__hash_iterator::__node_ and read it in
    // from process memory.
    //
    // The following shows the contiguous block of memory:
    //
    //         +-----------------------------+ class __hash_node_base
    // __node_ | __next_pointer __next_;     |
    //         +-----------------------------+ class __hash_node
    //         | size_t __hash_;             |
    //         | __node_value_type __value_; | <<< our key/value pair
    //         +-----------------------------+
    //
    CompilerType tree_node_type = ast_ctx->CreateStructForIdentifier(
        llvm::StringRef(),
        {{"__next_",
          ast_ctx->GetBasicType(lldb::eBasicTypeVoid).GetPointerType()},
         {"__hash_", ast_ctx->GetBasicType(lldb::eBasicTypeUnsignedLongLong)},
         {"__value_", pair_type}});
    std::optional<uint64_t> size = tree_node_type.GetByteSize(nullptr);
    if (!size)
      return lldb::ChildCacheState::eRefetch;
    WritableDataBufferSP buffer_sp(new DataBufferHeap(*size, 0));
    ProcessSP process_sp(target_sp->GetProcessSP());
    Status error;
    process_sp->ReadMemory(addr, buffer_sp->GetBytes(),
                           buffer_sp->GetByteSize(), error);
    if (error.Fail())
      return lldb::ChildCacheState::eRefetch;
    DataExtractor extractor(buffer_sp, process_sp->GetByteOrder(),
                            process_sp->GetAddressByteSize());
    auto pair_sp = CreateValueObjectFromData(
        "pair", extractor, valobj_sp->GetExecutionContextRef(), tree_node_type);
    if (pair_sp)
      m_pair_sp = pair_sp->GetChildAtIndex(2);
  }

  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<uint32_t> lldb_private::formatters::
    LibCxxUnorderedMapIteratorSyntheticFrontEnd::CalculateNumChildren() {
  return 2;
}

lldb::ValueObjectSP lldb_private::formatters::
    LibCxxUnorderedMapIteratorSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (m_pair_sp)
    return m_pair_sp->GetChildAtIndex(idx);
  return lldb::ValueObjectSP();
}

bool lldb_private::formatters::LibCxxUnorderedMapIteratorSyntheticFrontEnd::
    MightHaveChildren() {
  return true;
}

size_t lldb_private::formatters::LibCxxUnorderedMapIteratorSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  if (name == "first")
    return 0;
  if (name == "second")
    return 1;
  return UINT32_MAX;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibCxxUnorderedMapIteratorSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new LibCxxUnorderedMapIteratorSyntheticFrontEnd(valobj_sp)
                    : nullptr);
}
