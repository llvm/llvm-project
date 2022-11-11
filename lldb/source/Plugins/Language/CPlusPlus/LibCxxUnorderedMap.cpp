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

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

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
} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    LibcxxStdUnorderedMapSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type(),
      m_elements_cache() {
  if (valobj_sp)
    Update();
}

size_t lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    CalculateNumChildren() {
  return m_num_elements;
}

static bool isStdTemplate(ConstString type_name, llvm::StringRef type) {
  llvm::StringRef name = type_name.GetStringRef();
  // The type name may or may not be prefixed with `std::` or `std::__1::`.
  if (name.consume_front("std::"))
    name.consume_front("__1::");
  return name.consume_front(type) && name.startswith("<");
}

static bool isUnorderedMap(ConstString type_name) {
  return isStdTemplate(type_name, "unordered_map") ||
         isStdTemplate(type_name, "unordered_multimap");
}

lldb::ValueObjectSP lldb_private::formatters::
    LibcxxStdUnorderedMapSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  if (idx >= CalculateNumChildren())
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

    ValueObjectSP value_sp =
        node_sp->GetChildMemberWithName(ConstString("__value_"), true);
    ValueObjectSP hash_sp =
        node_sp->GetChildMemberWithName(ConstString("__hash_"), true);
    if (!hash_sp || !value_sp) {
      if (!m_element_type) {
        auto p1_sp = m_backend.GetChildAtNamePath({ConstString("__table_"),
                                                   ConstString("__p1_")});
        if (!p1_sp)
          return nullptr;

        ValueObjectSP first_sp = nullptr;
        switch (p1_sp->GetCompilerType().GetNumDirectBaseClasses()) {
        case 1:
          // Assume a pre llvm r300140 __compressed_pair implementation:
          first_sp = p1_sp->GetChildMemberWithName(ConstString("__first_"),
                                                   true);
          break;
        case 2: {
          // Assume a post llvm r300140 __compressed_pair implementation:
          ValueObjectSP first_elem_parent_sp =
            p1_sp->GetChildAtIndex(0, true);
          first_sp = p1_sp->GetChildMemberWithName(ConstString("__value_"),
                                                   true);
          break;
        }
        default:
          return nullptr;
        }

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
      node_sp = node_sp->Cast(m_node_type);
      value_sp = node_sp->GetChildMemberWithName(ConstString("__value_"), true);
      hash_sp = node_sp->GetChildMemberWithName(ConstString("__hash_"), true);
      if (!value_sp || !hash_sp)
        return nullptr;
    }
    m_elements_cache.push_back(
        {value_sp.get(), hash_sp->GetValueAsUnsigned(0)});
    m_next_element =
        node_sp->GetChildMemberWithName(ConstString("__next_"), true).get();
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

bool lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    Update() {
  m_num_elements = 0;
  m_next_element = nullptr;
  m_elements_cache.clear();
  ValueObjectSP table_sp =
      m_backend.GetChildMemberWithName(ConstString("__table_"), true);
  if (!table_sp)
    return false;

  ValueObjectSP p2_sp = table_sp->GetChildMemberWithName(
    ConstString("__p2_"), true);
  ValueObjectSP num_elements_sp = nullptr;
  llvm::SmallVector<ConstString, 3> next_path;
  switch (p2_sp->GetCompilerType().GetNumDirectBaseClasses()) {
  case 1:
    // Assume a pre llvm r300140 __compressed_pair implementation:
    num_elements_sp = p2_sp->GetChildMemberWithName(
      ConstString("__first_"), true);
    next_path.append({ConstString("__p1_"), ConstString("__first_"),
                      ConstString("__next_")});
    break;
  case 2: {
    // Assume a post llvm r300140 __compressed_pair implementation:
    ValueObjectSP first_elem_parent = p2_sp->GetChildAtIndex(0, true);
    num_elements_sp = first_elem_parent->GetChildMemberWithName(
      ConstString("__value_"), true);
    next_path.append({ConstString("__p1_"), ConstString("__value_"),
                      ConstString("__next_")});
    break;
  }
  default:
    return false;
  }

  if (!num_elements_sp)
    return false;

  m_tree = table_sp->GetChildAtNamePath(next_path).get();
  if (m_tree == nullptr)
    return false;

  m_num_elements = num_elements_sp->GetValueAsUnsigned(0);

  if (m_num_elements > 0)
    m_next_element =
        table_sp->GetChildAtNamePath(next_path).get();
  return false;
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
