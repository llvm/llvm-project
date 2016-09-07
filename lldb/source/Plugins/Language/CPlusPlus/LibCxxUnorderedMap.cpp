//===-- LibCxxUnorderedMap.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "LibCxx.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Target.h"

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

  size_t GetIndexOfChildWithName(const ConstString &name) override;

private:
  ValueObject *m_tree;
  size_t m_num_elements;
  ValueObject *m_next_element;
  std::vector<std::pair<ValueObject *, uint64_t>> m_elements_cache;
};
} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    LibcxxStdUnorderedMapSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_tree(nullptr), m_num_elements(0),
      m_next_element(nullptr), m_elements_cache() {
  if (valobj_sp)
    Update();
}

size_t lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    CalculateNumChildren() {
  if (m_num_elements != UINT32_MAX)
    return m_num_elements;
  return 0;
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

    Error error;
    ValueObjectSP node_sp = m_next_element->Dereference(error);
    if (!node_sp || error.Fail())
      return lldb::ValueObjectSP();

    ValueObjectSP value_sp =
        node_sp->GetChildMemberWithName(ConstString("__value_"), true);
    ValueObjectSP hash_sp =
        node_sp->GetChildMemberWithName(ConstString("__hash_"), true);
    if (!hash_sp || !value_sp)
      return lldb::ValueObjectSP();
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
  Error error;
  val_hash.first->GetData(data, error);
  if (error.Fail())
    return lldb::ValueObjectSP();
  const bool thread_and_frame_only_if_stopped = true;
  ExecutionContext exe_ctx = val_hash.first->GetExecutionContextRef().Lock(
      thread_and_frame_only_if_stopped);
  return CreateValueObjectFromData(stream.GetData(), data, exe_ctx,
                                   val_hash.first->GetCompilerType());
}

bool lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    Update() {
  m_num_elements = UINT32_MAX;
  m_next_element = nullptr;
  m_elements_cache.clear();
  ValueObjectSP table_sp =
      m_backend.GetChildMemberWithName(ConstString("__table_"), true);
  if (!table_sp)
    return false;
  ValueObjectSP num_elements_sp = table_sp->GetChildAtNamePath(
      {ConstString("__p2_"), ConstString("__first_")});
  if (!num_elements_sp)
    return false;
  m_num_elements = num_elements_sp->GetValueAsUnsigned(0);
  m_tree =
      table_sp
          ->GetChildAtNamePath({ConstString("__p1_"), ConstString("__first_"),
                                ConstString("__next_")})
          .get();
  if (m_num_elements > 0)
    m_next_element =
        table_sp
            ->GetChildAtNamePath({ConstString("__p1_"), ConstString("__first_"),
                                  ConstString("__next_")})
            .get();
  return false;
}

bool lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    MightHaveChildren() {
  return true;
}

size_t lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEnd::
    GetIndexOfChildWithName(const ConstString &name) {
  return ExtractIndexFromString(name.GetCString());
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new LibcxxStdUnorderedMapSyntheticFrontEnd(valobj_sp)
                    : nullptr);
}
