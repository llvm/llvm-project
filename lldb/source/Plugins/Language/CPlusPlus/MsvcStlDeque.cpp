//===-- MsvcStlDeque.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

using namespace lldb;

namespace lldb_private {
namespace formatters {

class MsvcStlDequeSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlDequeSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  ValueObject *m_map = nullptr;
  ExecutionContextRef m_exe_ctx_ref;

  size_t m_block_size = 0;
  size_t m_offset = 0;
  size_t m_map_size = 0;

  size_t m_element_size = 0;
  CompilerType m_element_type;

  uint32_t m_size = 0;
};

} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::MsvcStlDequeSyntheticFrontEnd::
    MsvcStlDequeSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t> lldb_private::formatters::
    MsvcStlDequeSyntheticFrontEnd::CalculateNumChildren() {
  if (!m_map)
    return llvm::createStringError("Failed to read size");
  return m_size;
}

lldb::ValueObjectSP
lldb_private::formatters::MsvcStlDequeSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  if (idx >= m_size || !m_map)
    return nullptr;
  ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
  if (!process_sp)
    return nullptr;

  // _EEN_DS = _Block_size
  // _Map[(($i + _Myoff) / _EEN_DS) % _Mapsize][($i + _Myoff) % _EEN_DS]
  size_t first_idx = ((idx + m_offset) / m_block_size) % m_map_size;
  lldb::addr_t first_address = m_map->GetValueAsUnsigned(0) +
                               first_idx * process_sp->GetAddressByteSize();

  Status err;
  lldb::addr_t second_base =
      process_sp->ReadPointerFromMemory(first_address, err);
  if (err.Fail())
    return nullptr;

  size_t second_idx = (idx + m_offset) % m_block_size;
  size_t second_address = second_base + second_idx * m_element_size;

  StreamString name;
  name.Printf("[%" PRIu64 "]", (uint64_t)idx);
  return CreateValueObjectFromAddress(name.GetString(), second_address,
                                      m_backend.GetExecutionContextRef(),
                                      m_element_type);
}

lldb::ChildCacheState
lldb_private::formatters::MsvcStlDequeSyntheticFrontEnd::Update() {
  m_size = 0;
  m_map = nullptr;
  m_element_type.Clear();

  ValueObjectSP storage_sp =
      m_backend.GetChildAtNamePath({"_Mypair", "_Myval2"});
  if (!storage_sp)
    return lldb::eRefetch;

  CompilerType deque_type = m_backend.GetCompilerType();
  if (!deque_type)
    return lldb::eRefetch;

  CompilerDecl block_size_decl =
      deque_type.GetStaticFieldWithName("_Block_size");
  if (!block_size_decl)
    return lldb::eRefetch;
  Scalar block_size = block_size_decl.GetConstantValue();
  if (!block_size.IsValid())
    return lldb::eRefetch;

  ValueObjectSP offset_sp = storage_sp->GetChildMemberWithName("_Myoff");
  ValueObjectSP map_size_sp = storage_sp->GetChildMemberWithName("_Mapsize");
  ValueObjectSP map_sp = storage_sp->GetChildMemberWithName("_Map");
  ValueObjectSP size_sp = storage_sp->GetChildMemberWithName("_Mysize");
  if (!offset_sp || !map_size_sp || !map_sp || !size_sp)
    return lldb::eRefetch;

  bool ok = false;
  uint64_t offset = offset_sp->GetValueAsUnsigned(0, &ok);
  if (!ok)
    return lldb::eRefetch;

  uint64_t map_size = map_size_sp->GetValueAsUnsigned(0, &ok);
  if (!ok)
    return lldb::eRefetch;

  uint64_t size = size_sp->GetValueAsUnsigned(0, &ok);
  if (!ok)
    return lldb::eRefetch;

  CompilerType element_type = deque_type.GetTypeTemplateArgument(0);
  if (!element_type) {
    // PDB doesn't have the template type, so use the type of _Map (T**).
    element_type = map_sp->GetCompilerType().GetPointeeType().GetPointeeType();
    if (!element_type)
      return lldb::eRefetch;
  }
  auto element_size = element_type.GetByteSize(nullptr);
  if (!element_size)
    return lldb::eRefetch;

  m_map = map_sp.get();
  m_exe_ctx_ref = m_backend.GetExecutionContextRef();
  m_block_size = block_size.ULongLong();
  m_offset = offset;
  m_map_size = map_size;
  m_element_size = *element_size;
  m_element_type = element_type;
  m_size = size;
  return lldb::eRefetch;
}

llvm::Expected<size_t> lldb_private::formatters::MsvcStlDequeSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  if (!m_map)
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  if (auto optional_idx = ExtractIndexFromString(name.GetCString()))
    return *optional_idx;

  return llvm::createStringError("Type has no child named '%s'",
                                 name.AsCString());
}

bool lldb_private::formatters::IsMsvcStlDeque(ValueObject &valobj) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue())
    return valobj_sp->GetChildMemberWithName("_Mypair") != nullptr;
  return false;
}

lldb_private::SyntheticChildrenFrontEnd *
lldb_private::formatters::MsvcStlDequeSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return new MsvcStlDequeSyntheticFrontEnd(valobj_sp);
}
