//===-- LibStdcpp.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibStdcpp.h"
#include "LibCxx.h"

#include "Plugins/Language/CPlusPlus/Generic.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/VectorIterator.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

class LibstdcppMapIteratorSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
  /*
   (std::_Rb_tree_iterator<std::pair<const int, std::basic_string<char,
   std::char_traits<char>, std::allocator<char> > > >) ibeg = {
   (_Base_ptr) _M_node = 0x0000000100103910 {
   (std::_Rb_tree_color) _M_color = _S_black
   (std::_Rb_tree_node_base::_Base_ptr) _M_parent = 0x00000001001038c0
   (std::_Rb_tree_node_base::_Base_ptr) _M_left = 0x0000000000000000
   (std::_Rb_tree_node_base::_Base_ptr) _M_right = 0x0000000000000000
   }
   }
   */

public:
  explicit LibstdcppMapIteratorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  ExecutionContextRef m_exe_ctx_ref;
  lldb::addr_t m_pair_address = 0;
  CompilerType m_pair_type;
  lldb::ValueObjectSP m_pair_sp;
};

class LibStdcppSharedPtrSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  explicit LibStdcppSharedPtrSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:

  // The lifetime of a ValueObject and all its derivative ValueObjects
  // (children, clones, etc.) is managed by a ClusterManager. These
  // objects are only destroyed when every shared pointer to any of them
  // is destroyed, so we must not store a shared pointer to any ValueObject
  // derived from our backend ValueObject (since we're in the same cluster).
  ValueObject *m_ptr_obj = nullptr; // Underlying pointer (held, not owned)
};

} // end of anonymous namespace

LibstdcppMapIteratorSyntheticFrontEnd::LibstdcppMapIteratorSyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_exe_ctx_ref(), m_pair_type(),
      m_pair_sp() {
  if (valobj_sp)
    Update();
}

lldb::ChildCacheState LibstdcppMapIteratorSyntheticFrontEnd::Update() {
  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  TargetSP target_sp(valobj_sp->GetTargetSP());

  if (!target_sp)
    return lldb::ChildCacheState::eRefetch;

  bool is_64bit = (target_sp->GetArchitecture().GetAddressByteSize() == 8);

  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;
  m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();

  ValueObjectSP _M_node_sp(valobj_sp->GetChildMemberWithName("_M_node"));
  if (!_M_node_sp)
    return lldb::ChildCacheState::eRefetch;

  m_pair_address = _M_node_sp->GetValueAsUnsigned(0);
  if (m_pair_address == 0)
    return lldb::ChildCacheState::eRefetch;

  m_pair_address += (is_64bit ? 32 : 16);

  CompilerType my_type(valobj_sp->GetCompilerType());
  if (my_type.GetNumTemplateArguments() >= 1) {
    CompilerType pair_type = my_type.GetTypeTemplateArgument(0);
    if (!pair_type)
      return lldb::ChildCacheState::eRefetch;
    m_pair_type = pair_type;
  } else
    return lldb::ChildCacheState::eRefetch;

  return lldb::ChildCacheState::eReuse;
}

llvm::Expected<uint32_t>
LibstdcppMapIteratorSyntheticFrontEnd::CalculateNumChildren() {
  return 2;
}

lldb::ValueObjectSP
LibstdcppMapIteratorSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (m_pair_address != 0 && m_pair_type) {
    if (!m_pair_sp)
      m_pair_sp = CreateValueObjectFromAddress("pair", m_pair_address,
                                               m_exe_ctx_ref, m_pair_type);
    if (m_pair_sp)
      return m_pair_sp->GetChildAtIndex(idx);
  }
  return lldb::ValueObjectSP();
}

llvm::Expected<size_t>
LibstdcppMapIteratorSyntheticFrontEnd::GetIndexOfChildWithName(
    ConstString name) {
  if (name == "first")
    return 0;
  if (name == "second")
    return 1;
  return llvm::createStringError("Type has no child named '%s'",
                                 name.AsCString());
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new LibstdcppMapIteratorSyntheticFrontEnd(valobj_sp)
                    : nullptr);
}

/*
 (lldb) fr var ibeg --ptr-depth 1
 (__gnu_cxx::__normal_iterator<int *, std::vector<int, std::allocator<int> > >)
 ibeg = {
 _M_current = 0x00000001001037a0 {
 *_M_current = 1
 }
 }
 */

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibStdcppVectorIteratorSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new VectorIteratorSyntheticFrontEnd(
                          valobj_sp, {ConstString("_M_current")})
                    : nullptr);
}

lldb_private::formatters::VectorIteratorSyntheticFrontEnd::
    VectorIteratorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp,
                                    llvm::ArrayRef<ConstString> item_names)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_exe_ctx_ref(),
      m_item_names(item_names), m_item_sp() {
  if (valobj_sp)
    Update();
}

lldb::ChildCacheState VectorIteratorSyntheticFrontEnd::Update() {
  m_item_sp.reset();

  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  ValueObjectSP item_ptr =
      formatters::GetChildMemberWithName(*valobj_sp, m_item_names);
  if (!item_ptr)
    return lldb::ChildCacheState::eRefetch;
  if (item_ptr->GetValueAsUnsigned(0) == 0)
    return lldb::ChildCacheState::eRefetch;
  Status err;
  m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
  m_item_sp = CreateValueObjectFromAddress(
      "item", item_ptr->GetValueAsUnsigned(0), m_exe_ctx_ref,
      item_ptr->GetCompilerType().GetPointeeType());
  if (err.Fail())
    m_item_sp.reset();
  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<uint32_t>
VectorIteratorSyntheticFrontEnd::CalculateNumChildren() {
  return 1;
}

lldb::ValueObjectSP
VectorIteratorSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx == 0)
    return m_item_sp;
  return lldb::ValueObjectSP();
}

llvm::Expected<size_t>
VectorIteratorSyntheticFrontEnd::GetIndexOfChildWithName(ConstString name) {
  if (name == "item")
    return 0;
  return llvm::createStringError("Type has no child named '%s'",
                                 name.AsCString());
}

bool lldb_private::formatters::LibStdcppStringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ValueObjectSP ptr = valobj.GetChildAtNamePath({"_M_dataplus", "_M_p"});
  if (!ptr || !ptr->GetError().Success())
    stream << "Summary Unavailable";
  else
    stream << ptr->GetSummaryAsCString();

  return true;
}

LibStdcppSharedPtrSyntheticFrontEnd::LibStdcppSharedPtrSyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t>
LibStdcppSharedPtrSyntheticFrontEnd::CalculateNumChildren() {
  return 1;
}

lldb::ValueObjectSP
LibStdcppSharedPtrSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (!m_ptr_obj)
    return nullptr;

  if (idx == 0)
    return m_ptr_obj->GetSP();

  if (idx == 1) {
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
      return nullptr;

    Status status;
    ValueObjectSP value_sp = m_ptr_obj->Dereference(status);
    if (status.Success())
      return value_sp;
  }
  return lldb::ValueObjectSP();
}

lldb::ChildCacheState LibStdcppSharedPtrSyntheticFrontEnd::Update() {
  auto backend = m_backend.GetSP();
  if (!backend)
    return lldb::ChildCacheState::eRefetch;

  auto valobj_sp = backend->GetNonSyntheticValue();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  auto ptr_obj_sp = valobj_sp->GetChildMemberWithName("_M_ptr");
  if (!ptr_obj_sp)
    return lldb::ChildCacheState::eRefetch;

  auto cast_ptr_sp = GetDesugaredSmartPointerValue(*ptr_obj_sp, *valobj_sp);
  if (!cast_ptr_sp)
    return lldb::ChildCacheState::eRefetch;

  m_ptr_obj = cast_ptr_sp->Clone(ConstString("pointer")).get();

  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t>
LibStdcppSharedPtrSyntheticFrontEnd::GetIndexOfChildWithName(ConstString name) {
  if (name == "pointer")
    return 0;

  if (name == "object" || name == "$$dereference$$")
    return 1;

  return llvm::createStringError("Type has no child named '%s'",
                                 name.AsCString());
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibStdcppSharedPtrSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new LibStdcppSharedPtrSyntheticFrontEnd(valobj_sp)
                    : nullptr);
}

bool lldb_private::formatters::LibStdcppSmartPointerSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ValueObjectSP valobj_sp(valobj.GetNonSyntheticValue());
  if (!valobj_sp)
    return false;

  ValueObjectSP ptr_sp(valobj_sp->GetChildMemberWithName("_M_ptr"));
  if (!ptr_sp)
    return false;

  DumpCxxSmartPtrPointerSummary(stream, *ptr_sp, options);

  ValueObjectSP pi_sp = valobj_sp->GetChildAtNamePath({"_M_refcount", "_M_pi"});
  if (!pi_sp)
    return false;

  bool success;
  uint64_t pi_addr = pi_sp->GetValueAsUnsigned(0, &success);
  // Empty control field. We're done.
  if (!success || pi_addr == 0)
    return true;

  int64_t shared_count = 0;
  if (auto count_sp = pi_sp->GetChildMemberWithName("_M_use_count")) {
    bool success;
    shared_count = count_sp->GetValueAsSigned(0, &success);
    if (!success)
      return false;

    stream.Printf(" strong=%" PRId64, shared_count);
  }

  // _M_weak_count is the number of weak references + (_M_use_count != 0).
  if (auto weak_count_sp = pi_sp->GetChildMemberWithName("_M_weak_count")) {
    bool success;
    int64_t count = weak_count_sp->GetValueAsUnsigned(0, &success);
    if (!success)
      return false;

    stream.Printf(" weak=%" PRId64, count - (shared_count != 0));
  }

  return true;
}

static uint64_t LibStdcppVariantNposValue(size_t index_byte_size) {
  switch (index_byte_size) {
  case 1:
    return 0xff;
  case 2:
    return 0xffff;
  default:
    return 0xffff'ffff;
  }
}

bool formatters::LibStdcppVariantSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ValueObjectSP valobj_sp = valobj.GetNonSyntheticValue();
  if (!valobj_sp)
    return false;

  ValueObjectSP index_obj = valobj_sp->GetChildMemberWithName("_M_index");
  ValueObjectSP data_obj = valobj_sp->GetChildMemberWithName("_M_u");
  if (!index_obj || !data_obj)
    return false;

  auto index_bytes = index_obj->GetByteSize();
  if (!index_bytes)
    return false;
  auto npos_value = LibStdcppVariantNposValue(*index_bytes);
  auto index = index_obj->GetValueAsUnsigned(0);
  if (index == npos_value) {
    stream.Printf(" No Value");
    return true;
  }

  auto variant_type =
      valobj_sp->GetCompilerType().GetCanonicalType().GetNonReferenceType();
  if (!variant_type)
    return false;
  if (index >= variant_type.GetNumTemplateArguments(true)) {
    stream.Printf(" <Invalid>");
    return true;
  }

  auto active_type = variant_type.GetTypeTemplateArgument(index, true);
  stream << " Active Type = " << active_type.GetDisplayTypeName() << " ";
  return true;
}
