//===-- LibStdcppTuple.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibStdcpp.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Utility/ConstString.h"

#include <memory>
#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

class LibStdcppTupleSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  explicit LibStdcppTupleSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(ConstString name) override;

private:
  // The lifetime of a ValueObject and all its derivative ValueObjects
  // (children, clones, etc.) is managed by a ClusterManager. These
  // objects are only destroyed when every shared pointer to any of them
  // is destroyed, so we must not store a shared pointer to any ValueObject
  // derived from our backend ValueObject (since we're in the same cluster).
  std::vector<ValueObject*> m_members;
};

} // end of anonymous namespace

LibStdcppTupleSyntheticFrontEnd::LibStdcppTupleSyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  Update();
}

bool LibStdcppTupleSyntheticFrontEnd::Update() {
  m_members.clear();

  ValueObjectSP valobj_backend_sp = m_backend.GetSP();
  if (!valobj_backend_sp)
    return false;

  ValueObjectSP next_child_sp = valobj_backend_sp->GetNonSyntheticValue();
  while (next_child_sp != nullptr) {
    ValueObjectSP current_child = next_child_sp;
    next_child_sp = nullptr;

    size_t child_count = current_child->GetNumChildren();
    for (size_t i = 0; i < child_count; ++i) {
      ValueObjectSP child_sp = current_child->GetChildAtIndex(i, true);
      llvm::StringRef name_str = child_sp->GetName().GetStringRef();
      if (name_str.startswith("std::_Tuple_impl<")) {
        next_child_sp = child_sp;
      } else if (name_str.startswith("std::_Head_base<")) {
        ValueObjectSP value_sp =
            child_sp->GetChildMemberWithName(ConstString("_M_head_impl"), true);
        if (value_sp) {
          StreamString name;
          name.Printf("[%zd]", m_members.size());
          m_members.push_back(value_sp->Clone(ConstString(name.GetString())).get());
        }
      }
    }
  }

  return false;
}

bool LibStdcppTupleSyntheticFrontEnd::MightHaveChildren() { return true; }

lldb::ValueObjectSP
LibStdcppTupleSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  if (idx < m_members.size() && m_members[idx])
    return m_members[idx]->GetSP();
  return lldb::ValueObjectSP();
}

size_t LibStdcppTupleSyntheticFrontEnd::CalculateNumChildren() {
  return m_members.size();
}

size_t LibStdcppTupleSyntheticFrontEnd::GetIndexOfChildWithName(
    ConstString name) {
  return ExtractIndexFromString(name.GetCString());
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibStdcppTupleSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new LibStdcppTupleSyntheticFrontEnd(valobj_sp) : nullptr);
}
