//===-- MsvcStlTree.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include <cstdint>
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

// A Node looks as follows:
// struct _Tree_node {
//   _Tree_node *_Left;
//   _Tree_node *_Parent;
//   _Tree_node *_Right;
//   char _Color;
//   char _Isnil;         // true (!= 0) if head or nil node
//   value_type _Myval;
// };

namespace {

class MapEntry {
public:
  MapEntry() = default;
  explicit MapEntry(ValueObjectSP entry_sp) : m_entry_sp(entry_sp) {}
  explicit MapEntry(ValueObject *entry)
      : m_entry_sp(entry ? entry->GetSP() : ValueObjectSP()) {}

  ValueObjectSP left() const {
    if (!m_entry_sp)
      return m_entry_sp;
    return m_entry_sp->GetSyntheticChildAtOffset(
        0, m_entry_sp->GetCompilerType(), true);
  }

  ValueObjectSP right() const {
    if (!m_entry_sp)
      return m_entry_sp;
    return m_entry_sp->GetSyntheticChildAtOffset(
        2 * m_entry_sp->GetProcessSP()->GetAddressByteSize(),
        m_entry_sp->GetCompilerType(), true);
  }

  ValueObjectSP parent() const {
    if (!m_entry_sp)
      return m_entry_sp;
    return m_entry_sp->GetSyntheticChildAtOffset(
        m_entry_sp->GetProcessSP()->GetAddressByteSize(),
        m_entry_sp->GetCompilerType(), true);
  }

  uint64_t value() const {
    if (!m_entry_sp)
      return 0;
    return m_entry_sp->GetValueAsUnsigned(0);
  }

  bool is_nil() const {
    if (!m_entry_sp)
      return true;
    auto isnil_sp = m_entry_sp->GetChildMemberWithName("_Isnil");
    if (!isnil_sp)
      return true;
    return isnil_sp->GetValueAsUnsigned(1) != 0;
  }

  bool error() const {
    if (!m_entry_sp)
      return true;
    return m_entry_sp->GetError().Fail();
  }

  bool is_nullptr() const { return (value() == 0); }

  ValueObjectSP GetEntry() const { return m_entry_sp; }

  void SetEntry(ValueObjectSP entry) { m_entry_sp = entry; }

  bool operator==(const MapEntry &rhs) const {
    return (rhs.m_entry_sp.get() == m_entry_sp.get());
  }

private:
  ValueObjectSP m_entry_sp;
};

class MapIterator {
public:
  MapIterator(ValueObject *entry, size_t depth = 0)
      : m_entry(entry), m_max_depth(depth) {}

  MapIterator() = default;

  ValueObjectSP value() { return m_entry.GetEntry(); }

  ValueObjectSP advance(size_t count) {
    ValueObjectSP fail;
    if (m_error)
      return fail;
    size_t steps = 0;
    while (count > 0) {
      next();
      count--, steps++;
      if (m_error || m_entry.is_nullptr() || (steps > m_max_depth))
        return fail;
    }
    return m_entry.GetEntry();
  }

private:
  /// Mimicks _Tree_unchecked_const_iterator::operator++()
  void next() {
    if (m_entry.is_nullptr())
      return;
    MapEntry right(m_entry.right());
    if (!right.is_nil()) {
      m_entry = tree_min(std::move(right));
      return;
    }
    size_t steps = 0;
    MapEntry pnode(m_entry.parent());
    while (!pnode.is_nil() &&
           m_entry.value() == MapEntry(pnode.right()).value()) {
      m_entry = pnode;
      steps++;
      if (steps > m_max_depth) {
        m_entry = MapEntry();
        return;
      }
      pnode.SetEntry(m_entry.parent());
    }
    m_entry = std::move(pnode);
  }

  /// Mimicks MSVC STL's _Min() algorithm (finding the leftmost node in the
  /// subtree).
  MapEntry tree_min(MapEntry pnode) {
    if (pnode.is_nullptr())
      return MapEntry();
    MapEntry left(pnode.left());
    size_t steps = 0;
    while (!left.is_nil()) {
      if (left.error()) {
        m_error = true;
        return MapEntry();
      }
      pnode = left;
      left.SetEntry(pnode.left());
      steps++;
      if (steps > m_max_depth)
        return MapEntry();
    }
    return pnode;
  }

  MapEntry m_entry;
  size_t m_max_depth = 0;
  bool m_error = false;
};

} // namespace

namespace lldb_private {
namespace formatters {
class MsvcStlTreeSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlTreeSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~MsvcStlTreeSyntheticFrontEnd() override = default;

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  /// Returns the ValueObject for the _Tree_node at index \ref idx.
  ///
  /// \param[in] idx The child index that we're looking to get the value for.
  ///
  /// \param[in] max_depth The maximum search depth after which we stop trying
  ///                      to find the node for.
  ///
  /// \returns On success, returns the ValueObjectSP corresponding to the
  ///          _Tree_node's _Myval member.
  ///          On failure, nullptr is returned.
  ValueObjectSP GetValueAt(size_t idx, size_t max_depth);

  ValueObject *m_tree = nullptr;
  ValueObject *m_begin_node = nullptr;
  size_t m_count = UINT32_MAX;
  std::map<size_t, MapIterator> m_iterators;
};

class MsvcStlTreeIterSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlTreeIterSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp) {}

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    if (!m_inner_sp)
      return 0;
    return m_inner_sp->GetNumChildren();
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_inner_sp)
      return nullptr;
    return m_inner_sp->GetChildAtIndex(idx);
  }

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_inner_sp)
      return llvm::createStringError("There are no children.");
    return m_inner_sp->GetIndexOfChildWithName(name);
  }

  lldb::ValueObjectSP GetSyntheticValue() override { return m_inner_sp; }

private:
  ValueObjectSP m_inner_sp;
};

} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::MsvcStlTreeSyntheticFrontEnd::
    MsvcStlTreeSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t>
lldb_private::formatters::MsvcStlTreeSyntheticFrontEnd::CalculateNumChildren() {
  if (m_count != UINT32_MAX)
    return m_count;

  if (m_tree == nullptr)
    return 0;

  if (auto node_sp = m_tree->GetChildMemberWithName("_Mysize")) {
    m_count = node_sp->GetValueAsUnsigned(0);
    return m_count;
  }

  return llvm::createStringError("Failed to read size.");
}

ValueObjectSP
lldb_private::formatters::MsvcStlTreeSyntheticFrontEnd::GetValueAt(
    size_t idx, size_t max_depth) {
  MapIterator iterator(m_begin_node, max_depth);

  size_t advance_by = idx;
  if (idx > 0) {
    // If we have already created the iterator for the previous
    // index, we can start from there and advance by 1.
    auto cached_iterator = m_iterators.find(idx - 1);
    if (cached_iterator != m_iterators.end()) {
      iterator = cached_iterator->second;
      advance_by = 1;
    }
  }

  ValueObjectSP iterated_sp(iterator.advance(advance_by));
  if (!iterated_sp)
    // this tree is garbage - stop
    return nullptr;

  ValueObjectSP value_sp = iterated_sp->GetChildMemberWithName("_Myval");
  if (!value_sp)
    return nullptr;

  m_iterators[idx] = iterator;

  return value_sp;
}

lldb::ValueObjectSP
lldb_private::formatters::MsvcStlTreeSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  uint32_t num_children = CalculateNumChildrenIgnoringErrors();
  if (idx >= num_children)
    return nullptr;

  if (m_tree == nullptr || m_begin_node == nullptr)
    return nullptr;

  ValueObjectSP val_sp = GetValueAt(idx, /*max_depth=*/num_children);
  if (!val_sp) {
    // this will stop all future searches until an Update() happens
    m_tree = nullptr;
    return nullptr;
  }

  // at this point we have a valid pair
  // we need to copy current_sp into a new object otherwise we will end up with
  // all items named _Myval
  StreamString name;
  name.Printf("[%" PRIu64 "]", (uint64_t)idx);
  return val_sp->Clone(ConstString(name.GetString()));
}

lldb::ChildCacheState
lldb_private::formatters::MsvcStlTreeSyntheticFrontEnd::Update() {
  m_count = UINT32_MAX;
  m_tree = m_begin_node = nullptr;
  m_iterators.clear();
  m_tree =
      m_backend.GetChildAtNamePath({"_Mypair", "_Myval2", "_Myval2"}).get();
  if (!m_tree)
    return lldb::ChildCacheState::eRefetch;

  m_begin_node = m_tree->GetChildAtNamePath({"_Myhead", "_Left"}).get();

  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t>
lldb_private::formatters::MsvcStlTreeSyntheticFrontEnd::GetIndexOfChildWithName(
    ConstString name) {
  auto optional_idx = formatters::ExtractIndexFromString(name.GetCString());
  if (!optional_idx) {
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  }
  return *optional_idx;
}

lldb::ChildCacheState MsvcStlTreeIterSyntheticFrontEnd::Update() {
  m_inner_sp = nullptr;
  auto node_sp = m_backend.GetChildMemberWithName("_Ptr");
  if (!node_sp)
    return lldb::eRefetch;

  MapEntry entry(node_sp.get());
  if (entry.is_nil())
    return lldb::eRefetch; // end

  m_inner_sp = node_sp->GetChildMemberWithName("_Myval");
  return lldb::eRefetch;
}

bool formatters::IsMsvcStlTreeIter(ValueObject &valobj) {
  return valobj.GetChildMemberWithName("_Ptr") != nullptr;
}

bool formatters::MsvcStlTreeIterSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  auto valobj_sp = valobj.GetNonSyntheticValue();
  if (!valobj_sp)
    return false;
  auto node_sp = valobj_sp->GetChildMemberWithName("_Ptr");
  if (!node_sp)
    return false;

  MapEntry entry(node_sp.get());
  if (entry.is_nil()) {
    stream.Printf("end");
    return true;
  }

  auto value_sp = node_sp->GetChildMemberWithName("_Myval");
  if (!value_sp)
    return false;

  auto *summary = value_sp->GetSummaryAsCString();
  if (summary)
    stream << summary;
  return true;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::MsvcStlTreeIterSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new MsvcStlTreeIterSyntheticFrontEnd(valobj_sp)
                    : nullptr);
}

bool formatters::IsMsvcStlMapLike(ValueObject &valobj) {
  return valobj.GetChildMemberWithName("_Mypair") != nullptr;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::MsvcStlMapLikeSyntheticFrontEndCreator(
    lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new MsvcStlTreeSyntheticFrontEnd(valobj_sp) : nullptr);
}
