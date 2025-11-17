//===-- GenericList.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibCxx.h"
#include "MsvcStl.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

enum class StlType {
  LibCxx,
  MsvcStl,
};

template <StlType Stl> class ListEntry {
public:
  ListEntry() = default;
  ListEntry(ValueObjectSP entry_sp) : m_entry_sp(std::move(entry_sp)) {}
  ListEntry(ValueObject *entry)
      : m_entry_sp(entry ? entry->GetSP() : ValueObjectSP()) {}

  uint64_t value() const {
    if (!m_entry_sp)
      return 0;
    return m_entry_sp->GetValueAsUnsigned(0);
  }

  ListEntry next();
  ListEntry prev();

  bool null() { return (value() == 0); }

  explicit operator bool() { return GetEntry() && !null(); }

  ValueObjectSP GetEntry() { return m_entry_sp; }

  void SetEntry(ValueObjectSP entry) { m_entry_sp = entry; }

  bool operator==(const ListEntry &rhs) const { return value() == rhs.value(); }

  bool operator!=(const ListEntry &rhs) const { return !(*this == rhs); }

private:
  ValueObjectSP m_entry_sp;
};

template <> ListEntry<StlType::LibCxx> ListEntry<StlType::LibCxx>::next() {
  if (!m_entry_sp)
    return ListEntry();
  return ListEntry(m_entry_sp->GetChildMemberWithName("__next_"));
}

template <> ListEntry<StlType::LibCxx> ListEntry<StlType::LibCxx>::prev() {
  if (!m_entry_sp)
    return ListEntry();
  return ListEntry(m_entry_sp->GetChildMemberWithName("__prev_"));
}

template <> ListEntry<StlType::MsvcStl> ListEntry<StlType::MsvcStl>::next() {
  if (!m_entry_sp)
    return ListEntry();
  return ListEntry(m_entry_sp->GetChildMemberWithName("_Next"));
}

template <> ListEntry<StlType::MsvcStl> ListEntry<StlType::MsvcStl>::prev() {
  if (!m_entry_sp)
    return ListEntry();
  return ListEntry(m_entry_sp->GetChildMemberWithName("_Prev"));
}

template <StlType Stl> class ListIterator {
public:
  ListIterator() = default;
  ListIterator(ListEntry<Stl> entry) : m_entry(std::move(entry)) {}
  ListIterator(ValueObjectSP entry) : m_entry(std::move(entry)) {}
  ListIterator(ValueObject *entry) : m_entry(entry) {}

  ValueObjectSP value() { return m_entry.GetEntry(); }

  ValueObjectSP advance(size_t count) {
    if (count == 0)
      return m_entry.GetEntry();
    if (count == 1) {
      next();
      return m_entry.GetEntry();
    }
    while (count > 0) {
      next();
      count--;
      if (m_entry.null())
        return lldb::ValueObjectSP();
    }
    return m_entry.GetEntry();
  }

  bool operator==(const ListIterator &rhs) const {
    return (rhs.m_entry == m_entry);
  }

protected:
  void next() { m_entry = m_entry.next(); }

  void prev() { m_entry = m_entry.prev(); }

private:
  ListEntry<Stl> m_entry;
};

template <StlType Stl>
class AbstractListFrontEnd : public SyntheticChildrenFrontEnd {
public:
  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    auto optional_idx = formatters::ExtractIndexFromString(name.GetCString());
    if (!optional_idx) {
      return llvm::createStringError("Type has no child named '%s'",
                                     name.AsCString());
    }
    return *optional_idx;
  }
  lldb::ChildCacheState Update() override;

protected:
  AbstractListFrontEnd(ValueObject &valobj)
      : SyntheticChildrenFrontEnd(valobj) {}

  size_t m_count = 0;
  ValueObject *m_head = nullptr;

  static constexpr bool g_use_loop_detect = true;
  size_t m_loop_detected = 0;   // The number of elements that have had loop
                                // detection run over them.
  ListEntry<Stl> m_slow_runner; // Used for loop detection
  ListEntry<Stl> m_fast_runner; // Used for loop detection

  size_t m_list_capping_size = 0;
  CompilerType m_element_type;
  std::map<size_t, ListIterator<Stl>> m_iterators;

  bool HasLoop(size_t count);
  ValueObjectSP GetItem(size_t idx);
};

class LibCxxForwardListFrontEnd : public AbstractListFrontEnd<StlType::LibCxx> {
public:
  LibCxxForwardListFrontEnd(ValueObject &valobj);

  llvm::Expected<uint32_t> CalculateNumChildren() override;
  ValueObjectSP GetChildAtIndex(uint32_t idx) override;
  lldb::ChildCacheState Update() override;
};

class LibCxxListFrontEnd : public AbstractListFrontEnd<StlType::LibCxx> {
public:
  LibCxxListFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

private:
  lldb::addr_t m_node_address = 0;
  ValueObject *m_tail = nullptr;
};

class MsvcStlForwardListFrontEnd
    : public AbstractListFrontEnd<StlType::MsvcStl> {
public:
  MsvcStlForwardListFrontEnd(ValueObject &valobj);

  llvm::Expected<uint32_t> CalculateNumChildren() override;
  ValueObjectSP GetChildAtIndex(uint32_t idx) override;
  lldb::ChildCacheState Update() override;
};

class MsvcStlListFrontEnd : public AbstractListFrontEnd<StlType::MsvcStl> {
public:
  MsvcStlListFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

private:
  ValueObject *m_tail = nullptr;
};

} // end anonymous namespace

template <StlType Stl>
lldb::ChildCacheState AbstractListFrontEnd<Stl>::Update() {
  m_loop_detected = 0;
  m_count = UINT32_MAX;
  m_head = nullptr;
  m_list_capping_size = 0;
  m_slow_runner.SetEntry(nullptr);
  m_fast_runner.SetEntry(nullptr);
  m_iterators.clear();

  if (m_backend.GetTargetSP())
    m_list_capping_size =
        m_backend.GetTargetSP()->GetMaximumNumberOfChildrenToDisplay();
  if (m_list_capping_size == 0)
    m_list_capping_size = 255;

  CompilerType list_type = m_backend.GetCompilerType();
  if (list_type.IsReferenceType())
    list_type = list_type.GetNonReferenceType();

  if (list_type.GetNumTemplateArguments() == 0)
    return lldb::ChildCacheState::eRefetch;
  m_element_type = list_type.GetTypeTemplateArgument(0);

  return lldb::ChildCacheState::eRefetch;
}

template <StlType Stl> bool AbstractListFrontEnd<Stl>::HasLoop(size_t count) {
  if (!g_use_loop_detect)
    return false;
  // don't bother checking for a loop if we won't actually need to jump nodes
  if (m_count < 2)
    return false;

  if (m_loop_detected == 0) {
    // This is the first time we are being run (after the last update). Set up
    // the loop invariant for the first element.
    m_slow_runner = ListEntry<Stl>(m_head).next();
    m_fast_runner = m_slow_runner.next();
    m_loop_detected = 1;
  }

  // Loop invariant:
  // Loop detection has been run over the first m_loop_detected elements. If
  // m_slow_runner == m_fast_runner then the loop has been detected after
  // m_loop_detected elements.
  const size_t steps_to_run = std::min(count, m_count);
  while (m_loop_detected < steps_to_run && m_slow_runner && m_fast_runner &&
         m_slow_runner != m_fast_runner) {

    m_slow_runner = m_slow_runner.next();
    m_fast_runner = m_fast_runner.next().next();
    m_loop_detected++;
  }
  if (count <= m_loop_detected)
    return false; // No loop in the first m_loop_detected elements.
  if (!m_slow_runner || !m_fast_runner)
    return false; // Reached the end of the list. Definitely no loops.
  return m_slow_runner == m_fast_runner;
}

template <StlType Stl>
ValueObjectSP AbstractListFrontEnd<Stl>::GetItem(size_t idx) {
  size_t advance = idx;
  ListIterator<Stl> current(m_head);
  if (idx > 0) {
    auto cached_iterator = m_iterators.find(idx - 1);
    if (cached_iterator != m_iterators.end()) {
      current = cached_iterator->second;
      advance = 1;
    }
  }
  ValueObjectSP value_sp = current.advance(advance);
  m_iterators[idx] = current;
  return value_sp;
}

LibCxxForwardListFrontEnd::LibCxxForwardListFrontEnd(ValueObject &valobj)
    : AbstractListFrontEnd(valobj) {
  Update();
}

llvm::Expected<uint32_t> LibCxxForwardListFrontEnd::CalculateNumChildren() {
  if (m_count != UINT32_MAX)
    return m_count;

  ListEntry<StlType::LibCxx> current(m_head);
  m_count = 0;
  while (current && m_count < m_list_capping_size) {
    ++m_count;
    current = current.next();
  }
  return m_count;
}

ValueObjectSP LibCxxForwardListFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx >= CalculateNumChildrenIgnoringErrors())
    return nullptr;

  if (!m_head)
    return nullptr;

  if (HasLoop(idx + 1))
    return nullptr;

  ValueObjectSP current_sp = GetItem(idx);
  if (!current_sp)
    return nullptr;

  current_sp = current_sp->GetChildAtIndex(1); // get the __value_ child
  if (!current_sp)
    return nullptr;

  // we need to copy current_sp into a new object otherwise we will end up with
  // all items named __value_
  DataExtractor data;
  Status error;
  current_sp->GetData(data, error);
  if (error.Fail())
    return nullptr;

  return CreateValueObjectFromData(llvm::formatv("[{0}]", idx).str(), data,
                                   m_backend.GetExecutionContextRef(),
                                   m_element_type);
}

lldb::ChildCacheState LibCxxForwardListFrontEnd::Update() {
  AbstractListFrontEnd::Update();

  Status err;
  ValueObjectSP backend_addr(m_backend.AddressOf(err));
  if (err.Fail() || !backend_addr)
    return lldb::ChildCacheState::eRefetch;

  auto list_base_sp = m_backend.GetChildAtIndex(0);
  if (!list_base_sp)
    return lldb::ChildCacheState::eRefetch;

  // Anonymous strucutre index is in base class at index 0.
  auto [impl_sp, is_compressed_pair] =
      GetValueOrOldCompressedPair(*list_base_sp, /*anon_struct_idx=*/0,
                                  "__before_begin_", "__before_begin_");
  if (!impl_sp)
    return ChildCacheState::eRefetch;

  if (is_compressed_pair)
    impl_sp = GetFirstValueOfLibCXXCompressedPair(*impl_sp);

  if (!impl_sp)
    return ChildCacheState::eRefetch;

  m_head = impl_sp->GetChildMemberWithName("__next_").get();

  return ChildCacheState::eRefetch;
}

LibCxxListFrontEnd::LibCxxListFrontEnd(lldb::ValueObjectSP valobj_sp)
    : AbstractListFrontEnd(*valobj_sp) {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t> LibCxxListFrontEnd::CalculateNumChildren() {
  if (m_count != UINT32_MAX)
    return m_count;
  if (!m_head || !m_tail || m_node_address == 0)
    return 0;

  auto [size_node_sp, is_compressed_pair] = GetValueOrOldCompressedPair(
      m_backend, /*anon_struct_idx=*/1, "__size_", "__size_alloc_");
  if (is_compressed_pair)
    size_node_sp = GetFirstValueOfLibCXXCompressedPair(*size_node_sp);

  if (size_node_sp)
    m_count = size_node_sp->GetValueAsUnsigned(UINT32_MAX);

  if (m_count != UINT32_MAX)
    return m_count;

  uint64_t next_val = m_head->GetValueAsUnsigned(0);
  uint64_t prev_val = m_tail->GetValueAsUnsigned(0);
  if (next_val == 0 || prev_val == 0)
    return 0;
  if (next_val == m_node_address)
    return 0;
  if (next_val == prev_val)
    return 1;
  uint64_t size = 2;
  ListEntry<StlType::LibCxx> current(m_head);
  while (current.next() && current.next().value() != m_node_address) {
    size++;
    current = current.next();
    if (size > m_list_capping_size)
      break;
  }
  return m_count = (size - 1);
}

lldb::ValueObjectSP LibCxxListFrontEnd::GetChildAtIndex(uint32_t idx) {
  static ConstString g_value("__value_");
  static ConstString g_next("__next_");

  if (idx >= CalculateNumChildrenIgnoringErrors())
    return lldb::ValueObjectSP();

  if (!m_head || !m_tail || m_node_address == 0)
    return lldb::ValueObjectSP();

  if (HasLoop(idx + 1))
    return lldb::ValueObjectSP();

  ValueObjectSP current_sp = GetItem(idx);
  if (!current_sp)
    return lldb::ValueObjectSP();

  current_sp = current_sp->GetChildAtIndex(1); // get the __value_ child
  if (!current_sp)
    return lldb::ValueObjectSP();

  if (current_sp->GetName() == g_next) {
    ProcessSP process_sp(current_sp->GetProcessSP());
    if (!process_sp)
      return lldb::ValueObjectSP();

    // if we grabbed the __next_ pointer, then the child is one pointer deep-er
    lldb::addr_t addr = current_sp->GetParent()->GetPointerValue().address;
    addr = addr + 2 * process_sp->GetAddressByteSize();
    ExecutionContext exe_ctx(process_sp);
    current_sp =
        CreateValueObjectFromAddress("__value_", addr, exe_ctx, m_element_type);
    if (!current_sp)
      return lldb::ValueObjectSP();
  }

  // we need to copy current_sp into a new object otherwise we will end up with
  // all items named __value_
  DataExtractor data;
  Status error;
  current_sp->GetData(data, error);
  if (error.Fail())
    return lldb::ValueObjectSP();

  StreamString name;
  name.Printf("[%" PRIu64 "]", (uint64_t)idx);
  return CreateValueObjectFromData(name.GetString(), data,
                                   m_backend.GetExecutionContextRef(),
                                   m_element_type);
}

lldb::ChildCacheState LibCxxListFrontEnd::Update() {
  AbstractListFrontEnd::Update();
  m_tail = nullptr;
  m_node_address = 0;

  Status err;
  ValueObjectSP backend_addr(m_backend.AddressOf(err));
  if (err.Fail() || !backend_addr)
    return lldb::ChildCacheState::eRefetch;
  m_node_address = backend_addr->GetValueAsUnsigned(0);
  if (!m_node_address || m_node_address == LLDB_INVALID_ADDRESS)
    return lldb::ChildCacheState::eRefetch;
  ValueObjectSP impl_sp(m_backend.GetChildMemberWithName("__end_"));
  if (!impl_sp)
    return lldb::ChildCacheState::eRefetch;
  m_head = impl_sp->GetChildMemberWithName("__next_").get();
  m_tail = impl_sp->GetChildMemberWithName("__prev_").get();
  return lldb::ChildCacheState::eRefetch;
}

MsvcStlForwardListFrontEnd::MsvcStlForwardListFrontEnd(ValueObject &valobj)
    : AbstractListFrontEnd(valobj) {
  Update();
}

llvm::Expected<uint32_t> MsvcStlForwardListFrontEnd::CalculateNumChildren() {
  if (m_count != UINT32_MAX)
    return m_count;

  ListEntry<StlType::MsvcStl> current(m_head);
  m_count = 0;
  while (current && m_count < m_list_capping_size) {
    ++m_count;
    current = current.next();
  }
  return m_count;
}

ValueObjectSP MsvcStlForwardListFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx >= CalculateNumChildrenIgnoringErrors())
    return nullptr;

  if (!m_head)
    return nullptr;

  if (HasLoop(idx + 1))
    return nullptr;

  ValueObjectSP current_sp = GetItem(idx);
  if (!current_sp)
    return nullptr;

  current_sp = current_sp->GetChildAtIndex(1); // get the _Myval child
  if (!current_sp)
    return nullptr;

  // we need to copy current_sp into a new object otherwise we will end up with
  // all items named _Myval
  DataExtractor data;
  Status error;
  current_sp->GetData(data, error);
  if (error.Fail())
    return nullptr;

  return CreateValueObjectFromData(llvm::formatv("[{0}]", idx).str(), data,
                                   m_backend.GetExecutionContextRef(),
                                   m_element_type);
}

lldb::ChildCacheState MsvcStlForwardListFrontEnd::Update() {
  AbstractListFrontEnd::Update();

  if (auto head_sp =
          m_backend.GetChildAtNamePath({"_Mypair", "_Myval2", "_Myhead"}))
    m_head = head_sp.get();

  return ChildCacheState::eRefetch;
}

MsvcStlListFrontEnd::MsvcStlListFrontEnd(lldb::ValueObjectSP valobj_sp)
    : AbstractListFrontEnd(*valobj_sp) {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t> MsvcStlListFrontEnd::CalculateNumChildren() {
  if (m_count != UINT32_MAX)
    return m_count;
  if (!m_head || !m_tail)
    return 0;

  auto size_sp =
      m_backend.GetChildAtNamePath({"_Mypair", "_Myval2", "_Mysize"});
  if (!size_sp)
    return llvm::createStringError("Failed to resolve size.");

  m_count = size_sp->GetValueAsUnsigned(UINT32_MAX);
  if (m_count == UINT32_MAX)
    return llvm::createStringError("Failed to read size value.");

  return m_count;
}

lldb::ValueObjectSP MsvcStlListFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx >= CalculateNumChildrenIgnoringErrors())
    return lldb::ValueObjectSP();

  if (!m_head || !m_tail)
    return lldb::ValueObjectSP();

  if (HasLoop(idx + 1))
    return lldb::ValueObjectSP();

  ValueObjectSP current_sp = GetItem(idx);
  if (!current_sp)
    return lldb::ValueObjectSP();

  current_sp = current_sp->GetChildAtIndex(2); // get the _Myval child
  if (!current_sp)
    return lldb::ValueObjectSP();

  // we need to copy current_sp into a new object otherwise we will end up with
  // all items named _Myval
  DataExtractor data;
  Status error;
  current_sp->GetData(data, error);
  if (error.Fail())
    return lldb::ValueObjectSP();

  StreamString name;
  name.Printf("[%" PRIu64 "]", (uint64_t)idx);
  return CreateValueObjectFromData(name.GetString(), data,
                                   m_backend.GetExecutionContextRef(),
                                   m_element_type);
}

lldb::ChildCacheState MsvcStlListFrontEnd::Update() {
  AbstractListFrontEnd::Update();
  m_tail = nullptr;
  m_head = nullptr;

  ValueObjectSP last =
      m_backend.GetChildAtNamePath({"_Mypair", "_Myval2", "_Myhead"});
  if (!last)
    return lldb::ChildCacheState::eRefetch;
  ValueObjectSP first = last->GetChildMemberWithName("_Next");
  if (!first)
    return lldb::ChildCacheState::eRefetch;

  m_head = first.get();
  m_tail = last.get();

  return lldb::ChildCacheState::eRefetch;
}

SyntheticChildrenFrontEnd *formatters::LibcxxStdListSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new LibCxxListFrontEnd(valobj_sp) : nullptr);
}

SyntheticChildrenFrontEnd *
formatters::LibcxxStdForwardListSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return valobj_sp ? new LibCxxForwardListFrontEnd(*valobj_sp) : nullptr;
}

bool formatters::IsMsvcStlList(ValueObject &valobj) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue())
    return valobj_sp->GetChildMemberWithName("_Mypair") != nullptr;

  return false;
}

SyntheticChildrenFrontEnd *
formatters::MsvcStlListSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                                lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new MsvcStlListFrontEnd(valobj_sp) : nullptr);
}

SyntheticChildrenFrontEnd *
formatters::MsvcStlForwardListSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return valobj_sp ? new MsvcStlForwardListFrontEnd(*valobj_sp) : nullptr;
}
