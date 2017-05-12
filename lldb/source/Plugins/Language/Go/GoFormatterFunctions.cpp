//===-- GoFormatterFunctions.cpp---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
#include <map>

// Other libraries and framework includes
// Project includes
#include "GoFormatterFunctions.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {
class GoSliceSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  GoSliceSyntheticFrontEnd(ValueObject &valobj)
      : SyntheticChildrenFrontEnd(valobj) {
    Update();
  }

  ~GoSliceSyntheticFrontEnd() override = default;

  size_t CalculateNumChildren() override { return m_len; }

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override {
    if (idx < m_len) {
      ValueObjectSP &cached = m_children[idx];
      if (!cached) {
        StreamString idx_name;
        idx_name.Printf("[%" PRIu64 "]", (uint64_t)idx);
        lldb::addr_t object_at_idx = m_base_data_address;
        object_at_idx += idx * m_type.GetByteSize(nullptr);
        cached = CreateValueObjectFromAddress(
            idx_name.GetString(), object_at_idx,
            m_backend.GetExecutionContextRef(), m_type);
      }
      return cached;
    }
    return ValueObjectSP();
  }

  bool Update() override {
    size_t old_count = m_len;

    ConstString array_const_str("array");
    ValueObjectSP array_sp =
        m_backend.GetChildMemberWithName(array_const_str, true);
    if (!array_sp) {
      m_children.clear();
      return old_count == 0;
    }
    m_type = array_sp->GetCompilerType().GetPointeeType();
    m_base_data_address = array_sp->GetPointerValue();

    ConstString len_const_str("len");
    ValueObjectSP len_sp =
        m_backend.GetChildMemberWithName(len_const_str, true);
    if (len_sp) {
      m_len = len_sp->GetValueAsUnsigned(0);
      m_children.clear();
    }

    return old_count == m_len;
  }

  bool MightHaveChildren() override { return true; }

  size_t GetIndexOfChildWithName(const ConstString &name) override {
    return ExtractIndexFromString(name.AsCString());
  }

private:
  CompilerType m_type;
  lldb::addr_t m_base_data_address;
  size_t m_len;
  std::map<size_t, lldb::ValueObjectSP> m_children;
};

} // anonymous namespace

bool lldb_private::formatters::GoStringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &opts) {
  ProcessSP process_sp = valobj.GetProcessSP();
  if (!process_sp)
    return false;

  if (valobj.IsPointerType()) {
    Status err;
    ValueObjectSP deref = valobj.Dereference(err);
    if (!err.Success())
      return false;
    return GoStringSummaryProvider(*deref, stream, opts);
  }

  ConstString str_name("str");
  ConstString len_name("len");

  ValueObjectSP data_sp = valobj.GetChildMemberWithName(str_name, true);
  ValueObjectSP len_sp = valobj.GetChildMemberWithName(len_name, true);
  if (!data_sp || !len_sp)
    return false;
  bool success;
  lldb::addr_t valobj_addr = data_sp->GetValueAsUnsigned(0, &success);

  if (!success)
    return false;

  uint64_t length = len_sp->GetValueAsUnsigned(0);
  if (length == 0) {
    stream.Printf("\"\"");
    return true;
  }

  StringPrinter::ReadStringAndDumpToStreamOptions options(valobj);
  options.SetLocation(valobj_addr);
  options.SetProcessSP(process_sp);
  options.SetStream(&stream);
  options.SetSourceSize(length);
  options.SetNeedsZeroTermination(false);
  options.SetLanguage(eLanguageTypeGo);

  if (!StringPrinter::ReadStringAndDumpToStream<
          StringPrinter::StringElementType::UTF8>(options)) {
    stream.Printf("Summary Unavailable");
    return true;
  }

  return true;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::GoSliceSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
  if (!process_sp)
    return nullptr;
  return new GoSliceSyntheticFrontEnd(*valobj_sp);
}
