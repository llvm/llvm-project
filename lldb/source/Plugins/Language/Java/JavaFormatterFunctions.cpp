//===-- JavaFormatterFunctions.cpp-------------------------------*- C++ -*-===//
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
#include "JavaFormatterFunctions.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/Symbol/JavaASTContext.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

class JavaArraySyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  JavaArraySyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp) {
    if (valobj_sp)
      Update();
  }

  size_t CalculateNumChildren() override {
    ValueObjectSP valobj = GetDereferencedValueObject();
    if (!valobj)
      return 0;

    CompilerType type = valobj->GetCompilerType();
    uint32_t size = JavaASTContext::CalculateArraySize(type, *valobj);
    if (size == UINT32_MAX)
      return 0;
    return size;
  }

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override {
    ValueObjectSP valobj = GetDereferencedValueObject();
    if (!valobj)
      return nullptr;

    ProcessSP process_sp = valobj->GetProcessSP();
    if (!process_sp)
      return nullptr;

    CompilerType type = valobj->GetCompilerType();
    CompilerType element_type = type.GetArrayElementType();
    lldb::addr_t address =
        valobj->GetAddressOf() +
        JavaASTContext::CalculateArrayElementOffset(type, idx);

    Status error;
    size_t byte_size = element_type.GetByteSize(nullptr);
    DataBufferSP buffer_sp(new DataBufferHeap(byte_size, 0));
    size_t bytes_read = process_sp->ReadMemory(address, buffer_sp->GetBytes(),
                                               byte_size, error);
    if (error.Fail() || byte_size != bytes_read)
      return nullptr;

    StreamString name;
    name.Printf("[%" PRIu64 "]", (uint64_t)idx);
    DataExtractor data(buffer_sp, process_sp->GetByteOrder(),
                       process_sp->GetAddressByteSize());
    return CreateValueObjectFromData(
        name.GetString(), data, valobj->GetExecutionContextRef(), element_type);
  }

  bool Update() override { return false; }

  bool MightHaveChildren() override { return true; }

  size_t GetIndexOfChildWithName(const ConstString &name) override {
    return ExtractIndexFromString(name.GetCString());
  }

private:
  ValueObjectSP GetDereferencedValueObject() {
    if (!m_backend.IsPointerOrReferenceType())
      return m_backend.GetSP();

    Status error;
    return m_backend.Dereference(error);
  }
};

} // end of anonymous namespace

bool lldb_private::formatters::JavaStringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &opts) {
  if (valobj.IsPointerOrReferenceType()) {
    Status error;
    ValueObjectSP deref = valobj.Dereference(error);
    if (error.Fail())
      return false;
    return JavaStringSummaryProvider(*deref, stream, opts);
  }

  ProcessSP process_sp = valobj.GetProcessSP();
  if (!process_sp)
    return false;

  ConstString data_name("value");
  ConstString length_name("count");

  ValueObjectSP length_sp = valobj.GetChildMemberWithName(length_name, true);
  ValueObjectSP data_sp = valobj.GetChildMemberWithName(data_name, true);
  if (!data_sp || !length_sp)
    return false;

  bool success = false;
  uint64_t length = length_sp->GetValueAsUnsigned(0, &success);
  if (!success)
    return false;

  if (length == 0) {
    stream.Printf("\"\"");
    return true;
  }
  lldb::addr_t valobj_addr = data_sp->GetAddressOf();

  StringPrinter::ReadStringAndDumpToStreamOptions options(valobj);
  options.SetLocation(valobj_addr);
  options.SetProcessSP(process_sp);
  options.SetStream(&stream);
  options.SetSourceSize(length);
  options.SetNeedsZeroTermination(false);
  options.SetLanguage(eLanguageTypeJava);

  if (StringPrinter::ReadStringAndDumpToStream<
          StringPrinter::StringElementType::UTF16>(options))
    return true;

  stream.Printf("Summary Unavailable");
  return true;
}

bool lldb_private::formatters::JavaArraySummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  if (valobj.IsPointerOrReferenceType()) {
    Status error;
    ValueObjectSP deref = valobj.Dereference(error);
    if (error.Fail())
      return false;
    return JavaArraySummaryProvider(*deref, stream, options);
  }

  CompilerType type = valobj.GetCompilerType();
  uint32_t size = JavaASTContext::CalculateArraySize(type, valobj);
  if (size == UINT32_MAX)
    return false;
  stream.Printf("[%u]{...}", size);
  return true;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::JavaArraySyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return valobj_sp ? new JavaArraySyntheticFrontEnd(valobj_sp) : nullptr;
}
