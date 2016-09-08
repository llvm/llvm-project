//===-- FoundationValueTypes.cpp ----------------------------------*- C++
//-*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "FoundationValueTypes.h"
#include "ObjCRuntimeSyntheticProvider.h"

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

bool lldb_private::formatters::swift::Date_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g__time("_time");

  ValueObjectSP time_sp(valobj.GetChildAtNamePath({g__time}));

  if (!time_sp)
    return false;

  DataExtractor data_extractor;
  Error error;
  if (!time_sp->GetData(data_extractor, error))
    return false;

  offset_t offset_ptr = 0;
  double date_value = data_extractor.GetDouble(&offset_ptr);

  if (date_value == -63114076800) {
    stream.Printf("0001-12-30 00:00:00 +0000");
    return true;
  }
  // this snippet of code assumes that time_t == seconds since Jan-1-1970
  // this is generally true and POSIXly happy, but might break if a library
  // vendor decides to get creative
  time_t epoch = GetOSXEpoch();
  epoch = epoch + (time_t)date_value;
  tm *tm_date = gmtime(&epoch);
  if (!tm_date)
    return false;
  std::string buffer(1024, 0);
  if (strftime(&buffer[0], 1023, "%Z", tm_date) == 0)
    return false;
  stream.Printf("%04d-%02d-%02d %02d:%02d:%02d %s", tm_date->tm_year + 1900,
                tm_date->tm_mon + 1, tm_date->tm_mday, tm_date->tm_hour,
                tm_date->tm_min, tm_date->tm_sec, buffer.c_str());
  return true;
}

bool lldb_private::formatters::swift::NotificationName_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g__rawValue("_rawValue");

  ValueObjectSP underlying_name_sp(valobj.GetChildAtNamePath({g__rawValue}));

  if (!underlying_name_sp)
    return false;

  std::string summary;
  if (!underlying_name_sp->GetSummaryAsCString(summary, options))
    return false;

  stream.PutCString(summary.c_str());
  return true;
}

bool lldb_private::formatters::swift::URL_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g__url("_url");

  ValueObjectSP underlying_url_sp(valobj.GetChildAtNamePath({g__url}));

  if (!underlying_url_sp)
    return false;

  std::string summary;
  if (!underlying_url_sp->GetSummaryAsCString(summary, options))
    return false;

  stream.PutCString(summary.c_str());
  return true;
}

bool lldb_private::formatters::swift::IndexPath_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g__indexes("_indexes");

  ValueObjectSP underlying_array_sp(valobj.GetChildAtNamePath({g__indexes}));

  if (!underlying_array_sp)
    return false;

  underlying_array_sp =
      underlying_array_sp->GetQualifiedRepresentationIfAvailable(
          lldb::eDynamicDontRunTarget, true);

  size_t num_children = underlying_array_sp->GetNumChildren();

  if (num_children == 1)
    stream.PutCString("1 index");
  else
    stream.Printf("%zu indices", num_children);
  return true;
}

bool lldb_private::formatters::swift::Measurement_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g_value("value");
  static ConstString g_unit("unit");
  static ConstString g__symbol("_symbol");

  ValueObjectSP value_sp(valobj.GetChildAtNamePath({g_value}));
  if (!value_sp)
    return false;

  ValueObjectSP unit_sp(valobj.GetChildAtNamePath({g_unit}));
  if (!unit_sp)
    return false;

  ProcessSP process_sp(valobj.GetProcessSP());
  if (!process_sp)
    return false;

  auto descriptor_sp(
      process_sp->GetObjCLanguageRuntime()->GetClassDescriptor(*unit_sp));
  if (!descriptor_sp)
    return false;

  if (descriptor_sp->GetNumIVars() == 0)
    return false;

  auto ivar = descriptor_sp->GetIVarAtIndex(0);
  if (!ivar.m_type.IsValid())
    return false;

  ValueObjectSP symbol_sp(
      unit_sp->GetSyntheticChildAtOffset(ivar.m_offset, ivar.m_type, true));
  if (!symbol_sp)
    return false;

  symbol_sp = symbol_sp->GetQualifiedRepresentationIfAvailable(
      lldb::eDynamicDontRunTarget, true);

  DataExtractor data_extractor;
  Error error;
  if (!value_sp->GetData(data_extractor, error))
    return false;

  offset_t offset_ptr = 0;
  double measurement_value = data_extractor.GetDouble(&offset_ptr);

  std::string unit;
  if (!symbol_sp->GetSummaryAsCString(unit, options))
    return false;

  if (unit.size() > 2 && unit[0] == '"') {
    unit = unit.substr(1);
    if (unit.back() == '"')
      unit.pop_back();
  }

  stream.Printf("%g %s", measurement_value, unit.c_str());
  return true;
}

bool lldb_private::formatters::swift::UUID_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g_uuid("uuid");

  ValueObjectSP uuid_sp(valobj.GetChildAtNamePath({g_uuid}));
  if (!uuid_sp)
    return false;

  if (uuid_sp->GetNumChildren() < 16)
    return false;

  ValueObjectSP children[] = {
      uuid_sp->GetChildAtIndex(0, true),  uuid_sp->GetChildAtIndex(1, true),
      uuid_sp->GetChildAtIndex(2, true),  uuid_sp->GetChildAtIndex(3, true),
      uuid_sp->GetChildAtIndex(4, true),  uuid_sp->GetChildAtIndex(5, true),
      uuid_sp->GetChildAtIndex(6, true),  uuid_sp->GetChildAtIndex(7, true),
      uuid_sp->GetChildAtIndex(8, true),  uuid_sp->GetChildAtIndex(9, true),
      uuid_sp->GetChildAtIndex(10, true), uuid_sp->GetChildAtIndex(11, true),
      uuid_sp->GetChildAtIndex(12, true), uuid_sp->GetChildAtIndex(13, true),
      uuid_sp->GetChildAtIndex(14, true), uuid_sp->GetChildAtIndex(15, true)};

  for (ValueObjectSP &child : children) {
    if (!child)
      return false;
    child = child->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
  }

  const char *separator = "-";
  stream.Printf("%2.2X%2.2X%2.2X%2.2X%s%2.2X%2.2X%s%2.2X%2.2X%s%2.2X%2.2X%s%2."
                "2X%2.2X%2.2X%2.2X%2.2X%2.2X",
                (uint8_t)children[0]->GetValueAsUnsigned(0),
                (uint8_t)children[1]->GetValueAsUnsigned(0),
                (uint8_t)children[2]->GetValueAsUnsigned(0),
                (uint8_t)children[3]->GetValueAsUnsigned(0), separator,
                (uint8_t)children[4]->GetValueAsUnsigned(0),
                (uint8_t)children[5]->GetValueAsUnsigned(0), separator,
                (uint8_t)children[6]->GetValueAsUnsigned(0),
                (uint8_t)children[7]->GetValueAsUnsigned(0), separator,
                (uint8_t)children[8]->GetValueAsUnsigned(0),
                (uint8_t)children[9]->GetValueAsUnsigned(0), separator,
                (uint8_t)children[10]->GetValueAsUnsigned(0),
                (uint8_t)children[11]->GetValueAsUnsigned(0),
                (uint8_t)children[12]->GetValueAsUnsigned(0),
                (uint8_t)children[13]->GetValueAsUnsigned(0),
                (uint8_t)children[14]->GetValueAsUnsigned(0),
                (uint8_t)children[15]->GetValueAsUnsigned(0));

  return true;
}

bool lldb_private::formatters::swift::Data_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g__wrapped("_wrapped");
  static ConstString g___wrapped("__wrapped");
  static ConstString g_Immutable("Immutable");
  static ConstString g_Mutable("Mutable");
  static ConstString g__value("_value");

  ValueObjectSP selected_case_sp =
      valobj.GetChildAtNamePath({g__wrapped, g___wrapped});
  if (!selected_case_sp)
    return false;

  ConstString selected_case(selected_case_sp->GetValueAsCString());
  if (selected_case == g_Immutable) {
    if (ValueObjectSP immutable_sp =
            selected_case_sp->GetChildAtNamePath({g_Immutable, g__value})) {
      std::string summary;
      if (immutable_sp->GetSummaryAsCString(summary, options)) {
        stream.Printf("%s", summary.c_str());
        return true;
      }
    }
  } else if (selected_case == g_Mutable) {
    if (ValueObjectSP mutable_sp =
            selected_case_sp->GetChildAtNamePath({g_Mutable, g__value})) {
      ProcessSP process_sp(valobj.GetProcessSP());
      if (!process_sp)
        return false;
      TargetSP target_sp(valobj.GetTargetSP());
      if (!target_sp)
        return false;
      if (SwiftLanguageRuntime *swift_runtime =
              valobj.GetProcessSP()->GetSwiftLanguageRuntime()) {
        lldb::addr_t value =
            mutable_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
        if (value != LLDB_INVALID_ADDRESS) {
          value = swift_runtime->MaskMaybeBridgedPointer(value);
          DataExtractor buffer(&value, process_sp->GetAddressByteSize(),
                               process_sp->GetByteOrder(),
                               process_sp->GetAddressByteSize());
          if (ClangASTContext *clang_ast_ctx =
                  target_sp->GetScratchClangASTContext()) {
            if (CompilerType id_type =
                    clang_ast_ctx->GetBasicType(lldb::eBasicTypeObjCID)) {
              if (ValueObjectSP nsdata_sp =
                      ValueObject::CreateValueObjectFromData(
                          "nsdata", buffer, process_sp, id_type)) {
                nsdata_sp = nsdata_sp->GetQualifiedRepresentationIfAvailable(
                    lldb::eDynamicDontRunTarget, false);
                std::string summary;
                if (nsdata_sp->GetSummaryAsCString(summary, options)) {
                  stream.Printf("%s", summary.c_str());
                  return true;
                }
              }
            }
          }
        }
      }
    }
  }

  return false;
}

class URLComponentsSyntheticChildrenFrontEnd
    : public SyntheticChildrenFrontEnd {
public:
  URLComponentsSyntheticChildrenFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp), m_synth_up(nullptr)
#define COMPONENT(Name, PrettyName, ID) , m_##Name(nullptr)
#include "URLComponents.def"
  {
  }

  ~URLComponentsSyntheticChildrenFrontEnd() override = default;

  size_t CalculateNumChildren() override {
    if (IsValid())
      return 9;
    return 0;
  }

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override {
    if (IsValid()) {
      switch (idx) {
#define COMPONENT(Name, PrettyName, ID)                                        \
  case ID:                                                                     \
    return (m_##Name)->GetSP();
#include "URLComponents.def"
      default:
        break;
      }
    }

    return nullptr;
  }

  bool Update() override {
    static ConstString g__handle("_handle");
    static ConstString g__pointer("_pointer");

#define COMPONENT(Name, PrettyName, ID)                                        \
  static ConstString g__##Name = ConstString("_" #Name);
#include "URLComponents.def"

    m_synth_up.reset();

#define COMPONENT(Name, PrettyName, ID) m_##Name = nullptr;
#include "URLComponents.def"

    SetValid(false);

    ValueObjectSP underlying_sp =
        m_backend.GetChildAtNamePath({g__handle, g__pointer});
    if (!underlying_sp)
      return false;

    ObjCLanguageRuntime *objc_runtime =
        m_backend.GetProcessSP()->GetObjCLanguageRuntime();
    if (!objc_runtime)
      return false;

    ObjCLanguageRuntime::ClassDescriptorSP class_descriptor_sp =
        objc_runtime->GetClassDescriptor(*underlying_sp);
    if (!class_descriptor_sp)
      return false;

    m_synth_up = ObjCRuntimeSyntheticProvider(SyntheticChildren::Flags(),
                                              class_descriptor_sp)
                     .GetFrontEnd(*underlying_sp);
    if (!m_synth_up)
      return false;
    else
      m_synth_up->Update();

#define COMPONENT(Name, PrettyName, ID)                                        \
  m_##Name =                                                                   \
      m_synth_up                                                               \
          ->GetChildAtIndex(m_synth_up->GetIndexOfChildWithName(g__##Name))    \
          .get();                                                              \
  if (m_##Name)                                                                \
    m_##Name->SetName(GetNameFor##Name());
#include "URLComponents.def"

    SetValid(CheckValid());

    return false;
  }

  bool MightHaveChildren() override { return true; }

  size_t GetIndexOfChildWithName(const ConstString &name) override {
#define COMPONENT(Name, PrettyName, ID)                                        \
  if (name == GetNameFor##Name())                                              \
    return ID;
#include "URLComponents.def"
    return UINT32_MAX;
  }

private:
#define COMPONENT(Name, PrettyName, ID)                                        \
  static ConstString GetNameFor##Name() {                                      \
    static ConstString g_value(#PrettyName);                                   \
    return g_value;                                                            \
  }
#include "URLComponents.def"

  SyntheticChildrenFrontEnd::AutoPointer m_synth_up;
#define COMPONENT(Name, PrettyName, ID) ValueObject *m_##Name;
#include "URLComponents.def"

  bool CheckValid() {
#define COMPONENT(Name, PrettyName, ID)                                        \
  if (m_##Name == nullptr)                                                     \
    return false;
#include "URLComponents.def"

    return true;
  }
};

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::URLComponentsSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  return new URLComponentsSyntheticChildrenFrontEnd(valobj_sp);
}
