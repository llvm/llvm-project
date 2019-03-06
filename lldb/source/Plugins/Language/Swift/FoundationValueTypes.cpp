//===-- FoundationValueTypes.cpp --------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "FoundationValueTypes.h"
#include "ObjCRuntimeSyntheticProvider.h"

#include "llvm/ADT/STLExtras.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Status.h"

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
  Status error;
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
  static ConstString g_empty("empty");
  static ConstString g_single("single");
  static ConstString g_pair("pair");
  static ConstString g_array("array");
  
  ValueObjectSP underlying_enum_sp(valobj.GetChildAtNamePath({g__indexes}));

  if (!underlying_enum_sp)
    return false;

  underlying_enum_sp =
      underlying_enum_sp->GetQualifiedRepresentationIfAvailable(
          lldb::eDynamicDontRunTarget, true);
  ConstString value(underlying_enum_sp->GetValueAsCString());
  if (value.IsEmpty())
    return false;
  
  if (value == g_empty)
    stream.PutCString("0 indices");
  else if (value == g_single)
    stream.PutCString("1 index");
  else if (value == g_pair)
    stream.PutCString("2 indices");
  else if (value == g_array)
  {
    if (underlying_enum_sp->GetNumChildren() != 1) 
      return false;
  
    underlying_enum_sp = underlying_enum_sp->GetChildAtIndex(0, true)
       ->GetQualifiedRepresentationIfAvailable(lldb::eDynamicDontRunTarget, true);
    size_t num_children = underlying_enum_sp->GetNumChildren();
    stream.Printf("%zu indices", num_children);
  }
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
  Status error;
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
  // Grab the underlying representation from
  //
  //   struct Data {
  //       enum _Representation { ... }
  //       var _representation: _Representation
  //   }
  static ConstString g__representation("_representation");
  ValueObjectSP representation_enum_sp =
      valobj.GetChildAtNamePath(g__representation);
  if (!representation_enum_sp)
    return false;

  representation_enum_sp =
      representation_enum_sp->GetQualifiedRepresentationIfAvailable(
          lldb::eDynamicDontRunTarget, true);
  if (!representation_enum_sp)
    return false;

  // representation_case holds the name of the enum case we're looking at.
  ConstString representation_case(representation_enum_sp->GetValueAsCString());
  if (!representation_case)
    return false;

  // Switch on
  //
  //   enum _Representation {
  //       case empty
  //       case inline(InlineData)
  //       case slice(InlineSlice)
  //       case large(LargeSlice)
  //   }
  static ConstString g_empty("empty");
  static ConstString g_inline("inline");
  static ConstString g_slice("slice");
  static ConstString g_large("large");

  int64_t count = 0;
  if (representation_case == g_empty) {
    // Do nothing; count is already 0.
  } else if (representation_case == g_inline) {
    // Grab the associated value from `case inline(InlineData)`.
    if (representation_enum_sp->GetNumChildren() != 1)
      return false;

    ValueObjectSP inline_data_sp =
        representation_enum_sp->GetChildAtIndex(0, true)
            ->GetQualifiedRepresentationIfAvailable(lldb::eDynamicDontRunTarget,
                                                    true);
    if (!inline_data_sp)
      return false;

    // Grab the length out of
    //
    //   struct InlineData {
    //       var length: UInt8
    //       var buffer: (...)
    //   }
    static ConstString g_length("length");
    ValueObjectSP length_sp = inline_data_sp->GetChildAtNamePath(g_length)
                                  ->GetQualifiedRepresentationIfAvailable(
                                      lldb::eDynamicDontRunTarget, true);
    if (!length_sp)
      return false;

    bool success = false;
    count = (int64_t)length_sp->GetValueAsUnsigned(0, &success);
    if (!success) {
      return false;
    }
  } else if (representation_case == g_slice) {
    // Grab the associated value from `case slice(InlineSlice)`.
    if (representation_enum_sp->GetNumChildren() != 1)
      return false;

    ValueObjectSP slice_data_sp =
        representation_enum_sp->GetChildAtIndex(0, true)
            ->GetQualifiedRepresentationIfAvailable(lldb::eDynamicDontRunTarget,
                                                    true);
    if (!slice_data_sp)
      return false;

    // Grab the slice out of
    //
    //   struct InlineSlice {
    //       var slice: Range<HalfInt>
    //       var storage: __DataStorage
    //   }
    static ConstString g_slice("slice");
    ValueObjectSP slice_storage_sp = slice_data_sp->GetChildAtNamePath(g_slice);
    if (!slice_storage_sp)
      return false;

    slice_storage_sp = slice_storage_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
    if (!slice_storage_sp)
      return false;

    // We need to manually calculate slice.upperBound - slice.lowerBound.
    static ConstString g_upperBound("upperBound");
    ValueObjectSP upper_bound_sp =
        slice_storage_sp->GetChildAtNamePath(g_upperBound);
    if (!upper_bound_sp)
      return false;

    upper_bound_sp = upper_bound_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
    if (!upper_bound_sp)
      return false;

    static ConstString g_lowerBound("lowerBound");
    ValueObjectSP lower_bound_sp =
        slice_storage_sp->GetChildAtNamePath(g_lowerBound);
    if (!lower_bound_sp)
      return false;

    lower_bound_sp = lower_bound_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
    if (!lower_bound_sp)
      return false;

    bool success = false;
    int64_t upperBound = upper_bound_sp->GetValueAsSigned(0, &success);
    if (!success)
      return false;

    int64_t lowerBound = lower_bound_sp->GetValueAsSigned(0, &success);
    if (!success)
      return false;

    count = upperBound - lowerBound;
  } else if (representation_case == g_large) {
    // Grab the associated value from `case large(LargeSlice)`.
    if (representation_enum_sp->GetNumChildren() != 1)
      return false;

    ValueObjectSP large_data_sp =
        representation_enum_sp->GetChildAtIndex(0, true)
            ->GetQualifiedRepresentationIfAvailable(lldb::eDynamicDontRunTarget,
                                                    true);
    if (!large_data_sp)
      return false;

    // Grab the reference out of
    //
    //   struct LargeSlice {
    //       var slice: RangeReference
    //       var storage: __DataStorage
    //   }
    static ConstString g_slice("slice");
    ValueObjectSP slice_ref_sp = large_data_sp->GetChildAtNamePath(g_slice);
    if (!slice_ref_sp)
      return false;

    slice_ref_sp = slice_ref_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
    if (!slice_ref_sp)
      return false;

    // Grab the range out of
    //
    //   class RangeReference {
    //       var range: Range<Int>
    //   }
    static ConstString g_range("range");
    ValueObjectSP range_sp = slice_ref_sp->GetChildAtNamePath(g_range);
    if (!range_sp)
      return false;

    range_sp = range_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
    if (!range_sp)
      return false;

    // We need to manually calculate range.upperBound - range.lowerBound.
    static ConstString g_upperBound("upperBound");
    ValueObjectSP upper_bound_sp = range_sp->GetChildAtNamePath(g_upperBound);
    if (!upper_bound_sp)
      return false;

    upper_bound_sp = upper_bound_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
    if (!upper_bound_sp)
      return false;

    static ConstString g_lowerBound("lowerBound");
    ValueObjectSP lower_bound_sp = range_sp->GetChildAtNamePath(g_lowerBound);
    if (!lower_bound_sp)
      return false;

    lower_bound_sp = lower_bound_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
    if (!lower_bound_sp)
      return false;

    bool success = false;
    int64_t upperBound = upper_bound_sp->GetValueAsSigned(0, &success);
    if (!success)
      return false;

    int64_t lowerBound = lower_bound_sp->GetValueAsSigned(0, &success);
    if (!success)
      return false;

    count = upperBound - lowerBound;
  } else {
    // Unknown enum case.
    return false;
  }

  if (count == 1)
    stream.Printf("1 byte");
  else
    stream.Printf("%lld bytes", count);

  return true;
}

bool lldb_private::formatters::swift::Decimal_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {

  // The layout of the type is:
  // public struct Decimal {
  //   fileprivate var __exponent : Int8
  //   fileprivate var __lengthAndFlags: UInt8
  //   fileprivate var __reserved: UInt16
  //   public var _mantissa: (UInt16, UInt16, UInt16, UInt16, UInt16, UInt16,
  //   UInt16, UInt16)
  // We do have to harcode the offset of the variables because they're
  // fileprivate, but we can access `_mantissa` by name.

  ProcessSP process(valobj.GetProcessSP());
  if (!process)
    return false;

  Status error;
  DataExtractor data_extractor;
  if (!valobj.GetData(data_extractor, error))
    return false;

  offset_t offset_ptr = 0;
  int8_t exponent = data_extractor.GetU8(&offset_ptr);
  uint8_t length_and_flags = data_extractor.GetU8(&offset_ptr);
  uint8_t length = length_and_flags & 0xf;
  bool isNegative = length_and_flags & 0x10;

  static ConstString g_mantissa("_mantissa");
  ValueObjectSP mantissa_sp = valobj.GetChildAtNamePath(g_mantissa);
  if (!mantissa_sp)
    return false;

  // Easy case. length == 0 is either `NaN` or `0`.
  if (length == 0) {
    if (isNegative)
      stream.Printf("NaN");
    else
      stream.Printf("0");
    return true;
  }

  // Mantissa is represented as a tuple of 8 UInt16.
  const uint8_t num_children = 8;
  if (mantissa_sp->GetNumChildren() != num_children)
    return false;

  std::vector<double> mantissa_elements;
  for (int i = 0; i < 8; ++i) {
    ValueObjectSP child_sp = mantissa_sp->GetChildAtIndex(i, true);
    if (!child_sp)
      return false;
    static ConstString g_value("_value");
    ValueObjectSP value_sp = child_sp->GetChildAtNamePath(g_value);
    if (!value_sp)
      return false;
    auto val = value_sp->GetValueAsUnsigned(0) & 0xffff;
    mantissa_elements.push_back(static_cast<double>(val));
  }

  // Compute the value using mantissa and exponent
  double d = 0.0;
  for (int i = std::min(length, num_children); i > 0; i--) {
    d = (d * 65536) + mantissa_elements[i - 1];
  }

  if (exponent < 0)
    for (int i = exponent; i < 0; ++i)
      d /= 10.0;
  else
    for (int i = 0; i < exponent; ++i)
      d *= 10.0;

  if (isNegative)
    d = -d;

  stream.Printf("%lf\n", d);
  return true;
}
class URLComponentsSyntheticChildrenFrontEnd
    : public SyntheticChildrenFrontEnd {
public:
  URLComponentsSyntheticChildrenFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp), m_synth_frontend_up(),
        m_synth_backend_up()
#define COMPONENT(Name, PrettyName, ID) , m_##Name(nullptr)
#include "URLComponents.def"
  {
    SetValid(false);
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
    return (m_##Name) ? (m_##Name)->GetSP() : nullptr;
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

    m_synth_frontend_up.reset();
    m_synth_backend_up.reset();

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

    m_synth_backend_up = llvm::make_unique<ObjCRuntimeSyntheticProvider>(
        SyntheticChildren::Flags(), class_descriptor_sp);
    m_synth_frontend_up = m_synth_backend_up->GetFrontEnd(*underlying_sp);
    if (!m_synth_frontend_up)
      return false;
    else
      m_synth_frontend_up->Update();

#define COMPONENT(Name, PrettyName, ID)                                        \
  m_##Name = m_synth_frontend_up                                               \
                 ->GetChildAtIndex(                                            \
                     m_synth_frontend_up->GetIndexOfChildWithName(g__##Name))  \
                 .get();                                                       \
  if (m_##Name)                                                                \
    m_##Name->SetName(GetNameFor##Name());
#include "URLComponents.def"

    SetValid(CheckValid());

    return false;
  }

  bool MightHaveChildren() override { return true; }

  size_t GetIndexOfChildWithName(ConstString name) override {
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

  SyntheticChildrenFrontEnd::AutoPointer m_synth_frontend_up;
  std::unique_ptr<ObjCRuntimeSyntheticProvider> m_synth_backend_up;
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
