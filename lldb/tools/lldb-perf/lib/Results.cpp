//===-- Results.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Results.h"
#include <assert.h>

#ifdef __APPLE__
#include "CFCMutableArray.h"
#include "CFCMutableDictionary.h"
#include "CFCReleaser.h"
#include "CFCString.h"
#endif

using namespace lldb_perf;

static void AddResultToArray(CFCMutableArray &array, Results::Result *result);

static void AddResultToDictionary(CFCMutableDictionary &parent_dict,
                                  const char *key, Results::Result *result);

static void AddResultToArray(CFCMutableArray &parent_array,
                             Results::Result *result) {
  switch (result->GetType()) {
  case Results::Result::Type::Invalid:
    break;

  case Results::Result::Type::Array: {
    Results::Array *value = result->GetAsArray();
    CFCMutableArray array;
    value->ForEach([&array](const Results::ResultSP &value_sp) -> bool {
      AddResultToArray(array, value_sp.get());
      return true;
    });
    parent_array.AppendValue(array.get(), true);
  } break;

  case Results::Result::Type::Dictionary: {
    Results::Dictionary *value = result->GetAsDictionary();
    CFCMutableDictionary dict;
    value->ForEach([&dict](const std::string &key,
                           const Results::ResultSP &value_sp) -> bool {
      AddResultToDictionary(dict, key.c_str(), value_sp.get());
      return true;
    });
    if (result->GetDescription()) {
      dict.AddValueCString(CFSTR("description"), result->GetDescription());
    }
    parent_array.AppendValue(dict.get(), true);
  } break;

  case Results::Result::Type::Double: {
    double d = result->GetAsDouble()->GetValue();
    CFCReleaser<CFNumberRef> cf_number(
        ::CFNumberCreate(kCFAllocatorDefault, kCFNumberDoubleType, &d));
    if (cf_number.get())
      parent_array.AppendValue(cf_number.get(), true);
  } break;
  case Results::Result::Type::String: {
    CFCString cfstr(result->GetAsString()->GetValue());
    if (cfstr.get())
      parent_array.AppendValue(cfstr.get(), true);
  } break;

  case Results::Result::Type::Unsigned: {
    uint64_t uval64 = result->GetAsUnsigned()->GetValue();
    CFCReleaser<CFNumberRef> cf_number(
        ::CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &uval64));
    if (cf_number.get())
      parent_array.AppendValue(cf_number.get(), true);
  } break;

  default:
    llvm_unreachable("unhandled result");
  }
}

static void AddResultToDictionary(CFCMutableDictionary &parent_dict,
                                  const char *key, Results::Result *result) {
  assert(key && key[0]);
  CFCString cf_key(key);
  switch (result->GetType()) {
  case Results::Result::Type::Invalid:
    break;

  case Results::Result::Type::Array: {
    Results::Array *value = result->GetAsArray();
    CFCMutableArray array;
    value->ForEach([&array](const Results::ResultSP &value_sp) -> bool {
      AddResultToArray(array, value_sp.get());
      return true;
    });
    parent_dict.AddValue(cf_key.get(), array.get(), true);
  } break;
  case Results::Result::Type::Dictionary: {
    Results::Dictionary *value = result->GetAsDictionary();
    CFCMutableDictionary dict;
    value->ForEach([&dict](const std::string &key,
                           const Results::ResultSP &value_sp) -> bool {
      AddResultToDictionary(dict, key.c_str(), value_sp.get());
      return true;
    });
    if (result->GetDescription()) {
      dict.AddValueCString(CFSTR("description"), result->GetDescription());
    }
    parent_dict.AddValue(cf_key.get(), dict.get(), true);
  } break;
  case Results::Result::Type::Double: {
    parent_dict.SetValueDouble(cf_key.get(), result->GetAsDouble()->GetValue(),
                               true);
  } break;
  case Results::Result::Type::String: {
    parent_dict.SetValueCString(cf_key.get(), result->GetAsString()->GetValue(),
                                true);
  } break;

  case Results::Result::Type::Unsigned: {
    parent_dict.SetValueUInt64(cf_key.get(),
                               result->GetAsUnsigned()->GetValue(), true);
  } break;
  default:
    llvm_unreachable("unhandled result");
  }
}
void Results::Write(const char *out_path) {
#ifdef __APPLE__
  CFCMutableDictionary dict;

  m_results.ForEach(
      [&dict](const std::string &key, const ResultSP &value_sp) -> bool {
        AddResultToDictionary(dict, key.c_str(), value_sp.get());
        return true;
      });
  CFDataRef xmlData = CFPropertyListCreateData(
      kCFAllocatorDefault, dict.get(), kCFPropertyListXMLFormat_v1_0, 0, NULL);

  if (out_path == NULL)
    out_path = "/dev/stdout";

  CFURLRef file = CFURLCreateFromFileSystemRepresentation(
      NULL, (const UInt8 *)out_path, strlen(out_path), FALSE);

  CFURLWriteDataAndPropertiesToResource(file, xmlData, NULL, NULL);
#endif
}

Results::ResultSP Results::Dictionary::AddUnsigned(const char *name,
                                                   const char *description,
                                                   uint64_t value) {
  assert(name && name[0]);
  if (description && description[0]) {
    std::unique_ptr<Results::Dictionary> value_dict_up(
        new Results::Dictionary());
    value_dict_up->AddString("description", NULL, description);
    value_dict_up->AddUnsigned("value", NULL, value);
    m_dictionary[std::string(name)] = ResultSP(value_dict_up.release());
  } else
    m_dictionary[std::string(name)] =
        ResultSP(new Unsigned(name, description, value));
  return m_dictionary[std::string(name)];
}

Results::ResultSP Results::Dictionary::AddDouble(const char *name,
                                                 const char *description,
                                                 double value) {
  assert(name && name[0]);

  if (description && description[0]) {
    std::unique_ptr<Results::Dictionary> value_dict_up(
        new Results::Dictionary());
    value_dict_up->AddString("description", NULL, description);
    value_dict_up->AddDouble("value", NULL, value);
    m_dictionary[std::string(name)] = ResultSP(value_dict_up.release());
  } else
    m_dictionary[std::string(name)] =
        ResultSP(new Double(name, description, value));
  return m_dictionary[std::string(name)];
}
Results::ResultSP Results::Dictionary::AddString(const char *name,
                                                 const char *description,
                                                 const char *value) {
  assert(name && name[0]);
  if (description && description[0]) {
    std::unique_ptr<Results::Dictionary> value_dict_up(
        new Results::Dictionary());
    value_dict_up->AddString("description", NULL, description);
    value_dict_up->AddString("value", NULL, value);
    m_dictionary[std::string(name)] = ResultSP(value_dict_up.release());
  } else
    m_dictionary[std::string(name)] =
        ResultSP(new String(name, description, value));
  return m_dictionary[std::string(name)];
}

Results::ResultSP Results::Dictionary::Add(const char *name,
                                           const char *description,
                                           const ResultSP &result_sp) {
  assert(name && name[0]);
  if (description && description[0]) {
    std::unique_ptr<Results::Dictionary> value_dict_up(
        new Results::Dictionary());
    value_dict_up->AddString("description", NULL, description);
    value_dict_up->Add("value", NULL, result_sp);
    m_dictionary[std::string(name)] = ResultSP(value_dict_up.release());
  } else
    m_dictionary[std::string(name)] = result_sp;
  return m_dictionary[std::string(name)];
}

void Results::Dictionary::ForEach(
    const std::function<bool(const std::string &, const ResultSP &)>
        &callback) {
  collection::const_iterator pos, end = m_dictionary.end();
  for (pos = m_dictionary.begin(); pos != end; ++pos) {
    if (callback(pos->first.c_str(), pos->second) == false)
      return;
  }
}

Results::ResultSP Results::Array::Append(const ResultSP &result_sp) {
  m_array.push_back(result_sp);
  return result_sp;
}

void Results::Array::ForEach(
    const std::function<bool(const ResultSP &)> &callback) {
  collection::const_iterator pos, end = m_array.end();
  for (pos = m_array.begin(); pos != end; ++pos) {
    if (callback(*pos) == false)
      return;
  }
}
