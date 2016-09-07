//===-- NSDictionary.h ---------------------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NSDictionary_h_
#define liblldb_NSDictionary_h_

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

#include <map>

namespace lldb_private {
namespace formatters {
template <bool name_entries>
bool NSDictionarySummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options);

extern template bool
NSDictionarySummaryProvider<true>(ValueObject &, Stream &,
                                  const TypeSummaryOptions &);

extern template bool
NSDictionarySummaryProvider<false>(ValueObject &, Stream &,
                                   const TypeSummaryOptions &);

SyntheticChildrenFrontEnd *
NSDictionarySyntheticFrontEndCreator(CXXSyntheticChildren *,
                                     lldb::ValueObjectSP);

class NSDictionary_Additionals {
public:
  static std::map<ConstString, CXXFunctionSummaryFormat::Callback> &
  GetAdditionalSummaries();

  static std::map<ConstString, CXXSyntheticChildren::CreateFrontEndCallback> &
  GetAdditionalSynthetics();
};
} // namespace formatters
} // namespace lldb_private

#endif // liblldb_NSDictionary_h_
