//===-- MsvcStl.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_MSVCSTL_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_MSVCSTL_H

#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

namespace lldb_private {
namespace formatters {

bool IsMsvcStlStringType(ValueObject &valobj);

template <StringPrinter::StringElementType element_type>
bool MsvcStlStringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions
        &summary_options); // VC 2015+ std::string,u8string,u16string,u32string

bool MsvcStlWStringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // VC 2015+ std::wstring

template <StringPrinter::StringElementType element_type>
bool MsvcStlStringViewSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options); // std::{u8,u16,u32}?string_view

bool MsvcStlWStringViewSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // std::wstring_view

// MSVC STL std::shared_ptr<> and std::weak_ptr<>
bool IsMsvcStlSmartPointer(ValueObject &valobj);
bool MsvcStlSmartPointerSummaryProvider(ValueObject &valobj, Stream &stream,
                                        const TypeSummaryOptions &options);

lldb_private::SyntheticChildrenFrontEnd *
MsvcStlSmartPointerSyntheticFrontEndCreator(lldb::ValueObjectSP valobj_sp);

// MSVC STL std::unique_ptr<>
bool IsMsvcStlUniquePtr(ValueObject &valobj);
bool MsvcStlUniquePtrSummaryProvider(ValueObject &valobj, Stream &stream,
                                     const TypeSummaryOptions &options);

lldb_private::SyntheticChildrenFrontEnd *
MsvcStlUniquePtrSyntheticFrontEndCreator(lldb::ValueObjectSP valobj_sp);

// MSVC STL std::tuple<>
bool IsMsvcStlTuple(ValueObject &valobj);
SyntheticChildrenFrontEnd *
MsvcStlTupleSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                     lldb::ValueObjectSP valobj_sp);

// MSVC STL std::vector<>
bool IsMsvcStlVector(ValueObject &valobj);
lldb_private::SyntheticChildrenFrontEnd *
MsvcStlVectorSyntheticFrontEndCreator(lldb::ValueObjectSP valobj_sp);

// MSVC STL std::list and std::forward_list
bool IsMsvcStlList(ValueObject &valobj);
SyntheticChildrenFrontEnd *
MsvcStlForwardListSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                           lldb::ValueObjectSP valobj_sp);
SyntheticChildrenFrontEnd *
MsvcStlListSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                    lldb::ValueObjectSP valobj_sp);

// MSVC STL std::optional<>
bool IsMsvcStlOptional(ValueObject &valobj);
SyntheticChildrenFrontEnd *
MsvcStlOptionalSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                        lldb::ValueObjectSP valobj_sp);

// MSVC STL std::variant<>
bool IsMsvcStlVariant(ValueObject &valobj);
bool MsvcStlVariantSummaryProvider(ValueObject &valobj, Stream &stream,
                                   const TypeSummaryOptions &options);
SyntheticChildrenFrontEnd *
MsvcStlVariantSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                       lldb::ValueObjectSP valobj_sp);

// MSVC STL std::atomic<>
bool IsMsvcStlAtomic(ValueObject &valobj);
bool MsvcStlAtomicSummaryProvider(ValueObject &valobj, Stream &stream,
                                  const TypeSummaryOptions &options);
SyntheticChildrenFrontEnd *
MsvcStlAtomicSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                      lldb::ValueObjectSP valobj_sp);

// MSVC STL std::unordered_(multi){map|set}<>
bool IsMsvcStlUnordered(ValueObject &valobj);
SyntheticChildrenFrontEnd *
MsvcStlUnorderedSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                         lldb::ValueObjectSP valobj_sp);
bool IsMsvcStlTreeIter(ValueObject &valobj);
bool MsvcStlTreeIterSummaryProvider(ValueObject &valobj, Stream &stream,
                                    const TypeSummaryOptions &options);
lldb_private::SyntheticChildrenFrontEnd *
MsvcStlTreeIterSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                        lldb::ValueObjectSP valobj_sp);

// std::map,set,multimap,multiset
bool IsMsvcStlMapLike(ValueObject &valobj);
lldb_private::SyntheticChildrenFrontEnd *
MsvcStlMapLikeSyntheticFrontEndCreator(lldb::ValueObjectSP valobj_sp);

// MSVC STL std::deque<>
bool IsMsvcStlDeque(ValueObject &valobj);
SyntheticChildrenFrontEnd *
MsvcStlDequeSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                     lldb::ValueObjectSP valobj_sp);

} // namespace formatters
} // namespace lldb_private

#endif
