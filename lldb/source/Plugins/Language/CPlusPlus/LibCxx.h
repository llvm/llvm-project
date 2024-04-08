//===-- LibCxx.h ---------------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_LIBCXX_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_LIBCXX_H

#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Utility/Stream.h"

namespace lldb_private {
namespace formatters {

/// Find a child member of \c obj_sp, trying all alternative names in order.
lldb::ValueObjectSP
GetChildMemberWithName(ValueObject &obj,
                       llvm::ArrayRef<ConstString> alternative_names);

lldb::ValueObjectSP GetFirstValueOfLibCXXCompressedPair(ValueObject &pair);
lldb::ValueObjectSP GetSecondValueOfLibCXXCompressedPair(ValueObject &pair);


bool LibcxxStringSummaryProviderASCII(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options); // libc++ std::string

bool LibcxxStringSummaryProviderUTF16(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options); // libc++ std::u16string

bool LibcxxStringSummaryProviderUTF32(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options); // libc++ std::u32string

bool LibcxxWStringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::wstring

bool LibcxxStringViewSummaryProviderASCII(
    ValueObject &valueObj, Stream &stream,
    const TypeSummaryOptions &summary_options); // libc++ std::string_view

bool LibcxxStringViewSummaryProviderUTF16(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options); // libc++ std::u16string_view

bool LibcxxStringViewSummaryProviderUTF32(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options); // libc++ std::u32string_view

bool LibcxxWStringViewSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::wstring_view

bool LibcxxStdSliceArraySummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::slice_array

bool LibcxxSmartPointerSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions
        &options); // libc++ std::shared_ptr<> and std::weak_ptr<>

// libc++ std::unique_ptr<>
bool LibcxxUniquePointerSummaryProvider(ValueObject &valobj, Stream &stream,
                                        const TypeSummaryOptions &options);

bool LibcxxFunctionSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::function<>

SyntheticChildrenFrontEnd *
LibcxxVectorBoolSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                         lldb::ValueObjectSP);

bool LibcxxContainerSummaryProvider(ValueObject &valobj, Stream &stream,
                                    const TypeSummaryOptions &options);

/// Formatter for libc++ std::span<>.
bool LibcxxSpanSummaryProvider(ValueObject &valobj, Stream &stream,
                               const TypeSummaryOptions &options);

class LibCxxMapIteratorSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  LibCxxMapIteratorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(ConstString name) override;

  ~LibCxxMapIteratorSyntheticFrontEnd() override;

private:
  ValueObject *m_pair_ptr;
  lldb::ValueObjectSP m_pair_sp;
};

SyntheticChildrenFrontEnd *
LibCxxMapIteratorSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                          lldb::ValueObjectSP);

/// Formats libcxx's std::unordered_map iterators
///
/// In raw form a std::unordered_map::iterator is represented as follows:
///
/// (lldb) var it --raw --ptr-depth 1
/// (std::__1::__hash_map_iterator<
///    std::__1::__hash_iterator<
///      std::__1::__hash_node<
///        std::__1::__hash_value_type<
///            std::__1::basic_string<char, std::__1::char_traits<char>,
///            std::__1::allocator<char> >, std::__1::basic_string<char,
///            std::__1::char_traits<char>, std::__1::allocator<char> > >,
///        void *> *> >)
///  it = {
///   __i_ = {
///     __node_ = 0x0000600001700040 {
///       __next_ = 0x0000600001704000
///     }
///   }
/// }
class LibCxxUnorderedMapIteratorSyntheticFrontEnd
    : public SyntheticChildrenFrontEnd {
public:
  LibCxxUnorderedMapIteratorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~LibCxxUnorderedMapIteratorSyntheticFrontEnd() override = default;

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(ConstString name) override;

private:
  ValueObject *m_iter_ptr = nullptr; ///< Held, not owned. Child of iterator
                                     ///< ValueObject supplied at construction.

  lldb::ValueObjectSP m_pair_sp; ///< ValueObject for the key/value pair
                                 ///< that the iterator currently points
                                 ///< to.
};

SyntheticChildrenFrontEnd *
LibCxxUnorderedMapIteratorSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                                   lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibCxxVectorIteratorSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                             lldb::ValueObjectSP);

class LibcxxSharedPtrSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  LibcxxSharedPtrSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(ConstString name) override;

  ~LibcxxSharedPtrSyntheticFrontEnd() override;

private:
  ValueObject *m_cntrl;
};

class LibcxxUniquePtrSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  LibcxxUniquePtrSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(ConstString name) override;

  ~LibcxxUniquePtrSyntheticFrontEnd() override;

private:
  lldb::ValueObjectSP m_value_ptr_sp;
  lldb::ValueObjectSP m_deleter_sp;
};

SyntheticChildrenFrontEnd *
LibcxxBitsetSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                     lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxSharedPtrSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                        lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxUniquePtrSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                        lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxStdVectorSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                        lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxStdValarraySyntheticFrontEndCreator(CXXSyntheticChildren *,
                                          lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxStdSliceArraySyntheticFrontEndCreator(CXXSyntheticChildren *,
                                            lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxStdListSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                      lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxStdForwardListSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                             lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxStdMapSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                     lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxStdUnorderedMapSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                              lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxInitializerListSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                              lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *LibcxxQueueFrontEndCreator(CXXSyntheticChildren *,
                                                      lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *LibcxxTupleFrontEndCreator(CXXSyntheticChildren *,
                                                      lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxOptionalSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                       lldb::ValueObjectSP valobj_sp);

SyntheticChildrenFrontEnd *
LibcxxVariantFrontEndCreator(CXXSyntheticChildren *,
                             lldb::ValueObjectSP valobj_sp);

SyntheticChildrenFrontEnd *
LibcxxStdSpanSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                      lldb::ValueObjectSP);

SyntheticChildrenFrontEnd *
LibcxxStdRangesRefViewSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                               lldb::ValueObjectSP);

bool LibcxxChronoSysSecondsSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::chrono::sys_seconds

bool LibcxxChronoSysDaysSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::chrono::sys_days

bool LibcxxChronoMonthSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::chrono::month

bool LibcxxChronoWeekdaySummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::chrono::weekday

bool LibcxxChronoYearMonthDaySummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::chrono::year_month_day

} // namespace formatters
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_LIBCXX_H
