//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Extensions on SB API.
//===----------------------------------------------------------------------===//

#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBThreadCollection.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <iterator>

namespace lldb {

/// An iterator helper for iterating over various SB API containers.
template <typename Container, typename Item, typename Index, auto Get>
class iter
    : public llvm::iterator_facade_base<iter<Container, Item, Index, Get>,
                                        std::random_access_iterator_tag, Item,
                                        Index> {
public:
  iter(const Container &container, Index index)
      : container(container), index(index) {}

  Item operator*() { return std::invoke(Get, container, index); }
  Item operator*() const { return std::invoke(Get, container, index); }
  iter &operator+=(Index N) {
    index += N;
    return *this;
  }
  iter &operator-=(Index N) {
    index -= N;
    return *this;
  }
  Index operator-(const iter &other) const { return index - other.index; }
  bool operator==(const iter &other) const { return index == other.index; }
  bool operator!=(const iter &other) const { return !(*this == other); }
  bool operator<(const iter &other) const { return index < other.index; }

private:
  Container container;
  Index index;
};

/// SBProcess thread iterator.
using process_thread_iter =
    iter<SBProcess, SBThread, size_t, &SBProcess::GetThreadAtIndex>;
inline process_thread_iter begin(SBProcess P) { return {P, 0}; }
inline process_thread_iter end(SBProcess P) { return {P, P.GetNumThreads()}; }

/// SBThreadCollection thread iterator.
using thread_collection_iter = iter<SBThreadCollection, SBThread, size_t,
                                    &SBThreadCollection::GetThreadAtIndex>;
inline thread_collection_iter begin(SBThreadCollection TC) { return {TC, 0}; }
inline thread_collection_iter end(SBThreadCollection TC) {
  return {TC, TC.GetSize()};
}

/// SBThread frame iterator.
using frame_iter =
    iter<SBThread, SBFrame, uint32_t, &SBThread::GetFrameAtIndex>;
inline frame_iter begin(SBThread T) { return {T, 0}; }
inline frame_iter end(SBThread T) { return {T, T.GetNumFrames()}; }

/// SBValue value iterators.
/// @{
using value_iter = iter<SBValue, SBValue, uint32_t,
                        static_cast<SBValue (SBValue::*)(uint32_t)>(
                            &SBValue::GetChildAtIndex)>;
inline value_iter begin(SBValue &T) { return {T, 0}; }
inline value_iter end(SBValue &T) { return {T, T.GetNumChildren()}; }
/// @}

/// SBValue value iterators.
/// @{
using value_list_iter =
    iter<SBValueList, SBValue, uint32_t, &SBValueList::GetValueAtIndex>;
inline value_list_iter begin(SBValueList &T) { return {T, 0}; }
inline value_list_iter end(SBValueList &T) { return {T, T.GetSize()}; }
/// @}

// llvm::raw_ostream print helpers.

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, SBStream &stream) {
  OS << llvm::StringRef{stream.GetData(), stream.GetSize()};
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, SBFrame &frame) {
  SBStream stream;
  if (frame.GetDescription(stream))
    OS << stream;
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, SBValue &value) {
  SBStream stream;
  if (value.GetDescription(stream))
    OS << stream;
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const SBStructuredData &data) {
  SBStream stream;
  if (data.GetDescription(stream))
    OS << stream;
  return OS;
}

} // namespace lldb
