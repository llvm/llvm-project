//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPTION_UNSPLITLIST_H
#define LLVM_OPTION_UNSPLITLIST_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
namespace opt {

/// The member type of an UnsplitListField: an ArrayRef whose elements are
/// whole values, commas included.
template <typename T> struct UnsplitList : ArrayRef<T> {
  using ArrayRef<T>::ArrayRef;
  constexpr UnsplitList() = default;
};

} // namespace opt
} // namespace llvm

#endif // LLVM_OPTION_UNSPLITLIST_H
