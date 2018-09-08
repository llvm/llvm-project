//===-- llvm/ADT/bit.h - C++20 <bit> ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the C++20 <bit> header.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_BIT_H
#define LLVM_ADT_BIT_H

#include "llvm/Support/type_traits.h"
#include <cstring>

namespace llvm {

template <typename To, typename From,
          typename = typename std::enable_if<sizeof(To) == sizeof(From)>::type,
          typename = typename std::enable_if<isPodLike<To>::value>::type,
          typename = typename std::enable_if<isPodLike<From>::value>::type>
inline To bit_cast(const From &from) noexcept {
  alignas(To) unsigned char storage[sizeof(To)];
  std::memcpy(&storage, &from, sizeof(To));
  return reinterpret_cast<To &>(storage);
}

} // namespace llvm

#endif
