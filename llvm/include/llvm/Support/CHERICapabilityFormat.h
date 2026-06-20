//===--- CHERICapabilityFormat.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CHERICAPABILITYFORMAT_H
#define LLVM_SUPPORT_CHERICAPABILITYFORMAT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Alignment.h"

namespace llvm {

template <typename Derived, typename AddressType>
struct CHERICapabilityFormatBase {
  CHERICapabilityFormatBase() = delete;

  static constexpr AddressType AddressMask = ~static_cast<AddressType>(0);

  /// Returns the "alignment mask" for an allocation of size \p Length. This
  /// mask is 0 where the capability format alignment requires the
  /// address to be 0, and 1 otherwise.
  LLVM_ABI static AddressType getAlignmentMask(AddressType Length);

  /// Returns the required alignment for an allocation of size \p Length.
  LLVM_ABI static Align getRequiredAlignment(AddressType Length);

  /// Returns \p Length rounded up to the nearest representable allocation
  /// length.
  LLVM_ABI static AddressType getRepresentableLength(AddressType Length);
};

template <typename AddressType, unsigned MW, unsigned MAX_E>
struct RVYCapabilityFormat
    : public CHERICapabilityFormatBase<
          RVYCapabilityFormat<AddressType, MW, MAX_E>, AddressType> {
  LLVM_ABI static AddressType getAlignmentMask(uint64_t Length);
};

using RV32YCapabilityFormat = RVYCapabilityFormat<uint32_t, 10, 24>;
using RV64YCapabilityFormat = RVYCapabilityFormat<uint64_t, 14, 52>;

struct CHERIoTCapabilityFormat
    : public CHERICapabilityFormatBase<CHERIoTCapabilityFormat, uint32_t> {
  LLVM_ABI static uint32_t getAlignmentMask(uint32_t Length);
};

} // namespace llvm

#endif
