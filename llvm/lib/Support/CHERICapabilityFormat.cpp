//===- CHERICapabilityFormat.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CHERICapabilityFormat.h"
#include "llvm/ADT/bit.h"

namespace llvm {

template <typename AddressType, unsigned MW, unsigned MAX_E>
AddressType RVYCapabilityFormat<AddressType, MW, MAX_E>::getAlignmentMaskImpl(
    uint64_t Length) {
  static constexpr unsigned int IE_TAKE_BITS = 3;

  if (Length == 0)
    return RVYCapabilityFormat::AddressMask;

  // Extract bits that overflow the uncompressed mantissa window.
  uint64_t Slice = static_cast<uint64_t>(Length) >> (MW - 1);
  unsigned int E = 64 - llvm::countl_zero(Slice);
  // We use internal exponent if length overflows OR the denormal boundary
  // bit is set.
  bool IE = (E != 0) || ((static_cast<uint64_t>(Length) >> (MW - 2)) & 1);
  // Include bits used by the internal exponent for the shift value.
  unsigned int Eprime = IE ? (E + IE_TAKE_BITS) : 0;

  assert(E <= MAX_E && "Raw exponent exceeds architecture maximum");
  assert(Eprime <= sizeof(AddressType) * 8 &&
         "Shift amount exceeds integer width");

  // Left-shift ~0 to mask out the lost precision bits
  return RVYCapabilityFormat::AddressMask << Eprime;
}

template struct CHERICapabilityFormatBase<RVYCapabilityFormat<uint32_t, 10, 24>,
                                          uint32_t>;
template struct CHERICapabilityFormatBase<RVYCapabilityFormat<uint64_t, 14, 52>,
                                          uint64_t>;
template struct RVYCapabilityFormat<uint32_t, 10, 24>;
template struct RVYCapabilityFormat<uint64_t, 14, 52>;

uint32_t CHERIoTCapabilityFormat::getAlignmentMaskImpl(uint32_t Length) {
  // Per section 7.13.4 and table 7.4 in the v1.0 CHERIoT specification.
  constexpr uint32_t NINE_SET_BITS = 511;
  uint32_t E;
  if (Length > NINE_SET_BITS << 14)
    E = 24;
  else {
    E = Length > NINE_SET_BITS ? 32 - llvm::countl_zero(Length) - 9 : 0;
    if (Length > NINE_SET_BITS << E)
      ++E;
    assert(E <= 14 && "CHERIoT capabilities cannot encode E between 14 and 24");
  }
  return CHERIoTCapabilityFormat::AddressMask << E;
}

template struct CHERICapabilityFormatBase<CHERIoTCapabilityFormat, uint32_t>;

} // namespace llvm
