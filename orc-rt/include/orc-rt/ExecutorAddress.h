//===------ ExecutorAddress.h - Executing process address -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilites for representing addresses and address ranges in the executing
// program that can be shared with an ORC controller.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_EXECUTORADDRESS_H
#define ORC_RT_EXECUTORADDRESS_H

#include "span.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace orc_rt {

using ExecutorAddrDiff = uint64_t;

/// Represents an address in the executor process.
class ExecutorAddr {
public:
  /// Return pointer unmodified.
  template <typename T> struct rawPtr {
    T *operator()(T *p) const { return p; }
  };

  /// Default wrap function to use on this host.
  template <typename T> using defaultWrap = rawPtr<T>;

  /// Default unwrap function to use on this host.
  template <typename T> using defaultUnwrap = rawPtr<T>;

  /// Merges a tag into the raw address value:
  ///   P' = P | (TagValue << TagOffset).
  class Tag {
  public:
    constexpr Tag(uintptr_t TagValue, uintptr_t TagOffset)
        : TagMask(TagValue << TagOffset) {}

    template <typename T> constexpr T *operator()(T *P) {
      return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(P) | TagMask);
    }

  private:
    uintptr_t TagMask;
  };

  /// Strips a tag of the given length from the given offset within the pointer:
  /// P' = P & ~(((1 << TagLen) -1) << TagOffset)
  class Untag {
  public:
    constexpr Untag(uintptr_t TagLen, uintptr_t TagOffset)
        : UntagMask(~(((uintptr_t(1) << TagLen) - 1) << TagOffset)) {}

    template <typename T> constexpr T *operator()(T *P) {
      return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(P) & UntagMask);
    }

  private:
    uintptr_t UntagMask;
  };

  constexpr ExecutorAddr() noexcept = default;
  explicit constexpr ExecutorAddr(uint64_t Addr) noexcept : Addr(Addr) {}

  /// Create an ExecutorAddr from the given pointer.
  template <typename T, typename UnwrapFn = defaultUnwrap<T>>
  static constexpr ExecutorAddr fromPtr(T *Ptr,
                                        UnwrapFn &&Unwrap = UnwrapFn()) {
    return ExecutorAddr(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Unwrap(Ptr))));
  }

  /// Cast this ExecutorAddr to a pointer of the given type.
  template <typename T, typename WrapFn = defaultWrap<std::remove_pointer_t<T>>>
  constexpr std::enable_if_t<std::is_pointer<T>::value, T>
  toPtr(WrapFn &&Wrap = WrapFn()) const {
    uintptr_t IntPtr = static_cast<uintptr_t>(Addr);
    assert(IntPtr == Addr && "ExecutorAddr value out of range for uintptr_t");
    return Wrap(reinterpret_cast<T>(IntPtr));
  }

  /// Cast this ExecutorAddr to a pointer of the given function type.
  template <typename T, typename WrapFn = defaultWrap<T>>
  constexpr std::enable_if_t<std::is_function<T>::value, T *>
  toPtr(WrapFn &&Wrap = WrapFn()) const {
    uintptr_t IntPtr = static_cast<uintptr_t>(Addr);
    assert(IntPtr == Addr && "ExecutorAddr value out of range for uintptr_t");
    return Wrap(reinterpret_cast<T *>(IntPtr));
  }

  constexpr uint64_t getValue() const noexcept { return Addr; }
  constexpr void setValue(uint64_t Addr) noexcept { this->Addr = Addr; }
  constexpr bool isNull() const noexcept { return Addr == 0; }

  constexpr explicit operator bool() const noexcept { return Addr != 0; }

  friend constexpr bool operator==(const ExecutorAddr &LHS,
                                   const ExecutorAddr &RHS) noexcept {
    return LHS.Addr == RHS.Addr;
  }

  friend constexpr bool operator!=(const ExecutorAddr &LHS,
                                   const ExecutorAddr &RHS) noexcept {
    return LHS.Addr != RHS.Addr;
  }

  friend constexpr bool operator<(const ExecutorAddr &LHS,
                                  const ExecutorAddr &RHS) noexcept {
    return LHS.Addr < RHS.Addr;
  }

  friend constexpr bool operator<=(const ExecutorAddr &LHS,
                                   const ExecutorAddr &RHS) noexcept {
    return LHS.Addr <= RHS.Addr;
  }

  friend constexpr bool operator>(const ExecutorAddr &LHS,
                                  const ExecutorAddr &RHS) noexcept {
    return LHS.Addr > RHS.Addr;
  }

  friend constexpr bool operator>=(const ExecutorAddr &LHS,
                                   const ExecutorAddr &RHS) noexcept {
    return LHS.Addr >= RHS.Addr;
  }

  constexpr ExecutorAddr &operator++() noexcept {
    ++Addr;
    return *this;
  }
  constexpr ExecutorAddr &operator--() noexcept {
    --Addr;
    return *this;
  }
  constexpr ExecutorAddr operator++(int) noexcept {
    return ExecutorAddr(Addr++);
  }
  constexpr ExecutorAddr operator--(int) noexcept {
    return ExecutorAddr(Addr++);
  }

  constexpr ExecutorAddr &operator+=(const ExecutorAddrDiff Delta) noexcept {
    Addr += Delta;
    return *this;
  }

  constexpr ExecutorAddr &operator-=(const ExecutorAddrDiff Delta) noexcept {
    Addr -= Delta;
    return *this;
  }

private:
  uint64_t Addr = 0;
};

/// Subtracting two addresses yields an offset.
inline constexpr ExecutorAddrDiff operator-(const ExecutorAddr &LHS,
                                            const ExecutorAddr &RHS) noexcept {
  return ExecutorAddrDiff(LHS.getValue() - RHS.getValue());
}

/// Adding an offset and an address yields an address.
inline constexpr ExecutorAddr operator+(const ExecutorAddr &LHS,
                                        const ExecutorAddrDiff &RHS) noexcept {
  return ExecutorAddr(LHS.getValue() + RHS);
}

/// Adding an address and an offset yields an address.
inline constexpr ExecutorAddr operator+(const ExecutorAddrDiff &LHS,
                                        const ExecutorAddr &RHS) noexcept {
  return ExecutorAddr(LHS + RHS.getValue());
}

/// Represents an address range in the exceutor process.
struct ExecutorAddrRange {
  constexpr ExecutorAddrRange() noexcept = default;
  constexpr ExecutorAddrRange(ExecutorAddr Start, ExecutorAddr End) noexcept
      : Start(Start), End(End) {}
  constexpr ExecutorAddrRange(ExecutorAddr Start,
                              ExecutorAddrDiff Size) noexcept
      : Start(Start), End(Start + Size) {}

  constexpr bool empty() const noexcept { return Start == End; }
  constexpr ExecutorAddrDiff size() const noexcept { return End - Start; }

  friend constexpr bool operator==(const ExecutorAddrRange &LHS,
                                   const ExecutorAddrRange &RHS) noexcept {
    return LHS.Start == RHS.Start && LHS.End == RHS.End;
  }
  friend constexpr bool operator!=(const ExecutorAddrRange &LHS,
                                   const ExecutorAddrRange &RHS) noexcept {
    return !(LHS == RHS);
  }
  constexpr bool contains(ExecutorAddr Addr) const noexcept {
    return Start <= Addr && Addr < End;
  }
  constexpr bool contains(const ExecutorAddrRange &Other) const noexcept {
    return (Other.Start >= Start && Other.End <= End);
  }
  constexpr bool overlaps(const ExecutorAddrRange &Other) const noexcept {
    return !(Other.End <= Start || End <= Other.Start);
  }

  template <typename T> constexpr span<T> toSpan() const noexcept {
    assert(size() % sizeof(T) == 0 &&
           "AddressRange is not a multiple of sizeof(T)");
    return span<T>(Start.toPtr<T *>(), size() / sizeof(T));
  }

  ExecutorAddr Start;
  ExecutorAddr End;
};

} // namespace orc_rt

// Make ExecutorAddr hashable.
template <> struct std::hash<orc_rt::ExecutorAddr> {
  constexpr size_t operator()(const orc_rt::ExecutorAddr &A) const noexcept {
    return std::hash<uint64_t>()(A.getValue());
  }
};

#endif // ORC_RT_EXECUTORADDRESS_H
