//==----------------- stream.hpp - SYCL standard header file ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/stream_impl.hpp>

namespace cl {
namespace sycl {

enum class stream_manipulator {
  dec,
  hex,
  oct,
  noshowbase,
  showbase,
  noshowpos,
  showpos,
  endl,
  fixed,
  scientific,
  hexfloat,
  defaultfloat
};

const stream_manipulator dec = stream_manipulator::dec;

const stream_manipulator hex = stream_manipulator::hex;

const stream_manipulator oct = stream_manipulator::oct;

const stream_manipulator noshowbase = stream_manipulator::noshowbase;

const stream_manipulator showbase = stream_manipulator::showbase;

const stream_manipulator noshowpos = stream_manipulator::noshowpos;

const stream_manipulator showpos = stream_manipulator::showpos;

const stream_manipulator endl = stream_manipulator::endl;

const stream_manipulator fixed = stream_manipulator::fixed;

const stream_manipulator scientific = stream_manipulator::scientific;

const stream_manipulator hexfloat = stream_manipulator::hexfloat;

const stream_manipulator defaultfloat = stream_manipulator::defaultfloat;

class stream;

class __precision_manipulator__ {
  int Precision_;

public:
  __precision_manipulator__(int Precision) : Precision_(Precision) {}
  friend const stream &operator<<(const stream &,
                                  const __precision_manipulator__ &);
};

class __width_manipulator__ {
  int Width_;

public:
  __width_manipulator__(int Width) : Width_(Width) {}
  friend const stream &operator<<(const stream &,
                                  const __width_manipulator__ &);
};

inline __precision_manipulator__ setprecision(int Precision) {
  return __precision_manipulator__(Precision);
}

inline __width_manipulator__ setw(int Width) {
  return __width_manipulator__(Width);
}

class stream {
public:
  stream(size_t BufferSize, size_t MaxStatementSize, handler &CGH);

  size_t get_size() const;

  size_t get_max_statement_size() const;

  size_t get_precision() const { return Precision; }

  stream_manipulator get_stream_mode() const { return Manipulator; }

  bool operator==(const stream &RHS) const;

  bool operator!=(const stream &LHS) const;

private:
#ifdef __SYCL_DEVICE_ONLY__
  char padding[sizeof(std::shared_ptr<detail::stream_impl>)];
#else
  std::shared_ptr<detail::stream_impl> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
#endif

  // Accessor to stream buffer
  detail::stream_impl::AccessorType Acc;

  // Atomic accessor to the offset variable. It represents an offset in the
  // stream buffer.
  detail::stream_impl::OffsetAccessorType OffsetAcc;
  mutable stream_manipulator Manipulator = defaultfloat;

  // Fields and methods to work with manipulators

  // Type used for format flags
  using FmtFlags = unsigned int;

  mutable int Precision;
  mutable int Width;
  mutable FmtFlags Flags;

  // Mapping from stream_manipulator to FmtFlags. Each manipulator corresponds
  // to the bit in FmtFlags.
  static constexpr FmtFlags Dec = 0x0001;
  static constexpr FmtFlags Hex = 0x0002;
  static constexpr FmtFlags Oct = 0x0004;
  static constexpr FmtFlags ShowBase = 0x0008;
  static constexpr FmtFlags ShowPos = 0x0010;
  static constexpr FmtFlags Fixed = 0x0020;
  static constexpr FmtFlags Scientific = 0x0040;

  // Bitmask made of the combination of the base flags. Base flags are mutually
  // exclusive, this mask is used to clean base field before setting the new
  // base flag.
  static constexpr FmtFlags BaseField = Dec | Hex | Oct;

  // Bitmask made of the combination of the floating point value format flags.
  // Thease flags are mutually exclusive, this mask is used to clean float field
  // before setting the new float flag.
  static constexpr FmtFlags FloatField = Scientific | Fixed;

  void set_flag(FmtFlags FormatFlag) const { Flags |= FormatFlag; }

  void unset_flag(FmtFlags FormatFlag) const { Flags &= ~FormatFlag; }

  // This method is used to set the flag for base and float manipulators. These
  // flags are mutually exclusive and base/float field needs to be cleared
  // before the setting new flag.
  void set_flag(FmtFlags FormatFlag, FmtFlags Mask) const {
    unset_flag(Mask);
    Flags |= FormatFlag & Mask;
  }

  // Set the flags which correspond to the input stream manipulator.
  void set_manipulator(const stream_manipulator &SM) const {
    switch (SM) {
    case stream_manipulator::dec:
      set_flag(Dec, BaseField);
      break;
    case stream_manipulator::hex:
      set_flag(Hex, BaseField);
      break;
    case stream_manipulator::oct:
      set_flag(Oct, BaseField);
      break;
    case stream_manipulator::noshowbase:
      unset_flag(ShowBase);
      break;
    case stream_manipulator::showbase:
      set_flag(ShowBase);
      break;
    case stream_manipulator::noshowpos:
      unset_flag(ShowPos);
      break;
    case stream_manipulator::showpos:
      set_flag(ShowPos);
      break;
    case stream_manipulator::fixed:
      set_flag(Fixed, FloatField);
      break;
    case stream_manipulator::scientific:
      set_flag(Scientific, FloatField);
      break;
    case stream_manipulator::hexfloat:
      set_flag(Fixed | Scientific, FloatField);
      break;
    case stream_manipulator::defaultfloat:
      unset_flag(FloatField);
      break;
    default:
      // Unknown manipulator
      break;
    }
  }

  friend const stream &operator<<(const stream &, const char);
  friend const stream &operator<<(const stream &, const char *);
  template <typename ValueType>
  friend typename std::enable_if<std::is_integral<ValueType>::value,
                                 const stream &>::type
  operator<<(const stream &, const ValueType &);
  friend const stream &operator<<(const stream &, const stream_manipulator &);

  // Helper method to update offset atomically according to the provided
  // operand size of the output operator. Return true if offset is updated and
  // false in case of overflow.
  bool update_offset(unsigned Size, unsigned &Cur) const {
    unsigned New;
    do {
      Cur = OffsetAcc[0].load();
      if (Acc.get_count() - Cur < Size)
        // Overflow
        return false;
      New = Cur + Size;
    } while (!OffsetAcc[0].compare_exchange_strong(Cur, New));
    return true;
  }
};

// Character
inline const stream &operator<<(const stream &Out, const char C) {
  unsigned Cur;
  if (!Out.update_offset(1, Cur))
    return Out;
  Out.Acc[Cur] = C;
  return Out;
}

// String
inline const stream &operator<<(const stream &Out, const char *Str) {
  unsigned Len;
  for (Len = 0; Str[Len] != '\0'; Len++)
    ;

  unsigned Cur;
  if (!Out.update_offset(Len, Cur))
    return Out;

  for (size_t i = 0; i < Len; i++) {
    Out.Acc[i + Cur] = Str[i];
  }
  return Out;
}

// Boolean
inline const stream &operator<<(const stream &Out, const bool &RHS) {
  Out << (RHS ? "true" : "false");
  return Out;
}

// Integral
template <typename ValueType>
typename std::enable_if<std::is_integral<ValueType>::value,
                        const stream &>::type
operator<<(const stream &Out, const ValueType &RHS) {
  // TODO
  return Out;
}

// Floating points

inline const stream &operator<<(const stream &Out, const float &RHS) {
  // TODO
  return Out;
}

inline const stream &operator<<(const stream &Out, const double &RHS) {
  // TODO
  return Out;
}

inline const stream &operator<<(const stream &Out, const half &RHS) {
  // TODO
  return Out;
}

// Pointer

template <typename ElementType, access::address_space Space>
inline const stream &operator<<(const stream &Out,
                                const multi_ptr<ElementType, Space> &RHS) {
  // TODO
  return Out;
}

template <typename T>
const stream &operator<<(const stream &Out, const T *RHS) {
  // TODO
  return Out;
}

// Manipulators

inline const stream &operator<<(const stream &Out,
                                const __precision_manipulator__ &RHS) {
  // TODO
  return Out;
}

inline const stream &operator<<(const stream &Out,
                                const __width_manipulator__ &RHS) {
  // TODO
  return Out;
}

inline const stream &operator<<(const stream &Out,
                                const stream_manipulator &RHS) {
  switch (RHS) {
  case stream_manipulator::endl:
    Out << '\n';
    break;
  default:
    Out.set_manipulator(RHS);
  }
  return Out;
}

// Vec

template <typename T, int Dimensions>
const stream &operator<<(const stream &Out, const vec<T, Dimensions> &RHS) {
  // TODO
  return Out;
}

// SYCL types

template <int Dimensions>
inline const stream &operator<<(const stream &Out, const id<Dimensions> &RHS) {
  // TODO
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const range<Dimensions> &RHS) {
  // TODO
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const item<Dimensions> &RHS) {
  // TODO
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const nd_range<Dimensions> &RHS) {
  // TODO
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const nd_item<Dimensions> &RHS) {
  // TODO
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const group<Dimensions> &RHS) {
  // TODO
  return Out;
}

} // namespace sycl
} // namespace cl
namespace std {
template <> struct hash<cl::sycl::stream> {
  size_t operator()(const cl::sycl::stream &S) const {
#ifdef __SYCL_DEVICE_ONLY__
    return 0;
#else
    return hash<std::shared_ptr<cl::sycl::detail::stream_impl>>()(
        cl::sycl::detail::getSyclObjImpl(S));
#endif
  }
};
} // namespace std

