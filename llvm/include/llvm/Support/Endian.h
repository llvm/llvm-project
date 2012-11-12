//===- Endian.h - Utilities for IO with endian specific data ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares generic functions to read and write endian specific data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ENDIAN_H
#define LLVM_SUPPORT_ENDIAN_H

#include "llvm/Support/Host.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/type_traits.h"

namespace llvm {
namespace support {

enum endianness {big, little};
enum alignment {unaligned, aligned};

namespace detail {

template<typename value_type, alignment align>
struct alignment_access_helper;

template<typename value_type>
struct alignment_access_helper<value_type, aligned>
{
  value_type val;
};

// Provides unaligned loads and stores.
#pragma pack(push)
#pragma pack(1)
template<typename value_type>
struct alignment_access_helper<value_type, unaligned>
{
  value_type val;
};
#pragma pack(pop)

} // end namespace detail

namespace endian {
  template<typename value_type, alignment align>
  inline value_type read_le(const void *memory) {
    value_type t =
      reinterpret_cast<const detail::alignment_access_helper
        <value_type, align> *>(memory)->val;
    if (sys::isBigEndianHost())
      return sys::SwapByteOrder(t);
    return t;
  }

  template<typename value_type, alignment align>
  inline void write_le(void *memory, value_type value) {
    if (sys::isBigEndianHost())
      value = sys::SwapByteOrder(value);
    reinterpret_cast<detail::alignment_access_helper<value_type, align> *>
      (memory)->val = value;
  }

  template<typename value_type, alignment align>
  inline value_type read_be(const void *memory) {
    value_type t =
      reinterpret_cast<const detail::alignment_access_helper
        <value_type, align> *>(memory)->val;
    if (sys::isLittleEndianHost())
      return sys::SwapByteOrder(t);
    return t;
  }

  template<typename value_type, alignment align>
  inline void write_be(void *memory, value_type value) {
    if (sys::isLittleEndianHost())
      value = sys::SwapByteOrder(value);
    reinterpret_cast<detail::alignment_access_helper<value_type, align> *>
      (memory)->val = value;
  }
}

namespace detail {

template<typename value_type,
         endianness endian,
         alignment  align>
class packed_endian_specific_integral;

template<typename value_type>
class packed_endian_specific_integral<value_type, little, unaligned> {
public:
  operator value_type() const {
    return endian::read_le<value_type, unaligned>(Value);
  }
  void operator=(value_type newValue) {
    endian::write_le<value_type, unaligned>((void *)&Value, newValue);
  }
private:
  uint8_t Value[sizeof(value_type)];
};

template<typename value_type>
class packed_endian_specific_integral<value_type, big, unaligned> {
public:
  operator value_type() const {
    return endian::read_be<value_type, unaligned>(Value);
  }
  void operator=(value_type newValue) {
    endian::write_be<value_type, unaligned>((void *)&Value, newValue);
  }
private:
  uint8_t Value[sizeof(value_type)];
};

template<typename value_type>
class packed_endian_specific_integral<value_type, little, aligned> {
public:
  operator value_type() const {
    return endian::read_le<value_type, aligned>(&Value);
  }
  void operator=(value_type newValue) {
    endian::write_le<value_type, aligned>((void *)&Value, newValue);
  }
private:
  value_type Value;
};

template<typename value_type>
class packed_endian_specific_integral<value_type, big, aligned> {
public:
  operator value_type() const {
    return endian::read_be<value_type, aligned>(&Value);
  }
  void operator=(value_type newValue) {
    endian::write_be<value_type, aligned>((void *)&Value, newValue);
  }
private:
  value_type Value;
};

} // end namespace detail

typedef detail::packed_endian_specific_integral
                  <uint8_t, little, unaligned>  ulittle8_t;
typedef detail::packed_endian_specific_integral
                  <uint16_t, little, unaligned> ulittle16_t;
typedef detail::packed_endian_specific_integral
                  <uint32_t, little, unaligned> ulittle32_t;
typedef detail::packed_endian_specific_integral
                  <uint64_t, little, unaligned> ulittle64_t;

typedef detail::packed_endian_specific_integral
                   <int8_t, little, unaligned>  little8_t;
typedef detail::packed_endian_specific_integral
                   <int16_t, little, unaligned> little16_t;
typedef detail::packed_endian_specific_integral
                   <int32_t, little, unaligned> little32_t;
typedef detail::packed_endian_specific_integral
                   <int64_t, little, unaligned> little64_t;

typedef detail::packed_endian_specific_integral
                    <uint8_t, little, aligned>  aligned_ulittle8_t;
typedef detail::packed_endian_specific_integral
                    <uint16_t, little, aligned> aligned_ulittle16_t;
typedef detail::packed_endian_specific_integral
                    <uint32_t, little, aligned> aligned_ulittle32_t;
typedef detail::packed_endian_specific_integral
                    <uint64_t, little, aligned> aligned_ulittle64_t;

typedef detail::packed_endian_specific_integral
                     <int8_t, little, aligned>  aligned_little8_t;
typedef detail::packed_endian_specific_integral
                     <int16_t, little, aligned> aligned_little16_t;
typedef detail::packed_endian_specific_integral
                     <int32_t, little, aligned> aligned_little32_t;
typedef detail::packed_endian_specific_integral
                     <int64_t, little, aligned> aligned_little64_t;

typedef detail::packed_endian_specific_integral
                  <uint8_t, big, unaligned>     ubig8_t;
typedef detail::packed_endian_specific_integral
                  <uint16_t, big, unaligned>    ubig16_t;
typedef detail::packed_endian_specific_integral
                  <uint32_t, big, unaligned>    ubig32_t;
typedef detail::packed_endian_specific_integral
                  <uint64_t, big, unaligned>    ubig64_t;

typedef detail::packed_endian_specific_integral
                   <int8_t, big, unaligned>     big8_t;
typedef detail::packed_endian_specific_integral
                   <int16_t, big, unaligned>    big16_t;
typedef detail::packed_endian_specific_integral
                   <int32_t, big, unaligned>    big32_t;
typedef detail::packed_endian_specific_integral
                   <int64_t, big, unaligned>    big64_t;

typedef detail::packed_endian_specific_integral
                    <uint8_t, big, aligned>     aligned_ubig8_t;
typedef detail::packed_endian_specific_integral
                    <uint16_t, big, aligned>    aligned_ubig16_t;
typedef detail::packed_endian_specific_integral
                    <uint32_t, big, aligned>    aligned_ubig32_t;
typedef detail::packed_endian_specific_integral
                    <uint64_t, big, aligned>    aligned_ubig64_t;

typedef detail::packed_endian_specific_integral
                     <int8_t, big, aligned>     aligned_big8_t;
typedef detail::packed_endian_specific_integral
                     <int16_t, big, aligned>    aligned_big16_t;
typedef detail::packed_endian_specific_integral
                     <int32_t, big, aligned>    aligned_big32_t;
typedef detail::packed_endian_specific_integral
                     <int64_t, big, aligned>    aligned_big64_t;

} // end namespace llvm
} // end namespace support

#endif
