//===- KernelArgInfo.h - Kernel argument info -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures used to encode kernel information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_KERNELARGINFO_H
#define LLVM_TRANSFORMS_UTILS_KERNELARGINFO_H

#include "llvm/Support/Endian.h"
#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>

namespace llvm {

struct KernelArgInfo {
public:
  using EncodeType = uint32_t;

private:
  static constexpr unsigned PayloadShift = 8;
  static constexpr EncodeType KindMask =
      static_cast<EncodeType>((1u << PayloadShift) - 1);

  EncodeType Encoded;

public:
  enum class Kind : uint8_t {
    Integer = 0,
    Float = 1,
    Double = 2,
    Pointer = 3,
    Unknown = KindMask,
  };

private:
  explicit KernelArgInfo(Kind K, EncodeType Payload)
      : Encoded((Payload << PayloadShift) | static_cast<EncodeType>(K)) {}

  explicit KernelArgInfo(EncodeType Encoded) : Encoded(Encoded) {}

public:
  static KernelArgInfo getIntegerTy(EncodeType BitWidth) {
    return KernelArgInfo(Kind::Integer, BitWidth);
  }

  static KernelArgInfo getFloatTy() { return KernelArgInfo(Kind::Float, 0); }

  static KernelArgInfo getDoubleTy() { return KernelArgInfo(Kind::Double, 0); }

  static KernelArgInfo getPointerTy() {
    return KernelArgInfo(Kind::Pointer, 0);
  }

  static KernelArgInfo getUnknownTy() {
    return KernelArgInfo(Kind::Unknown, 0);
  }

  Kind getKind() const { return static_cast<Kind>(Encoded & KindMask); }

  EncodeType getIntegerBitWidth() const { return Encoded >> PayloadShift; }

  EncodeType getEncodedLE() const {
    EncodeType E;
    support::endian::write32le(&E, Encoded);
    return E;
  }

  static KernelArgInfo fromEncodedLE(const void *P) {
    return KernelArgInfo(support::endian::read32le(P));
  }

  std::string typeStr() const {
    switch (getKind()) {
    case Kind::Integer:
      return "i" + std::to_string(getIntegerBitWidth());
    case Kind::Float:
      return "float";
    case Kind::Double:
      return "double";
    case Kind::Pointer:
      return "ptr";
    case Kind::Unknown:
      return "unknown";
    }
  }

  std::string valueStr(void *Value) const {
    assert(Value);
    switch (getKind()) {
    case Kind::Integer:
      switch (getIntegerBitWidth()) {
      case 8:
        return std::to_string(*reinterpret_cast<uint8_t *>(Value));
      case 16:
        return std::to_string(*reinterpret_cast<uint16_t *>(Value));
      case 32:
        return std::to_string(*reinterpret_cast<uint32_t *>(Value));
      case 64:
        return std::to_string(*reinterpret_cast<uint64_t *>(Value));
      default:
        return "<unsupported bit width>";
      }
    case Kind::Float:
      return std::to_string(*reinterpret_cast<float *>(Value));
    case Kind::Double:
      return std::to_string(*reinterpret_cast<double *>(Value));
    case Kind::Pointer: {
      std::ostringstream oss;
      oss << *reinterpret_cast<void **>(Value);
      return oss.str();
    }
    case Kind::Unknown:
      return "<no representation>";
    }
  }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_KERNELARGINFO_H
