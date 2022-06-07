//===-- llvm/BinaryFormat/DXContainer.h - The DXBC file format --*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines manifest constants for the DXContainer object file format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_DXCONTAINER_H
#define LLVM_BINARYFORMAT_DXCONTAINER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SwapByteOrder.h"

#include <stdint.h>

namespace llvm {

// The DXContainer file format is arranged as a header and "parts". Semantically
// parts are similar to sections in other object file formats. The File format
// structure is roughly:

// ┌────────────────────────────────┐
// │             Header             │
// ├────────────────────────────────┤
// │              Part              │
// ├────────────────────────────────┤
// │              Part              │
// ├────────────────────────────────┤
// │              ...               │
// └────────────────────────────────┘

namespace dxbc {

struct Hash {
  uint8_t Digest[16];
};

enum class HashFlags : uint32_t {
  None = 0,           // No flags defined.
  IncludesSource = 1, // This flag indicates that the shader hash was computed
                      // taking into account source information (-Zss)
};

struct ShaderHash {
  uint32_t Flags; // DxilShaderHashFlags
  uint8_t Digest[16];

  void swapBytes() { sys::swapByteOrder(Flags); }
};

struct ContainerVersion {
  uint16_t Major;
  uint16_t Minor;

  void swapBytes() {
    sys::swapByteOrder(Major);
    sys::swapByteOrder(Minor);
  }
};

struct Header {
  uint8_t Magic[4]; // "DXBC"
  Hash FileHash;
  ContainerVersion Version;
  uint32_t FileSize;
  uint32_t PartCount;

  void swapBytes() {
    Version.swapBytes();
    sys::swapByteOrder(FileSize);
    sys::swapByteOrder(PartCount);
  }
  // Structure is followed by part offsets: uint32_t PartOffset[PartCount];
  // The offset is to a PartHeader, which is followed by the Part Data.
};

/// Use this type to describe the size and type of a DXIL container part.
struct PartHeader {
  uint8_t Name[4];
  uint32_t Size;

  void swapBytes() { sys::swapByteOrder(Size); }
  StringRef getName() const {
    return StringRef(reinterpret_cast<const char *>(&Name[0]), 4);
  }
  // Structure is followed directly by part data: uint8_t PartData[PartSize].
};

struct BitcodeHeader {
  uint8_t Magic[4];     // ACSII "DXIL".
  uint8_t MajorVersion; // DXIL version.
  uint8_t MinorVersion; // DXIL version.
  uint16_t Unused;
  uint32_t Offset; // Offset to LLVM bitcode (from start of header).
  uint32_t Size;   // Size of LLVM bitcode (in bytes).
  // Followed by uint8_t[BitcodeHeader.Size] at &BitcodeHeader + Header.Offset

  void swapBytes() {
    sys::swapByteOrder(MinorVersion);
    sys::swapByteOrder(MajorVersion);
    sys::swapByteOrder(Offset);
    sys::swapByteOrder(Size);
  }
};

struct ProgramHeader {
  uint8_t MinorVersion : 4;
  uint8_t MajorVersion : 4;
  uint8_t Unused;
  uint16_t ShaderKind;
  uint32_t Size; // Size in uint32_t words including this header.
  BitcodeHeader Bitcode;

  void swapBytes() {
    sys::swapByteOrder(ShaderKind);
    sys::swapByteOrder(Size);
    Bitcode.swapBytes();
  }
};

static_assert(sizeof(ProgramHeader) == 24, "ProgramHeader Size incorrect!");

} // namespace dxbc
} // namespace llvm

#endif // LLVM_BINARYFORMAT_DXCONTAINER_H
