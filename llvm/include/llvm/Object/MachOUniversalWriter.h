//===- MachOUniversalWriter.h - MachO universal binary writer----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the Slice class and writeUniversalBinary function for writing a
// MachO universal binary file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHOUNIVERSALWRITER_H
#define LLVM_OBJECT_MACHOUNIVERSALWRITER_H

#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/MachO.h"

namespace llvm {
class LLVMContext;

namespace object {
class IRObjectFile;

class Slice {
  const Binary *B;
  uint32_t CPUType;
  uint32_t CPUSubType;
  std::string ArchName;

  // P2Alignment field stores slice alignment values from universal
  // binaries. This is also needed to order the slices so the total
  // file size can be calculated before creating the output buffer.
  uint32_t P2Alignment;

  Slice(const IRObjectFile &IRO, uint32_t CPUType, uint32_t CPUSubType,
        std::string ArchName, uint32_t Align);

public:
  explicit Slice(const MachOObjectFile &O);

  Slice(const MachOObjectFile &O, uint32_t Align);

  /// This constructor takes prespecified \param CPUType, \param CPUSubType,
  /// \param ArchName, \param Align instead of inferring them from the archive
  /// memebers.
  Slice(const Archive &A, uint32_t CPUType, uint32_t CPUSubType,
        std::string ArchName, uint32_t Align);

  static Expected<Slice> create(const Archive &A,
                                LLVMContext *LLVMCtx = nullptr);

  static Expected<Slice> create(const IRObjectFile &IRO, uint32_t Align);

  void setP2Alignment(uint32_t Align) { P2Alignment = Align; }

  const Binary *getBinary() const { return B; }

  uint32_t getCPUType() const { return CPUType; }

  uint32_t getCPUSubType() const { return CPUSubType; }

  uint32_t getP2Alignment() const { return P2Alignment; }

  uint64_t getCPUID() const {
    return static_cast<uint64_t>(CPUType) << 32 | CPUSubType;
  }

  std::string getArchString() const {
    if (!ArchName.empty())
      return ArchName;
    return ("unknown(" + Twine(CPUType) + "," +
            Twine(CPUSubType & ~MachO::CPU_SUBTYPE_MASK) + ")")
        .str();
  }

  friend bool operator<(const Slice &Lhs, const Slice &Rhs) {
    if (Lhs.CPUType == Rhs.CPUType)
      return Lhs.CPUSubType < Rhs.CPUSubType;
    // force arm64-family to follow after all other slices for
    // compatibility with cctools lipo
    if (Lhs.CPUType == MachO::CPU_TYPE_ARM64)
      return false;
    if (Rhs.CPUType == MachO::CPU_TYPE_ARM64)
      return true;
    // Sort by alignment to minimize file size
    return Lhs.P2Alignment < Rhs.P2Alignment;
  }
};

Error writeUniversalBinary(ArrayRef<Slice> Slices, StringRef OutputFileName);

Expected<std::unique_ptr<MemoryBuffer>>
writeUniversalBinaryToBuffer(ArrayRef<Slice> Slices);

} // end namespace object

} // end namespace llvm

#endif // LLVM_OBJECT_MACHOUNIVERSALWRITER_H
