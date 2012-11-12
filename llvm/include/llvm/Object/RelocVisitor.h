//===-- RelocVisitor.h - Visitor for object file relocations -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a wrapper around all the different types of relocations
// in different file formats, such that a client can handle them in a unified
// manner by only implementing a minimal number of functions.
//
//===----------------------------------------------------------------------===//

#ifndef _LLVM_OBJECT_RELOCVISITOR
#define _LLVM_OBJECT_RELOCVISITOR

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/ELF.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace object {

struct RelocToApply {
  // The computed value after applying the relevant relocations.
  int64_t Value;

  // The width of the value; how many bytes to touch when applying the
  // relocation.
  char Width;
  RelocToApply(const RelocToApply &In) : Value(In.Value), Width(In.Width) {}
  RelocToApply(int64_t Value, char Width) : Value(Value), Width(Width) {}
  RelocToApply() : Value(0), Width(0) {}
};

/// @brief Base class for object file relocation visitors.
class RelocVisitor {
public:
  explicit RelocVisitor(llvm::StringRef FileFormat)
    : FileFormat(FileFormat), HasError(false) {}

  // TODO: Should handle multiple applied relocations via either passing in the
  // previously computed value or just count paired relocations as a single
  // visit.
  RelocToApply visit(uint32_t RelocType, RelocationRef R, uint64_t SecAddr = 0,
                     uint64_t Value = 0) {
    if (FileFormat == "ELF64-x86-64") {
      switch (RelocType) {
        case llvm::ELF::R_X86_64_NONE:
          return visitELF_X86_64_NONE(R);
        case llvm::ELF::R_X86_64_64:
          return visitELF_X86_64_64(R, Value);
        case llvm::ELF::R_X86_64_PC32:
          return visitELF_X86_64_PC32(R, Value, SecAddr);
        case llvm::ELF::R_X86_64_32:
          return visitELF_X86_64_32(R, Value);
        case llvm::ELF::R_X86_64_32S:
          return visitELF_X86_64_32S(R, Value);
        default:
          HasError = true;
          return RelocToApply();
      }
    }
    return RelocToApply();
  }

  bool error() { return HasError; }

private:
  llvm::StringRef FileFormat;
  bool HasError;

  /// Operations

  // Width is the width in bytes of the extend.
  RelocToApply zeroExtend(RelocToApply r, char Width) {
    if (Width == r.Width)
      return r;
    r.Value &= (1 << ((Width * 8))) - 1;
    return r;
  }
  RelocToApply signExtend(RelocToApply r, char Width) {
    if (Width == r.Width)
      return r;
    bool SignBit = r.Value & (1 << ((Width * 8) - 1));
    if (SignBit) {
      r.Value |= ~((1 << (Width * 8)) - 1);
    } else {
      r.Value &= (1 << (Width * 8)) - 1;
    }
    return r;
  }

  /// X86-64 ELF
  RelocToApply visitELF_X86_64_NONE(RelocationRef R) {
    return RelocToApply(0, 0);
  }
  RelocToApply visitELF_X86_64_64(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    return RelocToApply(Value + Addend, 8);
  }
  RelocToApply visitELF_X86_64_PC32(RelocationRef R, uint64_t Value,
                                    uint64_t SecAddr) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    uint64_t Address;
    R.getAddress(Address);
    return RelocToApply(Value + Addend - Address, 4);
  }
  RelocToApply visitELF_X86_64_32(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    uint32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }
  RelocToApply visitELF_X86_64_32S(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    int32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }
};

}
}
#endif
