//===- DebugLoc.h - Debug Location Information ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of light weight data structures used
// to describe and track debug location information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGLOC_H
#define LLVM_IR_DEBUGLOC_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>


namespace llvm {
  
  class LLVMContext;
  class raw_ostream;
  class DILocalScope;
  class DISubprogram;
  class MDNode;
  class Metadata;
  class LLVMContext;

  /// A debug info location.
  ///
  /// This class is used to index the location of a debug info location in a
  /// particular \a DISubprogram.
  class DebugLoc {
  public:
    uint32_t SrcLocIndex;
    uint32_t LocScopeIndex;

  public:
    DebugLoc() : SrcLocIndex(0xFFFFFFFF), LocScopeIndex(0xFFFFFFFF) {
      static_assert(sizeof(DebugLoc) == sizeof(uint64_t), "Should fit into 64 bits.");
    };
    DebugLoc(std::nullptr_t) : DebugLoc() {}
    DebugLoc(uint32_t SrcLoc, uint32_t LocScope) :
      SrcLocIndex(SrcLoc), LocScopeIndex(LocScope) {}

    DebugLoc(uint64_t RawInt) :
      SrcLocIndex(static_cast<uint32_t>(RawInt >> 32)),
      LocScopeIndex(static_cast<uint32_t>(RawInt)) {}
    
    DebugLoc(const DebugLoc &) = default;
    DebugLoc(DebugLoc &&) = default;
    DebugLoc &operator=(const DebugLoc &) = default;
    DebugLoc &operator=(DebugLoc &&) = default;

    bool isUsingTransientDILocation() {
      return LocScopeIndex == 0xFFFFFFFF && SrcLocIndex != 0xFFFFFFFF;
    }
    MDNode *getTransientDILocation(DISubprogram *SP);

    inline uint64_t getAsRawInteger() const {
      return (static_cast<uint64_t>(SrcLocIndex) << 32) | LocScopeIndex;
    }

    bool hasLineZero() const {
      return SrcLocIndex != 0xFFFFFFFF && SrcLocIndex > 0;
    }
    DebugLoc getWithLineZero() const {
      return DebugLoc(0, LocScopeIndex);
    }

    DebugLoc &get() {
      return *this;
    }

    uint32_t getSrcLocIndex() const { return SrcLocIndex; };
    uint32_t getLocScopeIndex() const { return LocScopeIndex; };

    Metadata *toMetadata(LLVMContext &C) const;
    static std::optional<DebugLoc> fromMetadata(Metadata *MD);

    /// Check whether this is an empty location.
    explicit operator bool() const { return SrcLocIndex != 0xFFFFFFFF || LocScopeIndex != 0xFFFFFFFF; }
    bool isValid() const { return SrcLocIndex != 0xFFFFFFFF || LocScopeIndex != 0xFFFFFFFF; }

    /// Check whether this has a trivial destructor.
    bool hasTrivialDestructor() const { return true; }

    unsigned getLine(DISubprogram *SP) const;
    unsigned getCol(DISubprogram *SP) const;
    DILocalScope *getScope(DISubprogram *SP) const;
    DebugLoc getInlinedAt(DISubprogram *SP) const;
    uint64_t getAtomGroup(DISubprogram *SP) const;
    uint8_t getAtomRank(DISubprogram *SP) const;

    /// Get the fully inlined-at scope for a DebugLoc.
    ///
    /// Gets the inlined-at scope for a DebugLoc.
    DILocalScope *getInlinedAtScope(DISubprogram *SP) const;

    /// Check if the DebugLoc corresponds to an implicit code.
    bool isImplicitCode(DISubprogram *SP) const;

    bool operator==(const DebugLoc &DL) const {
      return getAsRawInteger() == DL.getAsRawInteger();
    }
    bool operator!=(const DebugLoc &DL) const {
      return getAsRawInteger() != DL.getAsRawInteger();
    }

    void dump() const;

    /// prints source location /path/to/file.exe:line:col @[inlined at]
    void print(raw_ostream &OS) const;
    void print(raw_ostream &OS, DISubprogram *SP) const;

    bool operator<(const DebugLoc &DL) const {
      return std::tie(SrcLocIndex, LocScopeIndex) < std::tie(DL.SrcLocIndex, DL.LocScopeIndex);
    }
  };

  inline hash_code hash_value(DebugLoc DL) {
    return llvm::hash_value(DL.getAsRawInteger());
  }

  template <>
  struct DenseMapInfo<DebugLoc> {
  
    static DebugLoc getEmptyKey() {
      return DebugLoc(0xfffffffe, 0xfffffffe);
    }
  
    static DebugLoc getTombstoneKey() {
      return DebugLoc(0xfffffffd, 0xfffffffd);
    }
  
    static unsigned getHashValue(DebugLoc V) {
      return hash_value(V.getAsRawInteger());
    }
  
    static bool isEqual(const DebugLoc &LHS, const DebugLoc &RHS) { return LHS == RHS; }
  };


  inline raw_ostream &operator<<(raw_ostream &OS, const DebugLoc &DL) {
    DL.print(OS);
    return OS;
  }

  /// Wrapper class for DILocRef that does not depend on DebugInfoMetadata.h,
  /// and can be implicitly converted to DILocRef when DebugInfoMetadata.h is
  /// included.
  class DILocRefWrapper {
  public:
    DILocRefWrapper() : SP(nullptr), Index() {}
    DILocRefWrapper(DISubprogram *SP, DebugLoc Index) : SP(SP), Index(Index) {}
    DISubprogram *SP;
    DebugLoc Index;
    operator bool() const { return (bool)Index; }
    operator DebugLoc() { return Index; }
  };

} // end namespace llvm

#endif // LLVM_IR_DEBUGLOC_H
