//===-- OMP.h - Core OpenMP definitions and declarations ---------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the core set of OpenMP definitions and declarations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OPENMP_OMP_H
#define LLVM_FRONTEND_OPENMP_OMP_H

#include "llvm/Frontend/OpenMP/OMP.h.inc"
#include "llvm/Support/Compiler.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm::omp {
template <typename Enum, size_t Size> struct EnumSet;

namespace detail {
template <size_t Size>
static constexpr inline size_t findFirstSet(size_t Begin, size_t End,
                                            const llvm::Bitset<Size> &Set) {
  unsigned BeginWord = Begin / 64;
  unsigned EndWord = (End + 63) / 64;

  for (unsigned I = BeginWord; I < EndWord; ++I) {
    uint64_t Word = Set.getWord64(I);
    if (I == BeginWord && Begin % 64 != 0) {
      Word &= ~uint64_t() << (Begin % 64);
    }
    auto Count = static_cast<unsigned>(llvm::countr_zero_constexpr(Word));
    if (Count < 64) {
      unsigned Idx = I * 64 + Count;
      if (Idx >= Begin && Idx < End)
        return Idx;
    }
  }
  return Size;
}

template <typename Enum, size_t Size> struct EnumSetIterator {
  using value_type = Enum;
  static constexpr size_t enum_size = Size;

  constexpr EnumSetIterator(const EnumSet<Enum, Size> &Set, size_t At)
      : Set(Set), At(At) {}

  constexpr Enum operator*() const;
  constexpr auto &operator++();

  constexpr bool operator==(const EnumSetIterator<Enum, Size> &Other) const {
    return &Set == &Other.Set && At == Other.At;
  }
  constexpr bool operator!=(const EnumSetIterator<Enum, Size> &Other) const {
    return !operator==(Other);
  }

private:
  const EnumSet<Enum, Size> &Set;
  size_t At;
};
} // namespace detail

template <typename Enum, size_t Size>
struct EnumSet : public llvm::Bitset<Size> {
  using value_type = Enum;
  using Base = llvm::Bitset<Size>;
  using Base::Base;
  using iterator = detail::EnumSetIterator<Enum, Size>;

  constexpr EnumSet(Base &&B) : Base(std::move(B)) {}
  constexpr EnumSet(std::initializer_list<value_type> Init) {
    for (value_type E : Init) {
      auto Value = static_cast<unsigned>(E);
      assert(Value < Base::size() && "Invalid enumeration value");
      Base::set(Value);
    }
  }

  constexpr bool empty() const { return Base::none(); }
  constexpr size_t size() const { return Base::count(); }
  constexpr size_t max_size() const { return Size; }

  constexpr bool test(Enum E) const {
    return Base::test(static_cast<unsigned>(E));
  }
  constexpr bool operator[](Enum E) const {
    return Base::operator[](static_cast<unsigned>(E));
  }
  constexpr EnumSet &flip(Enum E) {
    Base::flip(static_cast<unsigned>(E));
    return *this;
  }
  constexpr EnumSet &reset(Enum E) {
    Base::reset(static_cast<unsigned>(E));
    return *this;
  }
  constexpr EnumSet &set(Enum E) {
    Base::set(static_cast<unsigned>(E));
    return *this;
  }

  constexpr EnumSet &operator|=(const EnumSet &S) {
    Base::operator|=(S);
    return *this;
  }
  constexpr EnumSet &operator&=(const EnumSet &S) {
    Base::operator&=(S);
    return *this;
  }
  constexpr EnumSet operator|(const EnumSet &S) const {
    EnumSet T{*this};
    return T |= S;
  }
  constexpr EnumSet operator&(const EnumSet &S) const {
    EnumSet T{*this};
    return T &= S;
  }

  constexpr iterator begin() const {
    return iterator(*this, detail::findFirstSet<Size>(0, Size, *this));
  }
  constexpr iterator end() const { return iterator(*this, Size); }
};

namespace detail {
template <typename Enum, size_t Size>
constexpr Enum EnumSetIterator<Enum, Size>::operator*() const {
  assert(Set.Base::test(At));
  return static_cast<Enum>(At);
}

template <typename Enum, size_t Size>
constexpr auto &EnumSetIterator<Enum, Size>::operator++() {
  At = findFirstSet<Size>(At + 1, Size, Set);
  return *this;
}
} // namespace detail

using ClauseSet = EnumSet<llvm::omp::Clause, llvm::omp::Clause_enumSize>;
using DirectiveSet =
    EnumSet<llvm::omp::Directive, llvm::omp::Directive_enumSize>;

LLVM_ABI ArrayRef<Directive> getLeafConstructs(Directive D);
LLVM_ABI ArrayRef<Directive> getLeafConstructsOrSelf(Directive D);

LLVM_ABI ArrayRef<Directive>
getLeafOrCompositeConstructs(Directive D, SmallVectorImpl<Directive> &Output);

LLVM_ABI Directive getCompoundConstruct(ArrayRef<Directive> Parts);

LLVM_ABI bool isLeafConstruct(Directive D);
LLVM_ABI bool isCompositeConstruct(Directive D);
LLVM_ABI bool isCombinedConstruct(Directive D);

static constexpr inline auto clauses() {
  return llvm::enum_seq_inclusive(Clause::First_, Clause::Last_);
}

static constexpr inline auto directives() {
  return llvm::enum_seq_inclusive(Directive::First_, Directive::Last_);
}

/// Can clause C have an iterator-modifier.
static constexpr inline bool canHaveIterator(Clause C) {
  // [5.2:67:5]
  switch (C) {
  case OMPC_affinity:
  case OMPC_depend:
  case OMPC_from:
  case OMPC_map:
  case OMPC_to:
    return true;
  default:
    return false;
  }
}

// Can clause C create a private copy of a variable.
static constexpr inline bool isPrivatizingClause(Clause C, unsigned Version) {
  switch (C) {
  case OMPC_firstprivate:
  case OMPC_in_reduction:
  case OMPC_lastprivate:
  case OMPC_linear:
  case OMPC_private:
  case OMPC_reduction:
  case OMPC_task_reduction:
    return true;
  case OMPC_detach:
  case OMPC_induction:
  case OMPC_is_device_ptr:
  case OMPC_use_device_ptr:
    return Version >= 60;
  default:
    return false;
  }
}

static constexpr inline bool isDataSharingAttributeClause(Clause C,
                                                          unsigned Version) {
  // The "Version" parameter is in case the result is version-depenent
  // in the future.
  (void)Version;
  switch (C) {
  case OMPC_detach:
  case OMPC_firstprivate:
  case OMPC_has_device_addr:
  case OMPC_induction:
  case OMPC_in_reduction:
  case OMPC_is_device_ptr:
  case OMPC_lastprivate:
  case OMPC_linear:
  case OMPC_private:
  case OMPC_reduction:
  case OMPC_shared:
  case OMPC_task_reduction:
  case OMPC_use_device_addr:
  case OMPC_use_device_ptr:
  case OMPC_uses_allocators:
    return true;
  default:
    return false;
  }
}

static constexpr inline bool isEndClause(Clause C) {
  switch (C) {
  case OMPC_copyprivate:
  case OMPC_nowait:
    return true;
  default:
    return false;
  }
}

static constexpr unsigned FallbackVersion = 52;
LLVM_ABI ArrayRef<unsigned> getOpenMPVersions();

/// Can directive D, under some circumstances, create a private copy
/// of a variable in given OpenMP version?
LLVM_ABI bool isPrivatizingConstruct(Directive D, unsigned Version);

LLVM_ABI ArrayRef<StringRef> getReservedLocatorNames();

/// Create a nicer version of a function name for humans to look at.
LLVM_ABI std::string prettifyFunctionName(StringRef FunctionName);

/// Deconstruct an OpenMP kernel name into the parent function name and the line
/// number.
LLVM_ABI std::string deconstructOpenMPKernelName(StringRef KernelName,
                                                 unsigned &LineNo);

} // namespace llvm::omp

#endif // LLVM_FRONTEND_OPENMP_OMP_H
