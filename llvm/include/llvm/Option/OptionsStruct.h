//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for the options struct that -gen-opt-parser-defs generates from an
// OptionsStruct def, and for a tool that parses its command line into one.
// See "Declaring a Library's Options in TableGen" in llvm/docs/CommandLine.md.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPTION_OPTIONSSTRUCT_H
#define LLVM_OPTION_OPTIONSSTRUCT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BoolOrDefault.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"
#include <optional>
#include <type_traits>

namespace llvm {
namespace opt {

// Each accepts the spellings cl::opt accepts for the type.
inline bool parseArgValue(StringRef S, bool &V) {
  if (S == "true" || S == "1")
    V = true;
  else if (S == "false" || S == "0")
    V = false;
  else
    return false;
  return true;
}

inline bool parseArgValue(StringRef S, BoolOrDefault &V) {
  bool B;
  if (!parseArgValue(S, B))
    return false;
  V = B ? BoolOrDefault::True : BoolOrDefault::False;
  return true;
}

inline bool parseArgValue(StringRef S, StringRef &V) {
  V = S;
  return true;
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, bool>
parseArgValue(StringRef S, T &V) {
  if constexpr (std::is_floating_point_v<T>)
    return to_float(S, V);
  else
    return to_integer(S, V);
}

// A std::optional member is set only when its option is given.
template <typename T> bool parseArgValue(StringRef S, std::optional<T> &V) {
  T X{};
  if (!parseArgValue(S, X))
    return false;
  V = X;
  return true;
}

// The generated apply() passes Alloc, which only a list member uses.
template <typename T>
bool parseArgValue(StringRef S, T &V, BumpPtrAllocator &) {
  return parseArgValue(S, V);
}

// An enum member is set by Parse, the generated value-to-enumerator mapping.
template <typename T, typename F>
bool parseArgValue(StringRef S, T &V, BumpPtrAllocator &, F Parse) {
  return Parse(S, V);
}

// Appends Elems to a list member, whose storage comes from Alloc, keeping the
// options struct trivially destructible.
template <typename T>
void appendList(ArrayRef<T> &List, ArrayRef<T> Elems, BumpPtrAllocator &Alloc) {
  static_assert(std::is_trivially_copyable_v<T>);
  T *Storage = Alloc.Allocate<T>(List.size() + Elems.size());
  llvm::copy(Elems, llvm::copy(List, Storage));
  List = ArrayRef(Storage, List.size() + Elems.size());
}

// A ListField member appends the comma-separated values of each occurrence.
template <typename T, typename F>
bool parseArgValue(StringRef S, ArrayRef<T> &List, BumpPtrAllocator &Alloc,
                   F ParseElem) {
  SmallVector<T, 4> Elems;
  for (StringRef Item : split(S, ',')) {
    T Elem{};
    if (!ParseElem(Item, Elem))
      return false;
    Elems.push_back(Elem);
  }
  appendList(List, ArrayRef<T>(Elems), Alloc);
  return true;
}

template <typename T>
bool parseArgValue(StringRef S, ArrayRef<T> &List, BumpPtrAllocator &Alloc) {
  return parseArgValue(S, List, Alloc, [](StringRef Item, T &Elem) {
    return parseArgValue(Item, Elem);
  });
}

// A RepeatedField member appends the whole value of each occurrence.
template <typename T>
bool appendArgValue(StringRef S, ArrayRef<T> &List, BumpPtrAllocator &Alloc) {
  T Elem{};
  if (!parseArgValue(S, Elem))
    return false;
  appendList(List, ArrayRef(Elem), Alloc);
  return true;
}

/// An OptTable with a public constructor, shared by every options struct.
class LLVM_ABI OptionsStructTable : public OptTable {
public:
  explicit OptionsStructTable(const Tables &T) : OptTable(T) {
    setDashDashParsing(true);
  }
  ~OptionsStructTable() override;
};

/// Parses a tool's command line, expanding response files, with the options
/// table \p T. \p Apply sets the member an option names. Positional
/// arguments, and every argument after "--", are appended to \p Inputs.
namespace detail {
LLVM_ABI Error parseToolArgsImpl(const OptTable &T, int Argc,
                                 const char *const *Argv, StringSaver &Saver,
                                 function_ref<bool(const Arg &)> Apply,
                                 SmallVectorImpl<StringRef> &Inputs);
} // namespace detail

/// Parses a tool's command line into options struct \p Opts, whose string
/// members refer to \p Argv or \p Alloc.
template <typename T>
Error parseToolArgs(T &Opts, int Argc, const char *const *Argv,
                    BumpPtrAllocator &Alloc,
                    SmallVectorImpl<StringRef> &Inputs) {
  StringSaver Saver(Alloc);
  return detail::parseToolArgsImpl(
      T::optTable(), Argc, Argv, Saver,
      [&](const Arg &A) { return Opts.apply(A, Alloc); }, Inputs);
}

} // namespace opt
} // namespace llvm

#endif // LLVM_OPTION_OPTIONSSTRUCT_H
