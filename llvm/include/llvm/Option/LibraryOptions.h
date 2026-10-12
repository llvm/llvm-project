//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for the options struct that -gen-opt-parser-defs generates from an
// OptionsStruct def. See "Declaring a Library's Options in TableGen" in
// llvm/docs/CommandLine.md.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPTION_LIBRARYOPTIONS_H
#define LLVM_OPTION_LIBRARYOPTIONS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BoolOrDefault.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
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

// A list member appends the comma-separated values of each occurrence. Its
// storage comes from Alloc, keeping the options struct trivially destructible.
template <typename T, typename F>
bool parseArgValue(StringRef S, ArrayRef<T> &List, BumpPtrAllocator &Alloc,
                   F ParseElem) {
  static_assert(std::is_trivially_copyable_v<T>);
  SmallVector<T, 4> Elems(List.begin(), List.end());
  for (StringRef Item : split(S, ',')) {
    T Elem{};
    if (!ParseElem(Item, Elem))
      return false;
    Elems.push_back(Elem);
  }
  T *Storage = Alloc.Allocate<T>(Elems.size());
  llvm::copy(Elems, Storage);
  List = ArrayRef(Storage, Elems.size());
  return true;
}

template <typename T>
bool parseArgValue(StringRef S, ArrayRef<T> &List, BumpPtrAllocator &Alloc) {
  return parseArgValue(S, List, Alloc, [](StringRef Item, T &Elem) {
    return parseArgValue(Item, Elem);
  });
}

/// An OptTable with a public constructor, shared by every options struct.
class LLVM_ABI LibraryOptTable : public OptTable {
public:
  explicit LibraryOptTable(const Tables &T) : OptTable(T) {}
  ~LibraryOptTable() override;
};

/// Connects an options struct to cl::ParseCommandLineOptions.
class LLVM_ABI LibraryOptionsParser final : public cl::LibraryOptions {
public:
  using ApplyFn = bool (*)(const Arg &, BumpPtrAllocator &);
  using TableFn = const OptTable &(*)();
  LibraryOptionsParser(TableFn Table, ApplyFn Apply, void (*Reset)())
      : Table(Table), Apply(Apply), Reset(Reset) {}

  void forEachOption(
      function_ref<void(StringRef, StringRef, StringRef)> Fn) const override;
  Error parse(ArrayRef<const char *> Args, unsigned &Consumed,
              BumpPtrAllocator &Alloc) override;
  void reset() override { Reset(); }

private:
  TableFn Table;
  ApplyFn Apply;
  void (*Reset)();
};

/// Registers T::Global with cl::ParseCommandLineOptions. The library owning T
/// defines one static instance in the file that includes the struct's
/// definitions.
template <typename T> class RegisterLibraryOptions {
  static_assert(std::is_trivially_destructible_v<T>,
                "an options struct must not need an exit-time destructor");
  LibraryOptionsParser Parser{T::optTable,
                              [](const Arg &A, BumpPtrAllocator &Alloc) {
                                return T::Global.apply(A, Alloc);
                              },
                              [] { T::Global = T(); }};

public:
  RegisterLibraryOptions() { cl::addLibraryOptions(Parser); }
};

} // namespace opt
} // namespace llvm

#endif // LLVM_OPTION_LIBRARYOPTIONS_H
