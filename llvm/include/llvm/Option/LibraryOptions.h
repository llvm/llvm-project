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

#include "llvm/ADT/StringExtras.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
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

/// An OptTable with a public constructor, shared by every options struct.
class LLVM_ABI LibraryOptTable : public OptTable {
public:
  explicit LibraryOptTable(const Tables &T) : OptTable(T) {}
  ~LibraryOptTable() override;
};

/// Connects an options struct to cl::ParseCommandLineOptions.
class LLVM_ABI LibraryOptionsParser final : public cl::LibraryOptions {
public:
  using ApplyFn = bool (*)(const Arg &);
  using TableFn = const OptTable &(*)();
  LibraryOptionsParser(TableFn Table, ApplyFn Apply, void (*Reset)())
      : Table(Table), Apply(Apply), Reset(Reset) {}

  void forEachOption(function_ref<void(StringRef, StringRef, StringRef, bool)>
                         Fn) const override;
  Error parse(ArrayRef<const char *> Args, unsigned &Consumed) override;
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
  LibraryOptionsParser Parser{T::optTable,
                              [](const Arg &A) { return T::Global.apply(A); },
                              [] { T::Global = T(); }};

public:
  RegisterLibraryOptions() { cl::addLibraryOptions(Parser); }
};

} // namespace opt
} // namespace llvm

#endif // LLVM_OPTION_LIBRARYOPTIONS_H
