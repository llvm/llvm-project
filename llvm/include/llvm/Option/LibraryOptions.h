//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Connects the options struct that -gen-opt-parser-defs generates from an
// OptionsStruct def to cl::parseCommandLineOptions. See "Declaring a Library's
// Options in TableGen" in llvm/docs/CommandLine.md.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPTION_LIBRARYOPTIONS_H
#define LLVM_OPTION_LIBRARYOPTIONS_H

#include "llvm/Option/OptionsStruct.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include <utility>

namespace llvm {
namespace opt {

/// Connects options struct T to cl::parseCommandLineOptions and
/// cl::LibraryOptionsContext.
class LLVM_ABI LibraryOptionsParser final : public cl::LibraryOptions {
public:
  template <typename T>
  constexpr explicit LibraryOptionsParser(std::in_place_type_t<T>)
      : Table(T::optTable), Default(&T::Default),
        Apply([](void *Opts, const Arg &A, BumpPtrAllocator &Alloc) {
          return static_cast<T *>(Opts)->apply(A, Alloc);
        }),
        Copy([](BumpPtrAllocator &Alloc) -> void * {
          return new (Alloc) T(T::Default);
        }) {}

  void forEachOption(
      function_ref<void(StringRef, StringRef, StringRef)> Fn) const override;
  Error parse(void *Opts, ArrayRef<const char *> Args, unsigned &Consumed,
              BumpPtrAllocator &Alloc) const override;
  const void *getDefault() const override { return Default; }
  void *copyDefault(BumpPtrAllocator &Alloc) const override {
    return Copy(Alloc);
  }

private:
  const OptionsStructTable &(*Table)();
  const void *Default;
  bool (*Apply)(void *Opts, const Arg &A, BumpPtrAllocator &Alloc);
  void *(*Copy)(BumpPtrAllocator &Alloc);
};

/// Registers T with cl::parseCommandLineOptions. The library owning T
/// defines one static instance in the file that includes the struct's
/// definitions.
template <typename T> class RegisterLibraryOptions {
  static_assert(std::is_trivially_destructible_v<T>,
                "an options struct must not need an exit-time destructor");
  LibraryOptionsParser Parser{std::in_place_type<T>};

public:
  RegisterLibraryOptions() { cl::addLibraryOptions(Parser); }
};

} // namespace opt
} // namespace llvm

/// Defines the file-scope registration of options struct T for a library that
/// has no initialization function to register it from, exempting this static
/// constructor from -Wglobal-constructors.
// clang-format off
#ifdef __clang__
#define LLVM_REGISTER_LIBRARY_OPTIONS(T)                                       \
  _Pragma("clang diagnostic push")                                             \
  _Pragma("clang diagnostic ignored \"-Wglobal-constructors\"")                \
  static llvm::opt::RegisterLibraryOptions<T> Registration;                    \
  _Pragma("clang diagnostic pop")
#else
#define LLVM_REGISTER_LIBRARY_OPTIONS(T)                                       \
  static llvm::opt::RegisterLibraryOptions<T> Registration;
#endif
// clang-format on

#endif // LLVM_OPTION_LIBRARYOPTIONS_H
