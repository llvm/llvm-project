//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_LIBRARYOPTIONSCONTEXT_H
#define LLVM_SUPPORT_LIBRARYOPTIONSCONTEXT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Error;

namespace cl {

class LibraryOptions;

// Values of library options that override their global() in one compilation
// context, such as an LLVMContext.
class LLVM_ABI LibraryOptionsContext {
public:
  // Sets the library options that Args spell, e.g. {"-x=v"}. A library's values
  // start as a copy of its global(), and later calls add to them. Fails at the
  // first argument that is not a library option or has an invalid value.
  //
  // Strings copied from global() die with ResetAllOptionOccurrences. The first
  // call (or ParseCommandLineOptions) indexes the registry, which is not
  // thread-safe.
  Error set(ArrayRef<const char *> Args);

  // Parses the option of L that Args[0] spells, as set() does.
  Error parse(LibraryOptions &L, ArrayRef<const char *> Args,
              unsigned &Consumed);

  // The arguments set so far, for replaying into another context.
  ArrayRef<const char *> args() const { return Args; }

  // T's values, or T::Global if there are none.
  template <typename T> const T &get() const {
    if (const void *Opts = Values.lookup(&T::Global))
      return *static_cast<const T *>(Opts);
    // TODO: Return the defaults once cl::ParseCommandLineOptions callers have
    // migrated to parseCommandLineOptions and no longer set T::Global.
    return T::Global;
  }

private:
  DenseMap<const void *, void *> Values;
  SmallVector<const char *, 0> Args;
  BumpPtrAllocator Alloc;
};

} // namespace cl
} // namespace llvm

#endif // LLVM_SUPPORT_LIBRARYOPTIONSCONTEXT_H
