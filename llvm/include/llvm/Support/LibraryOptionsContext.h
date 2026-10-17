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

// The library options of one compilation context, such as an LLVMContext.
// Libraries declared with perContext read their options here.
class LLVM_ABI LibraryOptionsContext {
public:
  LibraryOptionsContext();

  // Replaces the library options with Other's by replaying its arguments.
  // Indexing libraries registered since the last parse is not thread-safe.
  void assign(const LibraryOptionsContext &Other);

  // Parses the option of L that Args[0] spells, keeping the other values.
  Error parse(LibraryOptions &L, ArrayRef<const char *> Args,
              unsigned &Consumed);

  // The arguments assigned so far.
  ArrayRef<const char *> args() const { return SavedArgs; }

  // T's values. T must have been registered when the context was created.
  template <typename T> const T &get() const {
    const void *Opts = Values.lookup(&T::Default);
    assert(Opts && "library options registered after the context's creation");
    return *static_cast<const T *>(Opts);
  }

  // C's values of T, or T::Global if C is null.
  template <typename T> static const T &get(const LibraryOptionsContext *C) {
    return C ? C->get<T>() : T::Global;
  }

private:
  void addDefaults();

  // Maps a library's defaults to its values, which are the defaults until an
  // option of the library is parsed.
  DenseMap<const void *, const void *> Values;
  SmallVector<const char *, 0> SavedArgs;
  BumpPtrAllocator Alloc;
};

} // namespace cl
} // namespace llvm

#endif // LLVM_SUPPORT_LIBRARYOPTIONSCONTEXT_H
