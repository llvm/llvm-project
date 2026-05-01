#ifndef LLVM_DWP_DWPSTRINGPOOL_H
#define LLVM_DWP_DWPSTRINGPOOL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

namespace llvm {
class DWPStringPool {
  SmallVectorImpl<char> &Buffer;
  // Use StringRef keys instead of const char* to avoid redundant strlen
  // on every hash computation and strcmp on every probe comparison.
  DenseMap<StringRef, uint64_t> Pool;
  uint64_t Offset = 0;

public:
  DWPStringPool(SmallVectorImpl<char> &Buffer) : Buffer(Buffer) {}

  uint64_t getOffset(const char *Str, unsigned Length) {
    assert(strlen(Str) + 1 == Length && "Ensure length hint is correct");

    StringRef Key(Str, Length - 1);
    auto Pair = Pool.insert(std::make_pair(Key, Offset));
    if (Pair.second) {
      Buffer.insert(Buffer.end(), Str, Str + Length);
      Offset += Length;
    }

    return Pair.first->second;
  }
};
} // namespace llvm

#endif // LLVM_DWP_DWPSTRINGPOOL_H
