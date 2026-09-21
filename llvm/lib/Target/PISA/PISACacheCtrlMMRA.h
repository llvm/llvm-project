//===-- PISACacheCtrlMMRA.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISACACHECTRLMMRA_H
#define LLVM_LIB_TARGET_PISA_PISACACHECTRLMMRA_H

#include <optional>

namespace llvm {
class Instruction;

namespace PISA {

// Reads the cache-ctrl integer from the "pisa.cache.ctrl" MMRA tag on I.
// Returns std::nullopt if the tag is absent or malformed. If more than
// one distinct cache-ctrl value is attached (e.g. a passthrough merge of
// two memory ops with different cache hints by an upstream pass such as
// the LoadStoreVectorizer), emits a warning through the LLVMContext
// listing the conflicting values and keeps one of them so the compiler
// can still produce valid output.
std::optional<unsigned> getCacheCtrlFromMMRA(const Instruction &I);

// Sets the "pisa.cache.ctrl" MMRA tag on I to Value. Preserves all
// other MMRA tags already on I, replacing any prior cache-ctrl tag.
void setCacheCtrlMMRA(Instruction &I, unsigned Value);

// Copies the "pisa.cache.ctrl" MMRA tag from From to To if present,
// preserving all other MMRA tags already on To.
void copyCacheCtrlMMRA(const Instruction &From, Instruction &To);

} // namespace PISA
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISACACHECTRLMMRA_H
