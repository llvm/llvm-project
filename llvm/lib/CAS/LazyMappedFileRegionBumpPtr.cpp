//===- LazyMappedFileRegionBumpPtr.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ScopeExit.h"
#include "llvm/CAS/LazyMappedFileRegionBumpPtr.h"

using namespace llvm;
using namespace llvm::cas;

void LazyMappedFileRegionBumpPtr::initialize(int64_t BumpPtrOffset) {
  int64_t BumpPtrEndOffset = BumpPtrOffset + sizeof(decltype(*BumpPtr));
  assert(BumpPtrEndOffset <= int64_t(LMFR.size()) &&
         "Expected end offset to be pre-allocated");
  assert(isAligned(Align::Of<decltype(*BumpPtr)>(), BumpPtrOffset) &&
         "Expected end offset to be aligned");
  BumpPtr = reinterpret_cast<decltype(BumpPtr)>(data() + BumpPtrOffset);

  int64_t ExistingValue = 0;
  if (!BumpPtr->compare_exchange_strong(ExistingValue, BumpPtrEndOffset))
    assert(ExistingValue >= BumpPtrEndOffset &&
           "Expected 0, or past the end of the BumpPtr itself");
}

int64_t LazyMappedFileRegionBumpPtr::allocateOffset(uint64_t AllocSize) {
  AllocSize = alignTo(AllocSize, getAlign());
  int64_t OldEnd = BumpPtr->fetch_add(AllocSize);
  int64_t NewEnd = OldEnd + AllocSize;
  if (Error E = LMFR.extendSize(OldEnd + AllocSize)) {
    // Try to return the allocation.
    (void)BumpPtr->compare_exchange_strong(OldEnd, NewEnd);
    report_fatal_error(std::move(E));
  }
  return OldEnd;
}
