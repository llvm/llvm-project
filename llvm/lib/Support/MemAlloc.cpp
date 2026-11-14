//===- MemAlloc.cpp - Memory allocation functions -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemAlloc.h"
#include <new>

// These are out of line to have __cpp_aligned_new not affect ABI.

LLVM_ATTRIBUTE_RETURNS_NONNULL LLVM_ATTRIBUTE_RETURNS_NOALIAS void *
llvm::allocate_buffer(size_t Size, size_t Alignment) {
  void *Result;
#ifdef __cpp_aligned_new
  // Aligned allocation is slower, so only use it when it's needed.
  if (Alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    Result = ::operator new(Size, std::align_val_t(Alignment), std::nothrow);
  else
#endif
    Result = ::operator new(Size, std::nothrow);
  if (Result == nullptr) {
    report_bad_alloc_error("Buffer allocation failed");
  }
  return Result;
}

void llvm::deallocate_buffer(void *Ptr, size_t Size, size_t Alignment) {
#ifdef __cpp_aligned_new
  if (Alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    ::operator delete(Ptr,
#ifdef __cpp_sized_deallocation
                      Size,
#endif
                      std::align_val_t(Alignment));
    return;
  }
#endif
  ::operator delete(Ptr
#ifdef __cpp_sized_deallocation
                    ,
                    Size
#endif
  );
}
