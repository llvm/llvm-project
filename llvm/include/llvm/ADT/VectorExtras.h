//===-- llvm/ADT/VectorExtras.h - Helpers for std::vector -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions which are useful for working with the
// std::vector class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_VECTOREXTRAS_H
#define LLVM_ADT_VECTOREXTRAS_H

#include <cstdarg>
#include <vector>

namespace llvm {

/// make_vector - Helper function which is useful for building temporary vectors
/// to pass into type construction of CallInst ctors.  This turns a null
/// terminated list of pointers (or other value types) into a real live vector.
///
template<typename T>
inline std::vector<T> make_vector(T A, ...) {
  va_list Args;
  va_start(Args, A);
  std::vector<T> Result;
  Result.push_back(A);
  while (T Val = va_arg(Args, T))
    Result.push_back(Val);
  va_end(Args);
  return Result;
}

} // End llvm namespace

#endif
