//===-- FileDescriptor.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a utility functions for working with file descriptors
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILEDESCRIPTOR_H
#define LLVM_SUPPORT_FILEDESCRIPTOR_H

#include "llvm/Support/Error.h"
#include <chrono>

namespace llvm {
// Helper function to get the value from either std::atomic<int> or int
template <typename T> int getFD(T &FD) {
  if constexpr (std::is_same_v<T, std::atomic<int>>) {
    return FD.load();
  } else {
    return FD;
  }
}

template <typename T>
llvm::Error manageTimeout(std::chrono::milliseconds Timeout, T &FD, int PipeFD);
} // namespace llvm
#endif
