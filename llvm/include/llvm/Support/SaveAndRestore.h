//===-- SaveAndRestore.h - Utility  -------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utility classes that use RAII to save and restore
/// values.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SAVEANDRESTORE_H
#define LLVM_SUPPORT_SAVEANDRESTORE_H

namespace llvm {

/// A utility class that uses RAII to save and restore the value of a variable.
template <typename T> struct SaveAndRestore {
  SaveAndRestore(T &X) : X(X), OldValue(X) {}
  SaveAndRestore(T &X, const T &NewValue) : X(X), OldValue(X) {
    X = NewValue;
  }
  ~SaveAndRestore() { X = OldValue; }
  T get() { return OldValue; }

private:
  T &X;
  T OldValue;
};

} // namespace llvm

#endif
