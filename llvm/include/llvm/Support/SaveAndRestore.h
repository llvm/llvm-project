//===-- SaveAndRestore.h - Utility  -------------------------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides utility classes that uses RAII to save and restore
//  values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SAVERESTORE
#define LLVM_ADT_SAVERESTORE

namespace llvm {

// SaveAndRestore - A utility class that uses RAII to save and restore
//  the value of a variable.
template<typename T>
struct SaveAndRestore {
  SaveAndRestore(T& x) : X(x), old_value(x) {}
  SaveAndRestore(T& x, const T &new_value) : X(x), old_value(x) {
    X = new_value;
  }
  ~SaveAndRestore() { X = old_value; }
  T get() { return old_value; }
private:
  T& X;
  T old_value;
};

// SaveOr - Similar to SaveAndRestore.  Operates only on bools; the old
//  value of a variable is saved, and during the dstor the old value is
//  or'ed with the new value.
struct SaveOr {
  SaveOr(bool& x) : X(x), old_value(x) { x = false; }
  ~SaveOr() { X |= old_value; }
private:
  bool& X;
  const bool old_value;
};

}
#endif
