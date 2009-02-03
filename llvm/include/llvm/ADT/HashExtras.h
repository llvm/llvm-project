//===-- llvm/ADT/HashExtras.h - Useful functions for STL hash ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some templates that are useful if you are working with the
// STL Hashed containers.
//
// No library is required when using these functinons.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_HASHEXTRAS_H
#define LLVM_ADT_HASHEXTRAS_H

#include "llvm/ADT/hash_map.h"
#include <string>

// Cannot specialize hash template from outside of the std namespace.
namespace HASH_NAMESPACE {

// Provide a hash function for arbitrary pointers...
template <class T> struct hash<T *> {
  inline size_t operator()(const T *Val) const {
    return reinterpret_cast<size_t>(Val);
  }
};

template <> struct hash<std::string> {
  size_t operator()(std::string const &str) const {
    return hash<char const *>()(str.c_str());
  }
};

}  // End namespace std

#endif
