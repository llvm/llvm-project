//===--- StringSet.h - The LLVM Compiler Driver -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  StringSet - A set-like wrapper for the StringMap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGSET_H
#define LLVM_ADT_STRINGSET_H

#include "llvm/ADT/StringMap.h"
#include <cassert>

namespace llvm {

  /// StringSet - A wrapper for StringMap that provides set-like
  /// functionality.  Only insert() and count() methods are used by my
  /// code.
  template <class AllocatorTy = llvm::MallocAllocator>
  class StringSet : public llvm::StringMap<char, AllocatorTy> {
    typedef llvm::StringMap<char, AllocatorTy> base;
  public:
    bool insert(const std::string& InLang) {
      assert(!InLang.empty());
      const char* KeyStart = &InLang[0];
      const char* KeyEnd = KeyStart + InLang.size();
      return base::insert(llvm::StringMapEntry<char>::
                          Create(KeyStart, KeyEnd, base::getAllocator(), '+'));
    }
  };
}

#endif // LLVM_ADT_STRINGSET_H
