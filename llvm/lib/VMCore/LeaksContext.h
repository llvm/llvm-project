//===- LeaksContext.h - LeadDetector Implementation ------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines various helper methods and classes used by
// LLVMContextImpl for leaks detectors.
//
//===----------------------------------------------------------------------===//

#include "llvm/Value.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {

template <class T>
struct PrinterTrait {
  static void print(const T* P) { errs() << P; }
};

template<>
struct PrinterTrait<Value> {
  static void print(const Value* P) { errs() << *P; }
};

template <typename T>
struct LeakDetectorImpl {
  explicit LeakDetectorImpl(const char* const name = "") : 
    Cache(0), Name(name) { }

  void clear() {
    Cache = 0;
    Ts.clear();
  }
    
  void setName(const char* n) { 
    Name = n;
  }
    
  // Because the most common usage pattern, by far, is to add a
  // garbage object, then remove it immediately, we optimize this
  // case.  When an object is added, it is not added to the set
  // immediately, it is added to the CachedValue Value.  If it is
  // immediately removed, no set search need be performed.
  void addGarbage(const T* o) {
    assert(Ts.count(o) == 0 && "Object already in set!");
    if (Cache) {
      assert(Cache != o && "Object already in set!");
      Ts.insert(Cache);
    }
    Cache = o;
  }

  void removeGarbage(const T* o) {
    if (o == Cache)
      Cache = 0; // Cache hit
    else
      Ts.erase(o);
  }

  bool hasGarbage(const std::string& Message) {
    addGarbage(0); // Flush the Cache

    assert(Cache == 0 && "No value should be cached anymore!");

    if (!Ts.empty()) {
      errs() << "Leaked " << Name << " objects found: " << Message << ":\n";
      for (typename SmallPtrSet<const T*, 8>::iterator I = Ts.begin(),
           E = Ts.end(); I != E; ++I) {
        errs() << '\t';
        PrinterTrait<T>::print(*I);
        errs() << '\n';
      }
      errs() << '\n';

      return true;
    }
    
    return false;
  }

private:
  SmallPtrSet<const T*, 8> Ts;
  const T* Cache;
  const char* Name;
};

}
