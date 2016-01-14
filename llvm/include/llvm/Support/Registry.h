//=== Registry.h - Linker-supported plugin registries -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines a registry template for discovering pluggable modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_REGISTRY_H
#define LLVM_SUPPORT_REGISTRY_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include <memory>

namespace llvm {
  /// A simple registry entry which provides only a name, description, and
  /// no-argument constructor.
  template <typename T>
  class SimpleRegistryEntry {
    const char *Name, *Desc;
    std::unique_ptr<T> (*Ctor)();

  public:
    SimpleRegistryEntry(const char *N, const char *D, std::unique_ptr<T> (*C)())
      : Name(N), Desc(D), Ctor(C)
    {}

    const char *getName() const { return Name; }
    const char *getDesc() const { return Desc; }
    std::unique_ptr<T> instantiate() const { return Ctor(); }
  };

  /// A global registry used in conjunction with static constructors to make
  /// pluggable components (like targets or garbage collectors) "just work" when
  /// linked with an executable.
  template <typename T>
  class Registry {
  public:
    typedef SimpleRegistryEntry<T> entry;

    class node;
    class iterator;

  private:
    Registry() = delete;

    friend class node;
    static node *Head, *Tail;

  public:
    /// Node in linked list of entries.
    ///
    class node {
      friend class iterator;

      node *Next;
      const entry& Val;

    public:
      node(const entry& V) : Next(nullptr), Val(V) {
        if (Tail)
          Tail->Next = this;
        else
          Head = this;
        Tail = this;
      }
    };

    /// Iterators for registry entries.
    ///
    class iterator {
      const node *Cur;

    public:
      explicit iterator(const node *N) : Cur(N) {}

      bool operator==(const iterator &That) const { return Cur == That.Cur; }
      bool operator!=(const iterator &That) const { return Cur != That.Cur; }
      iterator &operator++() { Cur = Cur->Next; return *this; }
      const entry &operator*() const { return Cur->Val; }
      const entry *operator->() const { return &Cur->Val; }
    };

    static iterator begin() { return iterator(Head); }
    static iterator end()   { return iterator(nullptr); }

    static iterator_range<iterator> entries() {
      return make_range(begin(), end());
    }

    /// A static registration template. Use like such:
    ///
    ///   Registry<Collector>::Add<FancyGC>
    ///   X("fancy-gc", "Newfangled garbage collector.");
    ///
    /// Use of this template requires that:
    ///
    ///  1. The registered subclass has a default constructor.
    template <typename V>
    class Add {
      entry Entry;
      node Node;

      static std::unique_ptr<T> CtorFn() { return make_unique<V>(); }

    public:
      Add(const char *Name, const char *Desc)
        : Entry(Name, Desc, CtorFn), Node(Entry) {}
    };
  };

  
  // Since these are defined in a header file, plugins must be sure to export
  // these symbols.
  template <typename T>
  typename Registry<T>::node *Registry<T>::Head;

  template <typename T>
  typename Registry<T>::node *Registry<T>::Tail;
} // end namespace llvm

#endif // LLVM_SUPPORT_REGISTRY_H
