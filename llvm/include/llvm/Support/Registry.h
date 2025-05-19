//=== Registry.h - Linker-supported plugin registries -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a registry template for discovering pluggable modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_REGISTRY_H
#define LLVM_SUPPORT_REGISTRY_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DynamicLibrary.h"
#include <memory>

namespace llvm {
  /// A simple registry entry which provides only a name, description, and
  /// no-argument constructor.
  template <typename T>
  class SimpleRegistryEntry {
    StringRef Name, Desc;
    std::unique_ptr<T> (*Ctor)();

  public:
    SimpleRegistryEntry(StringRef N, StringRef D, std::unique_ptr<T> (*C)())
        : Name(N), Desc(D), Ctor(C) {}

    StringRef getName() const { return Name; }
    StringRef getDesc() const { return Desc; }
    std::unique_ptr<T> instantiate() const { return Ctor(); }
  };

  /// A global registry used in conjunction with static constructors to make
  /// pluggable components (like targets or garbage collectors) "just work" when
  /// linked with an executable.
  template <typename T>
  class Registry {
  public:
    typedef T type;
    typedef SimpleRegistryEntry<T> entry;

    class node;
    class iterator;

  private:
    Registry() = delete;

    friend class node;
    // These must be must two separate declarations to workaround a 20 year
    // old MSVC bug with dllexport and multiple static fields in the same
    // declaration causing error C2487 "member of dll interface class may not
    // be declared with dll interface".
    // https://developercommunity.visualstudio.com/t/c2487-in-dllexport-class-with-static-members/69878
    static node *Head;
    static node *Tail;

  public:
    /// Node in linked list of entries.
    ///
    class node {
      friend class iterator;
      friend Registry<T>;

      node *Next;
      const entry& Val;

    public:
      node(const entry &V) : Next(nullptr), Val(V) {}
    };

    /// Add a node to the Registry: this is the interface between the plugin and
    /// the executable.
    ///
    /// This function is exported by the executable and called by the plugin to
    /// add a node to the executable's registry. Therefore it's not defined here
    /// to avoid it being instantiated in the plugin and is instead defined in
    /// the executable (see LLVM_INSTANTIATE_REGISTRY below).
    static void add_node(node *N) {
      if (Tail)
        Tail->Next = N;
      else
        Head = N;
      Tail = N;
    }

    /// Iterators for registry entries.
    ///
    class iterator
        : public llvm::iterator_facade_base<iterator, std::forward_iterator_tag,
                                            const entry> {
      const node *Cur;

    public:
      explicit iterator(const node *N) : Cur(N) {}

      bool operator==(const iterator &That) const { return Cur == That.Cur; }
      iterator &operator++() { Cur = Cur->Next; return *this; }
      const entry &operator*() const { return Cur->Val; }
    };

    // begin is not defined here in order to avoid usage of an undefined static
    // data member, instead it's instantiated by LLVM_INSTANTIATE_REGISTRY.
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

      static std::unique_ptr<T> CtorFn() { return std::make_unique<V>(); }

    public:
      Add(StringRef Name, StringRef Desc)
          : Entry(Name, Desc, CtorFn), Node(Entry) {
        add_node(&Node);
      }
    };
  };

} // end namespace llvm

#ifdef _WIN32
/// Instantiate a registry class.
#define LLVM_INSTANTIATE_REGISTRY(REGISTRY_CLASS)                              \
  namespace llvm {                                                             \
  template <typename T>                                                        \
  typename Registry<T>::node *Registry<T>::Head = nullptr;                     \
  template <typename T>                                                        \
  typename Registry<T>::node *Registry<T>::Tail = nullptr;                     \
  template class LLVM_ABI_EXPORT Registry<REGISTRY_CLASS::type>;               \
  }
#else
#define LLVM_INSTANTIATE_REGISTRY(REGISTRY_CLASS)                              \
  namespace llvm {                                                             \
  template <typename T>                                                        \
  typename Registry<T>::node *Registry<T>::Head = nullptr;                     \
  template <typename T>                                                        \
  typename Registry<T>::node *Registry<T>::Tail = nullptr;                     \
  template class Registry<REGISTRY_CLASS::type>;                               \
  }
#endif

#endif // LLVM_SUPPORT_REGISTRY_H
