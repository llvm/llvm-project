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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DynamicLibrary.h"
#include <memory>

namespace llvm {
/// A simple registry entry which provides only a name, description, and
/// an `CtorParamTypes&&` variadic parameterized constructor.
template <typename T, typename... CtorParamTypes> class SimpleRegistryEntry {
  using FactoryFnRef = function_ref<std::unique_ptr<T>(CtorParamTypes &&...)>;
  StringRef Name, Desc;
  FactoryFnRef Ctor;

public:
  SimpleRegistryEntry(StringRef N, StringRef D, FactoryFnRef C)
      : Name(N), Desc(D), Ctor(C) {}

  StringRef getName() const { return Name; }
  StringRef getDesc() const { return Desc; }
  std::unique_ptr<T> instantiate(CtorParamTypes &&...Params) const {
    return Ctor(std::forward<CtorParamTypes>(Params)...);
  }
};

template <typename T, typename... CtorParamTypes> class Registry;
namespace detail {
template <typename R> struct IsRegistryType : std::false_type {};
template <typename T, typename... CtorParamTypes>
struct IsRegistryType<Registry<T, CtorParamTypes...>> : std::true_type {};

/// The endpoint to the instance to hold registered components by a linked list.
///
/// This is split out from `Registry<T>` to guard against absence of or error in
/// an explicit instantiation, which causes an implicit instantiation.
/// Such instantiation breaks dylib buils on Windows because a reference to a
/// non dllimport-ed implicitly instantiated global variable can't be shared
/// across a DLL boundary.
template <typename R> struct RegistryLinkListStorage {
  typename R::node *Head;
  typename R::node *Tail;
};

/// The accessor to the endpoint.
///
/// ATTENTION: Be careful when attempting to "refactor" or "simplify" this.
///
/// - This shall not be a member of `RegistryLinkList`, to control instantiation
///   timing separately.
///
/// - The body also shall not be inlined here; otherwise, the guard from a
///   missed explicit instantiation definition won't work.
///
/// - Use an accessor function that returns a reference to a function-local
///   static variable, rather than using a variable template or a static member
///   variable in a class template directly since functions can be imported
///   without the `dllimport` annotation.
template <typename R> RegistryLinkListStorage<R> &getRegistryLinkListInstance();

/// Utility to guard against missing declarations or mismatched use of
/// `LLVM_DECLARE_REGISTRY` and `LLVM_INSTANTIATE_REGISTRY`.
template <typename R>
struct RegistryLinkListDeclarationMarker : std::false_type {};
} // namespace detail

/// A global registry used in conjunction with static constructors to make
/// pluggable components (like targets or garbage collectors) "just work" when
/// linked with an executable.
///
/// To provide a Registry interface, follow these steps:
///
/// 1. Define your plugin base interface. The interface must have a virtual
///    destructor and the appropriate dllexport/dllimport/visibility annotation.
///
///        namespace your_ns {
///          class YOURLIB_ABI SomethingPluginBase {
///            virtual ~SomethingPluginBase() = default;
///            virtual void TheInterface() = 0;
///          };
///        }
///
/// 2. Declare an alias to your `Registry` for convenience.
///
///        namespace your_ns {
///          using YourRegistry = llvm::Registry<SomethingPluginBase>;
///        }
///
/// 3. Declare the specialization of the `Registry` with `LLVM_DECLARE_REGISTRY`
///    in the global namespace. The declaration must be placed before any use of
///    the `YourRegistry`.
///
///        LLVM_DECLARE_REGISTRY(your_ns::YourRegistry)
///
/// 4. In a .cpp file, define the specialization with `LLVM_DEFINE_REGISTRY`
///    in the global namespace.
///
///        LLVM_DEFINE_REGISTRY(your_ns::YourRegistry)
///
template <typename T, typename... CtorParamTypes> class Registry {
  static_assert(
      !detail::IsRegistryType<T>::value,
      "Trying to instantiate a wrong specialization 'Registry<Registry<...>>'");

public:
  using type = T;
  using entry = SimpleRegistryEntry<T, CtorParamTypes...>;
  static constexpr bool HasCtorParamTypes = sizeof...(CtorParamTypes) != 0;

  class node;
  class iterator;

private:
  Registry() = delete;

  friend class node;

public:
  /// Node in linked list of entries.
  ///
  class node {
    friend class iterator;
    friend Registry<T, CtorParamTypes...>;

    node *Next;
    const entry &Val;

  public:
    node(const entry &V) : Next(nullptr), Val(V) {}
  };

  /// Add a node to the Registry: this is the interface between the plugin and
  /// the executable.
  ///
  static void add_node(node *N) {
    auto &[Head, Tail] = detail::getRegistryLinkListInstance<Registry>();
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
    iterator &operator++() {
      Cur = Cur->Next;
      return *this;
    }
    const entry &operator*() const { return Cur->Val; }
  };

  static iterator begin() {
    return iterator(detail::getRegistryLinkListInstance<Registry>().Head);
  }
  static iterator end() { return iterator(nullptr); }

  static iterator_range<iterator> entries() {
    return make_range(begin(), end());
  }

  /// A static registration template. Use like such:
  ///
  ///   Registry<Collector>::Add<FancyGC>
  ///   X("fancy-gc", "Newfangled garbage collector.");
  ///
  template <typename V> class Add {
    entry Entry;
    node Node;

    static std::unique_ptr<T> CtorFn(CtorParamTypes &&...Params) {
      static_assert(std::has_virtual_destructor_v<T>);
      return std::make_unique<V>(std::forward<CtorParamTypes>(Params)...);
    }

  public:
    Add(StringRef Name, StringRef Desc)
        : Entry(Name, Desc, CtorFn), Node(Entry) {
      add_node(&Node);
    }
  };
};

} // end namespace llvm

/// Helper macro to declare registry class.
///
/// The `LLVM_ABI_EXPORT` (i.e. __delcpsec(dllexport) on Win32) attached to the
/// declaration is mandatory since MSVC disallows adding dllexport after the
/// first non-exported specialization declaration, and it is generally safe. All
/// of link.exe, ld.bfd, and lld-link will attempt to import undefined symbols
/// (including template specializations) from DLLs, even if the symbol is
/// declared with dllexport, just like non-imported symbols. The dllimport
/// attribute is not eligible here, since the specialization may or may not be
/// defined in the same object, a static library, or an import library.
#define LLVM_DECLARE_REGISTRY(REGISTRY_CLASS)                                  \
  namespace llvm::detail {                                                     \
  template <>                                                                  \
  struct RegistryLinkListDeclarationMarker<REGISTRY_CLASS> : std::true_type {  \
  };                                                                           \
  template <>                                                                  \
  LLVM_ABI_EXPORT RegistryLinkListStorage<REGISTRY_CLASS> &                    \
  getRegistryLinkListInstance<REGISTRY_CLASS>();                               \
  }

/// Helper macro to define registry class.
///
/// Technically, these macros don't instantiate `Registry` despite the name.
/// They handle underlying storage that used by `Registry` indirectly, enforcing
/// proper declarations/definitions by compilers or linkers. If you forget to
/// place `LLVM_DEFINE_REGISTRY`, your linker will complain about a missing
/// `getRegistryLinkListInstance` definiton.
#define LLVM_DEFINE_REGISTRY(REGISTRY_CLASS)                                   \
  namespace llvm::detail {                                                     \
  static_assert(RegistryLinkListDeclarationMarker<REGISTRY_CLASS>::value,      \
                "Missing matching registry delcaration of " #REGISTRY_CLASS    \
                ". Place `LLVM_DECLARE_REGISTRY(" #REGISTRY_CLASS              \
                ")` in a header.");                                            \
  template <>                                                                  \
  LLVM_ABI_EXPORT RegistryLinkListStorage<REGISTRY_CLASS> &                    \
  getRegistryLinkListInstance<REGISTRY_CLASS>() {                              \
    static RegistryLinkListStorage<REGISTRY_CLASS> Instance;                   \
    return Instance;                                                           \
  }                                                                            \
  }

#define LLVM_DETAIL_INSTANTIATE_REGISTRY_1(ABITAG, REGISTRY_CLASS)             \
  LLVM_DECLARE_REGISTRY(REGISTRY_CLASS)                                        \
  LLVM_DEFINE_REGISTRY(REGISTRY_CLASS)                                         \
  namespace llvm {                                                             \
  template class ABITAG Registry<REGISTRY_CLASS::type>;                        \
  static_assert(!REGISTRY_CLASS::HasCtorParamTypes,                            \
                "LLVM_INSTANTIATE_REGISTRY can't be used with extra "          \
                "constructor parameter types. Use "                            \
                "LLVM_DECLARE/DEFINE_REGISTRY istead.");                       \
  }

/// Old style helper macro to instantiate registry class.
///
#ifdef _WIN32
#define LLVM_INSTANTIATE_REGISTRY(REGISTRY_CLASS)                              \
  LLVM_DETAIL_INSTANTIATE_REGISTRY_1(LLVM_ABI_EXPORT, REGISTRY_CLASS)
#else
#define LLVM_INSTANTIATE_REGISTRY(REGISTRY_CLASS)                              \
  LLVM_DETAIL_INSTANTIATE_REGISTRY_1(, REGISTRY_CLASS)
#endif

#endif // LLVM_SUPPORT_REGISTRY_H
