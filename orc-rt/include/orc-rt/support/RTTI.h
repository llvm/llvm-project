//===------------- RTTI.h - RTTI support for ORC RT -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
//
// Provides an extensible RTTI mechanism, that can be used regardless of whether
// the runtime is built with -frtti or not. This is predominantly used to
// support error handling.
//
// Types identify themselves by name: each participating class declares a
// RTTIName string, and RTTIRoot defines methods for comparing them.
// Implementations of these methods can be injected into new classes using the
// RTTIExtends class template, which also documents the requirements RTTIName
// must satisfy.
//
// Names rather than addresses are used because a type's identity has to survive
// crossing a library boundary. An object may be constructed by one library and
// have its type queried by another, each with its own copy of orc-rt, so any
// per-type address would differ between them. Comparing names is unaffected.
// Within a single library the addresses do agree, and RTTIRoot records which
// library produced each value so that case can be fast-pathed to a pointer
// comparison.
//
// E.g.
//
//   @code{.cpp}
//   class MyBaseClass : public RTTIExtends<MyBaseClass, RTTIRoot> {
//   public:
//     static constexpr const char *RTTIName = "mylib::MyBaseClass";
//
//     virtual void foo() = 0;
//   };
//
//   class MyDerivedClass1 : public RTTIExtends<MyDerivedClass1, MyBaseClass> {
//   public:
//     static constexpr const char *RTTIName = "mylib::MyDerivedClass1";
//
//     void foo() override {}
//   };
//
//   class MyDerivedClass2 : public RTTIExtends<MyDerivedClass2, MyBaseClass> {
//   public:
//     static constexpr const char *RTTIName = "mylib::MyDerivedClass2";
//
//     void foo() override {}
//   };
//
//   void fn() {
//     std::unique_ptr<MyBaseClass> B = std::make_unique<MyDerivedClass1>();
//     outs() << isa<MyBaseClass>(B) << "\n"; // Outputs "1".
//     outs() << isa<MyDerivedClass1>(B) << "\n"; // Outputs "1".
//     outs() << isa<MyDerivedClass2>(B) << "\n"; // Outputs "0'.
//   }
//
//   @endcode
//
// Note:
//   This header was adapted from llvm/Support/ExtensibleRTTI.h, however the
// data structures are not shared and the code need not be kept in sync.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_RTTI_H
#define ORC_RT_SUPPORT_RTTI_H

#include "orc-rt-c/support/RTTI.h"

#include <cstring>
#include <string_view>
#include <type_traits>

namespace orc_rt {

class RTTIRoot;

inline orc_rt_RTTIRootRef wrap(RTTIRoot *R) noexcept {
  return reinterpret_cast<orc_rt_RTTIRootRef>(R);
}

inline RTTIRoot *unwrap(orc_rt_RTTIRootRef R) noexcept {
  return reinterpret_cast<RTTIRoot *>(R);
}

/// Use this to implement C RTTI support on the given type.
///
/// Type must be named unqualified: it is pasted into both the C symbol name and
/// an isA<> query, so a namespace-qualified name would produce a nonsense
/// symbol. Use this macro from the defining scope, or bring the type into scope
/// with a using-declaration instead.
#define ORC_RT_C_RTTI_IMPL(Type)                                               \
  extern "C" ORC_RT_C_EXPORT orc_rt_##Type##Ref orc_rt_##Type##_fromRTTIRoot(  \
      orc_rt_RTTIRootRef Obj) noexcept {                                       \
    if (!Obj)                                                                  \
      return nullptr;                                                          \
    if (unwrap(Obj)->isA<Type>())                                              \
      return reinterpret_cast<orc_rt_##Type##Ref>(Obj);                        \
    return nullptr;                                                            \
  }

class ErrorInfoBase;

template <typename ThisT, typename ParentT> class RTTIExtends;

/// Base class for the extensible RTTI hierarchy.
///
/// This class defines virtual methods, dynamicClassID and isA, that enable
/// type comparisons.
class RTTIRoot {
public:
  virtual ~RTTIRoot() noexcept = default;

  /// The name identifying this type. See RTTIExtends for the requirements this
  /// must satisfy.
  static constexpr const char *RTTIName = "orc_rt::RTTIRoot";

  /// Return the library ID for this value.
  ///
  /// This identifies which dylib produced the value, allowing us to fast-path
  /// type equality checks within the same library.
  const void *libraryID() const noexcept { return LibraryID; }

  /// Returns the RTTIName of the dynamic type of this RTTIRoot instance.
  virtual const char *dynamicRTTIName() const noexcept = 0;

  /// Check whether this instance is a subclass of QueryT.
  template <typename QueryT> bool isA() const noexcept {
    return libraryID() == &ThisLibraryID ? sameDylibIsA(QueryT::RTTIName)
                                         : differentDylibIsA(QueryT::RTTIName);
  }

  static bool classof(const RTTIRoot *R) noexcept { return R->isA<RTTIRoot>(); }

protected:
  /// Fast-path isA for values produced by this dylib.
  virtual bool sameDylibIsA(const char *const ClassName) const noexcept {
    return ClassName == RTTIName;
  }

  /// Slow-path isA for values produced by different dylibs.
  virtual bool differentDylibIsA(const char *const ClassName) const noexcept {
    return strcmp(ClassName, RTTIName) == 0;
  }

private:
  static char ThisLibraryID;
  const char *const LibraryID = &ThisLibraryID;
  virtual void anchor() noexcept;
};

/// Inheritance utility for extensible RTTI.
///
/// Supports single inheritance only: A class can only have one
/// ExtensibleRTTI-parent (i.e. a parent for which the isa<> test will work),
/// though it can have many non-ExtensibleRTTI parents.
///
/// RTTIExtents uses CRTP so the first template argument to RTTIExtends is the
/// newly introduced type, and the *second* argument is the parent class.
///
/// Each participating type must declare its own RTTIName:
///
/// class MyType : public RTTIExtends<MyType, RTTIRoot> {
/// public:
///   static constexpr const char *RTTIName = "mylib::MyType";
///   ...
/// };
///
/// class MyDerivedType : public RTTIExtends<MyDerivedType, MyType> {
/// public:
///   static constexpr const char *RTTIName = "mylib::MyDerivedType";
///   ...
/// };
///
/// RTTINames must be unique across every library in the process, not just
/// within one hierarchy. Types from different libraries are compared by name,
/// so two unrelated types that share a name would satisfy each other's isa<>
/// checks. Qualifying the name with its namespace, as above, is usually enough
/// to keep it unique.
///
/// Forgetting to declare RTTIName leaves ParentT's visible by inheritance,
/// which would make the type indistinguishable from its parent; RTTIExtends
/// static_asserts against that. It cannot detect a collision with an unrelated
/// type.
///
template <typename ThisT, typename ParentT> class RTTIExtends : public ParentT {
public:
  static_assert(!std::is_base_of_v<ErrorInfoBase, ParentT>,
                "RTTIExtends should not be used to define orc_rt custom error "
                "types, use ErrorExtends instead");

  // Inherit constructors from ParentT.
  using ParentT::ParentT;

  const char *dynamicRTTIName() const noexcept override {
    static_assert(std::string_view(ThisT::RTTIName) !=
                      std::string_view(ParentT::RTTIName),
                  "ThisT must define its own RTTIName, distinct from "
                  "ParentT::RTTIName (did you forget to shadow it, or copy "
                  "the parent's string literal instead of writing a new "
                  "one?)");
    return ThisT::RTTIName;
  }

  static bool classof(const RTTIRoot *R) noexcept { return R->isA<ThisT>(); }

protected:
  bool sameDylibIsA(const char *const ClassName) const noexcept override {
    return ClassName == ThisT::RTTIName || ParentT::sameDylibIsA(ClassName);
  }

  bool differentDylibIsA(const char *const ClassName) const noexcept override {
    return strcmp(ClassName, ThisT::RTTIName) == 0 ||
           ParentT::differentDylibIsA(ClassName);
  }
};

/// Returns true if the given value is an instance of the template type
/// parameter.
template <typename To, typename From> bool isa(const From &Value) noexcept {
  return To::classof(&Value);
}

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_RTTI_H
