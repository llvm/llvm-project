//===----- CCallback.h - utility for generating C callbacks -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for generating C callback pointers from methods.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_CCALLBACK_H
#define ORC_RT_CCALLBACK_H

#include <type_traits>

namespace orc_rt {

namespace detail {
// asCCallback produces C-ABI function pointers, so wrapped methods may only
// use C-compatible parameter and return types: no references, and trivially
// copyable (which every genuine C type is). C++ types with non-trivial
// move/copy semantics cannot cross a C boundary, so they are rejected here
// rather than silently accommodated.
template <typename T>
inline constexpr bool isCCompatibleArg =
    !std::is_reference_v<T> && std::is_trivially_copyable_v<T>;

template <typename T>
inline constexpr bool isCCompatibleRet =
    std::is_void_v<T> || isCCompatibleArg<T>;

// Four specializations cover the const x noexcept qualifier combinations on the
// member-function pointer. `fn` is always noexcept regardless of Meth's
// noexcept-ness; const methods take a `const void *` context and expose
// `class_type` as `const ClassT` so asCCallbackContext stays const-correct.
// Because every argument is a trivially copyable value, `Args` is passed on
// directly -- std::forward would be a no-op for such types.
template <typename MethT, MethT Meth> struct CCallbackImpl;

template <typename RetT, typename ClassT, typename... ArgTs,
          RetT (ClassT::*Meth)(ArgTs...)>
struct CCallbackImpl<RetT (ClassT::*)(ArgTs...), Meth> {
  using class_type = ClassT;
  static RetT fn(void *Obj, ArgTs... Args) noexcept {
    static_assert(isCCompatibleRet<RetT> && (... && isCCompatibleArg<ArgTs>),
                  "asCCallback requires C-compatible (non-reference, trivially "
                  "copyable) return and argument types");
    return (reinterpret_cast<ClassT *>(Obj)->*Meth)(Args...);
  }
};

template <typename RetT, typename ClassT, typename... ArgTs,
          RetT (ClassT::*Meth)(ArgTs...) noexcept>
struct CCallbackImpl<RetT (ClassT::*)(ArgTs...) noexcept, Meth> {
  using class_type = ClassT;
  static RetT fn(void *Obj, ArgTs... Args) noexcept {
    static_assert(isCCompatibleRet<RetT> && (... && isCCompatibleArg<ArgTs>),
                  "asCCallback requires C-compatible (non-reference, trivially "
                  "copyable) return and argument types");
    return (reinterpret_cast<ClassT *>(Obj)->*Meth)(Args...);
  }
};

template <typename RetT, typename ClassT, typename... ArgTs,
          RetT (ClassT::*Meth)(ArgTs...) const>
struct CCallbackImpl<RetT (ClassT::*)(ArgTs...) const, Meth> {
  using class_type = const ClassT;
  static RetT fn(const void *Obj, ArgTs... Args) noexcept {
    static_assert(isCCompatibleRet<RetT> && (... && isCCompatibleArg<ArgTs>),
                  "asCCallback requires C-compatible (non-reference, trivially "
                  "copyable) return and argument types");
    return (reinterpret_cast<const ClassT *>(Obj)->*Meth)(Args...);
  }
};

template <typename RetT, typename ClassT, typename... ArgTs,
          RetT (ClassT::*Meth)(ArgTs...) const noexcept>
struct CCallbackImpl<RetT (ClassT::*)(ArgTs...) const noexcept, Meth> {
  using class_type = const ClassT;
  static RetT fn(const void *Obj, ArgTs... Args) noexcept {
    static_assert(isCCompatibleRet<RetT> && (... && isCCompatibleArg<ArgTs>),
                  "asCCallback requires C-compatible (non-reference, trivially "
                  "copyable) return and argument types");
    return (reinterpret_cast<const ClassT *>(Obj)->*Meth)(Args...);
  }
};

} // namespace detail

/// Produces a C-callable function pointer that forwards to member function
/// `Meth`. The returned pointer has signature
///
///   RetT (*)(CtxT *Ctx, ArgTs...) noexcept
///
/// where `CtxT` is `void` for non-const methods and `const void` for const
/// methods. `Ctx` must point to the `Meth`'s class subobject (see
/// asCCallbackContext); it is cast straight back to the class type, so a raw
/// pointer to a derived object is NOT acceptable when the class is a base at a
/// non-zero offset.
///
/// The trampoline is `noexcept`: a C caller cannot unwind a C++ exception, so
/// an exception escaping `Meth` calls std::terminate. This matches the ORC
/// runtime convention that methods wrapped as C callbacks do not throw.
template <auto Meth>
constexpr auto asCCallback = detail::CCallbackImpl<decltype(Meth), Meth>::fn;

/// Returns the context pointer to pass alongside `asCCallback<Meth>`. It
/// static_casts `&Obj` to `Meth`'s class, which (a) applies any base-class
/// offset so the trampoline's cast recovers the correct `this`, and (b)
/// preserves const-ness (yielding `const ClassT *` for const methods). Always
/// obtain the context this way rather than casting `&Obj` directly, so the two
/// halves compose correctly under inheritance and const-qualification.
template <auto Meth, typename ObjT> inline auto *asCCallbackContext(ObjT &Obj) {
  return static_cast<
      typename detail::CCallbackImpl<decltype(Meth), Meth>::class_type *>(&Obj);
}

} // namespace orc_rt

#endif // ORC_RT_CCALLBACK_H
