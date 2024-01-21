//===---- ExclusiveAccess.h - Helper for exclusive access data structures -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_EXCLUSIVE_ACCESS
#define OMPTARGET_EXCLUSIVE_ACCESS

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mutex>

/// Forward declaration.
template <typename Ty> struct Accessor;

/// A protected object is a simple wrapper to allocate an object of type \p Ty
/// together with a mutex that guards accesses to the object. The only way to
/// access the object is through the "exclusive accessor" which will lock the
/// mutex accordingly.
template <typename Ty> struct ProtectedObj {
  using AccessorTy = Accessor<Ty>;

  /// Get an exclusive access Accessor object. \p DoNotGetAccess allows to
  /// create an accessor that is not owning anything based on a boolean
  /// condition.
  AccessorTy getExclusiveAccessor(bool DoNotGetAccess = false);

private:
  Ty Obj;
  std::mutex Mtx;
  friend struct Accessor<Ty>;
};

/// Helper to provide transparent exclusive access to protected objects.
template <typename Ty> struct Accessor {
  /// Default constructor does not own anything and cannot access anything.
  Accessor() : Ptr(nullptr) {}

  /// Constructor to get exclusive access by locking the mutex protecting the
  /// underlying object.
  Accessor(ProtectedObj<Ty> &PO) : Ptr(&PO) { lock(); }

  /// Constructor to get exclusive access by taking it from \p Other.
  Accessor(Accessor<Ty> &&Other) : Ptr(Other.Ptr) { Other.Ptr = nullptr; }

  Accessor(Accessor &Other) = delete;

  /// If the object is still owned when the lifetime ends we give up access.
  ~Accessor() { unlock(); }

  /// Give up access to the underlying object, virtually "destroying" the
  /// accessor even if the object is still life.
  void destroy() {
    unlock();
    Ptr = nullptr;
  }

  /// Provide transparent access to the underlying object.
  Ty &operator*() {
    assert(Ptr && "Trying to access an object through a non-owning (or "
                  "destroyed) accessor!");
    return Ptr->Obj;
  }
  Ty *operator->() {
    assert(Ptr && "Trying to access an object through a non-owning (or "
                  "destroyed) accessor!");
    return &Ptr->Obj;
  }

private:
  /// Lock the underlying object if there is one.
  void lock() {
    if (Ptr)
      Ptr->Mtx.lock();
  }

  /// Unlock the underlying object if there is one.
  void unlock() {
    if (Ptr)
      Ptr->Mtx.unlock();
  }

  /// Pointer to the underlying object or null if the accessor lost access,
  /// e.g., after a destroy call.
  ProtectedObj<Ty> *Ptr;
};

template <typename Ty>
Accessor<Ty> ProtectedObj<Ty>::getExclusiveAccessor(bool DoNotGetAccess) {
  if (DoNotGetAccess)
    return Accessor<Ty>();
  return Accessor<Ty>(*this);
}

#endif
