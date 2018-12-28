//===--------------------------- Refcount.h ------------------------------===//
//
//                              SPIR Tools
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
/*
 * Contributed by: Intel Corporation
 */

#ifndef SPIRV_MANGLER_REFCOUNT_H
#define SPIRV_MANGLER_REFCOUNT_H

#include <assert.h>

namespace SPIR {

template <typename T> class RefCount {
public:
  RefCount() : Count(0), Ptr(0) {}

  RefCount(T *Ptr) : Ptr(Ptr) { Count = new int(1); }

  RefCount(const RefCount<T> &Other) { cpy(Other); }

  ~RefCount() {
    if (Count)
      dispose();
  }

  RefCount &operator=(const RefCount<T> &Other) {
    if (this == &Other)
      return *this;
    if (Count)
      dispose();
    cpy(Other);
    return *this;
  }

  void init(T *Ptr) {
    assert(!Ptr && "overrunning non NULL pointer");
    assert(!Count && "overrunning non NULL pointer");
    Count = new int(1);
    this->Ptr = Ptr;
  }

  bool isNull() const { return (!Ptr); }

  // Pointer access
  const T &operator*() const {
    sanity();
    return *Ptr;
  }

  T &operator*() {
    sanity();
    return *Ptr;
  }

  operator T *() { return Ptr; }

  operator const T *() const { return Ptr; }

  T *operator->() { return Ptr; }

  const T *operator->() const { return Ptr; }

private:
  void sanity() const {
    assert(Ptr && "NULL pointer");
    assert(Count && "NULL ref counter");
    assert(*Count && "zero ref counter");
  }

  void cpy(const RefCount<T> &Other) {
    Count = Other.Count;
    Ptr = Other.Ptr;
    if (Count)
      ++*Count;
  }

  void dispose() {
    sanity();
    if (0 == --*Count) {
      delete Count;
      delete Ptr;
      Ptr = 0;
      Count = 0;
    }
  }

  int *Count;
  T *Ptr;
}; // End RefCount

} // namespace SPIR

#endif // SPIRV_MANGLER_REFCOUNT_H
