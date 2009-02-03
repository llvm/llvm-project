//== llvm/ADT/IntrusiveRefCntPtr.h - Smart Refcounting Pointer ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines IntrusiveRefCntPtr, a template class that
// implements a "smart" pointer for objects that maintain their own
// internal reference count, and RefCountedBase/RefCountedBaseVPTR, two
// generic base classes for objects that wish to have their lifetimes
// managed using reference counting.
//
// IntrusiveRefCntPtr is similar to Boost's intrusive_ptr with added
// LLVM-style casting.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_INTRUSIVE_REF_CNT_PTR
#define LLVM_ADT_INTRUSIVE_REF_CNT_PTR

#include <cassert>

#include "llvm/Support/Casting.h"

namespace llvm {

  template <class T>
  class IntrusiveRefCntPtr;

//===----------------------------------------------------------------------===//
/// RefCountedBase - A generic base class for objects that wish to
///  have their lifetimes managed using reference counts. Classes
///  subclass RefCountedBase to obtain such functionality, and are
///  typically handled with IntrusivePtr "smart pointers" (see below)
///  which automatically handle the management of reference counts.
///  Objects that subclass RefCountedBase should not be allocated on
///  the stack, as invoking "delete" (which is called when the
///  reference count hits 0) on such objects is an error.
//===----------------------------------------------------------------------===//
  template <class Derived>
  class RefCountedBase {
    unsigned ref_cnt;

  protected:
    RefCountedBase() : ref_cnt(0) {}

    void Retain() { ++ref_cnt; }
    void Release() {
      assert (ref_cnt > 0 && "Reference count is already zero.");
      if (--ref_cnt == 0) delete static_cast<Derived*>(this);
    }

    friend class IntrusiveRefCntPtr<Derived>;
  };

//===----------------------------------------------------------------------===//
/// RefCountedBaseVPTR - A class that has the same function as
///  RefCountedBase, but with a virtual destructor. Should be used
///  instead of RefCountedBase for classes that already have virtual
///  methods to enforce dynamic allocation via 'new'. Classes that
///  inherit from RefCountedBaseVPTR can't be allocated on stack -
///  attempting to do this will produce a compile error.
//===----------------------------------------------------------------------===//
  template <class Derived>
  class RefCountedBaseVPTR {
    unsigned ref_cnt;

  protected:
    RefCountedBaseVPTR() : ref_cnt(0) {}
    virtual ~RefCountedBaseVPTR() {}

    void Retain() { ++ref_cnt; }
    void Release() {
      assert (ref_cnt > 0 && "Reference count is already zero.");
      if (--ref_cnt == 0) delete this;
    }

    friend class IntrusiveRefCntPtr<Derived>;
  };

//===----------------------------------------------------------------------===//
/// IntrusiveRefCntPtr - A template class that implements a "smart pointer"
///  that assumes the wrapped object has a reference count associated
///  with it that can be managed via calls to
///  IntrusivePtrAddRef/IntrusivePtrRelease.  The smart pointers
///  manage reference counts via the RAII idiom: upon creation of
///  smart pointer the reference count of the wrapped object is
///  incremented and upon destruction of the smart pointer the
///  reference count is decremented.  This class also safely handles
///  wrapping NULL pointers.
///
/// Reference counting is implemented via calls to
///  Obj->Retain()/Obj->Release(). Release() is required to destroy
///  the object when the reference count reaches zero. Inheriting from
///  RefCountedBase/RefCountedBaseVPTR takes care of this
///  automatically.
//===----------------------------------------------------------------------===//
  template <typename T>
  class IntrusiveRefCntPtr {
    T* Obj;
    typedef IntrusiveRefCntPtr this_type;
  public:
    typedef T element_type;

    explicit IntrusiveRefCntPtr() : Obj(0) {}

    explicit IntrusiveRefCntPtr(T* obj) : Obj(obj) {
      retain();
    }

    IntrusiveRefCntPtr(const IntrusiveRefCntPtr& S) : Obj(S.Obj) {
      retain();
    }

    template <class X>
    IntrusiveRefCntPtr(const IntrusiveRefCntPtr<X>& S)
      : Obj(S.getPtr()) {
      retain();
    }

    IntrusiveRefCntPtr& operator=(const IntrusiveRefCntPtr& S) {
      replace(S.getPtr());
      return *this;
    }

    template <class X>
    IntrusiveRefCntPtr& operator=(const IntrusiveRefCntPtr<X>& S) {
      replace(S.getPtr());
      return *this;
    }

    IntrusiveRefCntPtr& operator=(T * S) {
      replace(S);
      return *this;
    }

    ~IntrusiveRefCntPtr() { release(); }

    T& operator*() const { return *Obj; }

    T* operator->() const { return Obj; }

    T* getPtr() const { return Obj; }

    typedef T* (IntrusiveRefCntPtr::*unspecified_bool_type) () const;
    operator unspecified_bool_type() const {
      return Obj == 0 ? 0 : &IntrusiveRefCntPtr::getPtr;
    }

    void swap(IntrusiveRefCntPtr& other) {
      T* tmp = other.Obj;
      other.Obj = Obj;
      Obj = tmp;
    }

  private:
    void retain() { if (Obj) Obj->Retain(); }
    void release() { if (Obj) Obj->Release(); }

    void replace(T* S) {
      this_type(S).swap(*this);
    }
  };

  template<class T, class U>
  inline bool operator==(const IntrusiveRefCntPtr<T>& A,
                         const IntrusiveRefCntPtr<U>& B)
  {
    return A.getPtr() == B.getPtr();
  }

  template<class T, class U>
  inline bool operator!=(const IntrusiveRefCntPtr<T>& A,
                         const IntrusiveRefCntPtr<U>& B)
  {
    return A.getPtr() != B.getPtr();
  }

  template<class T, class U>
  inline bool operator==(const IntrusiveRefCntPtr<T>& A,
                         U* B)
  {
    return A.getPtr() == B;
  }

  template<class T, class U>
  inline bool operator!=(const IntrusiveRefCntPtr<T>& A,
                         U* B)
  {
    return A.getPtr() != B;
  }

  template<class T, class U>
  inline bool operator==(T* A,
                         const IntrusiveRefCntPtr<U>& B)
  {
    return A == B.getPtr();
  }

  template<class T, class U>
  inline bool operator!=(T* A,
                         const IntrusiveRefCntPtr<U>& B)
  {
    return A != B.getPtr();
  }

//===----------------------------------------------------------------------===//
// LLVM-style downcasting support for IntrusiveRefCntPtr objects
//===----------------------------------------------------------------------===//

  template<class T> struct simplify_type<IntrusiveRefCntPtr<T> > {
    typedef T* SimpleType;
    static SimpleType getSimplifiedValue(const IntrusiveRefCntPtr<T>& Val) {
      return Val.getPtr();
    }
  };

  template<class T> struct simplify_type<const IntrusiveRefCntPtr<T> > {
    typedef T* SimpleType;
    static SimpleType getSimplifiedValue(const IntrusiveRefCntPtr<T>& Val) {
      return Val.getPtr();
    }
  };

} // end namespace llvm

#endif // LLVM_ADT_INTRUSIVE_REF_CNT_PTR
