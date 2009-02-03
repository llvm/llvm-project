//===-- llvm/Support/Casting.h - Allow flexible, checked, casts -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the isa<X>(), cast<X>(), dyn_cast<X>(), cast_or_null<X>(),
// and dyn_cast_or_null<X>() templates.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CASTING_H
#define LLVM_SUPPORT_CASTING_H

#include <cassert>

namespace llvm {

//===----------------------------------------------------------------------===//
//                          isa<x> Support Templates
//===----------------------------------------------------------------------===//

template<typename FromCl> struct isa_impl_cl;

// Define a template that can be specialized by smart pointers to reflect the
// fact that they are automatically dereferenced, and are not involved with the
// template selection process...  the default implementation is a noop.
//
template<typename From> struct simplify_type {
  typedef       From SimpleType;        // The real type this represents...

  // An accessor to get the real value...
  static SimpleType &getSimplifiedValue(From &Val) { return Val; }
};

template<typename From> struct simplify_type<const From> {
  typedef const From SimpleType;
  static SimpleType &getSimplifiedValue(const From &Val) {
    return simplify_type<From>::getSimplifiedValue(static_cast<From&>(Val));
  }
};


// isa<X> - Return true if the parameter to the template is an instance of the
// template type argument.  Used like this:
//
//  if (isa<Type*>(myVal)) { ... }
//
template <typename To, typename From>
inline bool isa_impl(const From &Val) {
  return To::classof(&Val);
}

template<typename To, typename From, typename SimpleType>
struct isa_impl_wrap {
  // When From != SimplifiedType, we can simplify the type some more by using
  // the simplify_type template.
  static bool doit(const From &Val) {
    return isa_impl_cl<const SimpleType>::template
                    isa<To>(simplify_type<const From>::getSimplifiedValue(Val));
  }
};

template<typename To, typename FromTy>
struct isa_impl_wrap<To, const FromTy, const FromTy> {
  // When From == SimpleType, we are as simple as we are going to get.
  static bool doit(const FromTy &Val) {
    return isa_impl<To,FromTy>(Val);
  }
};

// isa_impl_cl - Use class partial specialization to transform types to a single
// canonical form for isa_impl.
//
template<typename FromCl>
struct isa_impl_cl {
  template<class ToCl>
  static bool isa(const FromCl &Val) {
    return isa_impl_wrap<ToCl,const FromCl,
                   typename simplify_type<const FromCl>::SimpleType>::doit(Val);
  }
};

// Specialization used to strip const qualifiers off of the FromCl type...
template<typename FromCl>
struct isa_impl_cl<const FromCl> {
  template<class ToCl>
  static bool isa(const FromCl &Val) {
    return isa_impl_cl<FromCl>::template isa<ToCl>(Val);
  }
};

// Define pointer traits in terms of base traits...
template<class FromCl>
struct isa_impl_cl<FromCl*> {
  template<class ToCl>
  static bool isa(FromCl *Val) {
    return isa_impl_cl<FromCl>::template isa<ToCl>(*Val);
  }
};

// Define reference traits in terms of base traits...
template<class FromCl>
struct isa_impl_cl<FromCl&> {
  template<class ToCl>
  static bool isa(FromCl &Val) {
    return isa_impl_cl<FromCl>::template isa<ToCl>(&Val);
  }
};

template <class X, class Y>
inline bool isa(const Y &Val) {
  return isa_impl_cl<Y>::template isa<X>(Val);
}

//===----------------------------------------------------------------------===//
//                          cast<x> Support Templates
//===----------------------------------------------------------------------===//

template<class To, class From> struct cast_retty;


// Calculate what type the 'cast' function should return, based on a requested
// type of To and a source type of From.
template<class To, class From> struct cast_retty_impl {
  typedef To& ret_type;         // Normal case, return Ty&
};
template<class To, class From> struct cast_retty_impl<To, const From> {
  typedef const To &ret_type;   // Normal case, return Ty&
};

template<class To, class From> struct cast_retty_impl<To, From*> {
  typedef To* ret_type;         // Pointer arg case, return Ty*
};

template<class To, class From> struct cast_retty_impl<To, const From*> {
  typedef const To* ret_type;   // Constant pointer arg case, return const Ty*
};

template<class To, class From> struct cast_retty_impl<To, const From*const> {
  typedef const To* ret_type;   // Constant pointer arg case, return const Ty*
};


template<class To, class From, class SimpleFrom>
struct cast_retty_wrap {
  // When the simplified type and the from type are not the same, use the type
  // simplifier to reduce the type, then reuse cast_retty_impl to get the
  // resultant type.
  typedef typename cast_retty<To, SimpleFrom>::ret_type ret_type;
};

template<class To, class FromTy>
struct cast_retty_wrap<To, FromTy, FromTy> {
  // When the simplified type is equal to the from type, use it directly.
  typedef typename cast_retty_impl<To,FromTy>::ret_type ret_type;
};

template<class To, class From>
struct cast_retty {
  typedef typename cast_retty_wrap<To, From,
                   typename simplify_type<From>::SimpleType>::ret_type ret_type;
};

// Ensure the non-simple values are converted using the simplify_type template
// that may be specialized by smart pointers...
//
template<class To, class From, class SimpleFrom> struct cast_convert_val {
  // This is not a simple type, use the template to simplify it...
  static typename cast_retty<To, From>::ret_type doit(const From &Val) {
    return cast_convert_val<To, SimpleFrom,
      typename simplify_type<SimpleFrom>::SimpleType>::doit(
                          simplify_type<From>::getSimplifiedValue(Val));
  }
};

template<class To, class FromTy> struct cast_convert_val<To,FromTy,FromTy> {
  // This _is_ a simple type, just cast it.
  static typename cast_retty<To, FromTy>::ret_type doit(const FromTy &Val) {
    return reinterpret_cast<typename cast_retty<To, FromTy>::ret_type>(
                         const_cast<FromTy&>(Val));
  }
};



// cast<X> - Return the argument parameter cast to the specified type.  This
// casting operator asserts that the type is correct, so it does not return null
// on failure.  But it will correctly return NULL when the input is NULL.
// Used Like this:
//
//  cast<Instruction>(myVal)->getParent()
//
template <class X, class Y>
inline typename cast_retty<X, Y>::ret_type cast(const Y &Val) {
  assert(isa<X>(Val) && "cast<Ty>() argument of incompatible type!");
  return cast_convert_val<X, Y,
                          typename simplify_type<Y>::SimpleType>::doit(Val);
}

// cast_or_null<X> - Functionally identical to cast, except that a null value is
// accepted.
//
template <class X, class Y>
inline typename cast_retty<X, Y*>::ret_type cast_or_null(Y *Val) {
  if (Val == 0) return 0;
  assert(isa<X>(Val) && "cast_or_null<Ty>() argument of incompatible type!");
  return cast<X>(Val);
}


// dyn_cast<X> - Return the argument parameter cast to the specified type.  This
// casting operator returns null if the argument is of the wrong type, so it can
// be used to test for a type as well as cast if successful.  This should be
// used in the context of an if statement like this:
//
//  if (const Instruction *I = dyn_cast<Instruction>(myVal)) { ... }
//

template <class X, class Y>
inline typename cast_retty<X, Y>::ret_type dyn_cast(const Y &Val) {
  return isa<X>(Val) ? cast<X, Y>(Val) : 0;
}

// dyn_cast_or_null<X> - Functionally identical to dyn_cast, except that a null
// value is accepted.
//
template <class X, class Y>
inline typename cast_retty<X, Y>::ret_type dyn_cast_or_null(const Y &Val) {
  return (Val && isa<X>(Val)) ? cast<X, Y>(Val) : 0;
}


#ifdef DEBUG_CAST_OPERATORS
#include "llvm/Support/Streams.h"

struct bar {
  bar() {}
private:
  bar(const bar &);
};
struct foo {
  void ext() const;
  /*  static bool classof(const bar *X) {
    cerr << "Classof: " << X << "\n";
    return true;
    }*/
};

template <> inline bool isa_impl<foo,bar>(const bar &Val) {
  cerr << "Classof: " << &Val << "\n";
  return true;
}


bar *fub();
void test(bar &B1, const bar *B2) {
  // test various configurations of const
  const bar &B3 = B1;
  const bar *const B4 = B2;

  // test isa
  if (!isa<foo>(B1)) return;
  if (!isa<foo>(B2)) return;
  if (!isa<foo>(B3)) return;
  if (!isa<foo>(B4)) return;

  // test cast
  foo &F1 = cast<foo>(B1);
  const foo *F3 = cast<foo>(B2);
  const foo *F4 = cast<foo>(B2);
  const foo &F8 = cast<foo>(B3);
  const foo *F9 = cast<foo>(B4);
  foo *F10 = cast<foo>(fub());

  // test cast_or_null
  const foo *F11 = cast_or_null<foo>(B2);
  const foo *F12 = cast_or_null<foo>(B2);
  const foo *F13 = cast_or_null<foo>(B4);
  const foo *F14 = cast_or_null<foo>(fub());  // Shouldn't print.

  // These lines are errors...
  //foo *F20 = cast<foo>(B2);  // Yields const foo*
  //foo &F21 = cast<foo>(B3);  // Yields const foo&
  //foo *F22 = cast<foo>(B4);  // Yields const foo*
  //foo &F23 = cast_or_null<foo>(B1);
  //const foo &F24 = cast_or_null<foo>(B3);
}

bar *fub() { return 0; }
void main() {
  bar B;
  test(B, &B);
}

#endif

} // End llvm namespace

#endif
