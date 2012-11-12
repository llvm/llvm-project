//===- llvm/Support/type_traits.h - Simplfied type traits -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a template class that determines if a type is a class or
// not. The basic mechanism, based on using the pointer to member function of
// a zero argument to a function was "boosted" from the boost type_traits
// library. See http://www.boost.org/ for all the gory details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TYPE_TRAITS_H
#define LLVM_SUPPORT_TYPE_TRAITS_H

#include "llvm/Support/DataTypes.h"
#include <cstddef>
#include <utility>

#ifndef __has_feature
#define LLVM_DEFINED_HAS_FEATURE
#define __has_feature(x) 0
#endif

// This is actually the conforming implementation which works with abstract
// classes.  However, enough compilers have trouble with it that most will use
// the one in boost/type_traits/object_traits.hpp. This implementation actually
// works with VC7.0, but other interactions seem to fail when we use it.

namespace llvm {
  
namespace dont_use
{
    // These two functions should never be used. They are helpers to
    // the is_class template below. They cannot be located inside
    // is_class because doing so causes at least GCC to think that
    // the value of the "value" enumerator is not constant. Placing
    // them out here (for some strange reason) allows the sizeof
    // operator against them to magically be constant. This is
    // important to make the is_class<T>::value idiom zero cost. it
    // evaluates to a constant 1 or 0 depending on whether the
    // parameter T is a class or not (respectively).
    template<typename T> char is_class_helper(void(T::*)());
    template<typename T> double is_class_helper(...);
}

template <typename T>
struct is_class
{
  // is_class<> metafunction due to Paul Mensonides (leavings@attbi.com). For
  // more details:
  // http://groups.google.com/groups?hl=en&selm=000001c1cc83%24e154d5e0%247772e50c%40c161550a&rnum=1
public:
  static const bool value =
      sizeof(char) == sizeof(dont_use::is_class_helper<T>(0));
};
  
  
/// isPodLike - This is a type trait that is used to determine whether a given
/// type can be copied around with memcpy instead of running ctors etc.
template <typename T>
struct isPodLike {
#if __has_feature(is_trivially_copyable)
  // If the compiler supports the is_trivially_copyable trait use it, as it
  // matches the definition of isPodLike closely.
  static const bool value = __is_trivially_copyable(T);
#else
  // If we don't know anything else, we can (at least) assume that all non-class
  // types are PODs.
  static const bool value = !is_class<T>::value;
#endif
};

// std::pair's are pod-like if their elements are.
template<typename T, typename U>
struct isPodLike<std::pair<T, U> > {
  static const bool value = isPodLike<T>::value && isPodLike<U>::value;
};
  

template <class T, T v>
struct integral_constant {
  typedef T value_type;
  static const value_type value = v;
  typedef integral_constant<T,v> type;
  operator value_type() { return value; }
};

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

/// \brief Metafunction that determines whether the two given types are 
/// equivalent.
template<typename T, typename U> struct is_same       : public false_type {};
template<typename T>             struct is_same<T, T> : public true_type {};

/// \brief Metafunction that removes const qualification from a type.
template <typename T> struct remove_const          { typedef T type; };
template <typename T> struct remove_const<const T> { typedef T type; };

/// \brief Metafunction that removes volatile qualification from a type.
template <typename T> struct remove_volatile             { typedef T type; };
template <typename T> struct remove_volatile<volatile T> { typedef T type; };

/// \brief Metafunction that removes both const and volatile qualification from
/// a type.
template <typename T> struct remove_cv {
  typedef typename remove_const<typename remove_volatile<T>::type>::type type;
};

/// \brief Helper to implement is_integral metafunction.
template <typename T> struct is_integral_impl           : false_type {};
template <> struct is_integral_impl<         bool>      : true_type {};
template <> struct is_integral_impl<         char>      : true_type {};
template <> struct is_integral_impl<  signed char>      : true_type {};
template <> struct is_integral_impl<unsigned char>      : true_type {};
template <> struct is_integral_impl<         wchar_t>   : true_type {};
template <> struct is_integral_impl<         short>     : true_type {};
template <> struct is_integral_impl<unsigned short>     : true_type {};
template <> struct is_integral_impl<         int>       : true_type {};
template <> struct is_integral_impl<unsigned int>       : true_type {};
template <> struct is_integral_impl<         long>      : true_type {};
template <> struct is_integral_impl<unsigned long>      : true_type {};
template <> struct is_integral_impl<         long long> : true_type {};
template <> struct is_integral_impl<unsigned long long> : true_type {};

/// \brief Metafunction that determines whether the given type is an integral
/// type.
template <typename T>
struct is_integral : is_integral_impl<T> {};

/// \brief Metafunction to remove reference from a type.
template <typename T> struct remove_reference { typedef T type; };
template <typename T> struct remove_reference<T&> { typedef T type; };

/// \brief Metafunction that determines whether the given type is a pointer
/// type.
template <typename T> struct is_pointer : false_type {};
template <typename T> struct is_pointer<T*> : true_type {};
template <typename T> struct is_pointer<T* const> : true_type {};
template <typename T> struct is_pointer<T* volatile> : true_type {};
template <typename T> struct is_pointer<T* const volatile> : true_type {};

/// \brief Metafunction that determines whether the given type is either an
/// integral type or an enumeration type.
///
/// Note that this accepts potentially more integral types than we whitelist
/// above for is_integral because it is based on merely being convertible
/// implicitly to an integral type.
template <typename T> class is_integral_or_enum {
  // Provide an overload which can be called with anything implicitly
  // convertible to an unsigned long long. This should catch integer types and
  // enumeration types at least. We blacklist classes with conversion operators
  // below.
  static double check_int_convertible(unsigned long long);
  static char check_int_convertible(...);

  typedef typename remove_reference<T>::type UnderlyingT;
  static UnderlyingT &nonce_instance;

public:
  static const bool
    value = (!is_class<UnderlyingT>::value && !is_pointer<UnderlyingT>::value &&
             !is_same<UnderlyingT, float>::value &&
             !is_same<UnderlyingT, double>::value &&
             sizeof(char) != sizeof(check_int_convertible(nonce_instance)));
};

// enable_if_c - Enable/disable a template based on a metafunction
template<bool Cond, typename T = void>
struct enable_if_c {
  typedef T type;
};

template<typename T> struct enable_if_c<false, T> { };
  
// enable_if - Enable/disable a template based on a metafunction
template<typename Cond, typename T = void>
struct enable_if : public enable_if_c<Cond::value, T> { };

namespace dont_use {
  template<typename Base> char base_of_helper(const volatile Base*);
  template<typename Base> double base_of_helper(...);
}

/// is_base_of - Metafunction to determine whether one type is a base class of
/// (or identical to) another type.
template<typename Base, typename Derived>
struct is_base_of {
  static const bool value 
    = is_class<Base>::value && is_class<Derived>::value &&
      sizeof(char) == sizeof(dont_use::base_of_helper<Base>((Derived*)0));
};

// remove_pointer - Metafunction to turn Foo* into Foo.  Defined in
// C++0x [meta.trans.ptr].
template <typename T> struct remove_pointer { typedef T type; };
template <typename T> struct remove_pointer<T*> { typedef T type; };
template <typename T> struct remove_pointer<T*const> { typedef T type; };
template <typename T> struct remove_pointer<T*volatile> { typedef T type; };
template <typename T> struct remove_pointer<T*const volatile> {
    typedef T type; };

template <bool, typename T, typename F>
struct conditional { typedef T type; };

template <typename T, typename F>
struct conditional<false, T, F> { typedef F type; };

}

#ifdef LLVM_DEFINED_HAS_FEATURE
#undef __has_feature
#endif

#endif
