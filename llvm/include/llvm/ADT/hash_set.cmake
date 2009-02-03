//===-- llvm/ADT/hash_set - "Portable" wrapper around hash_set --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// vim:ft=cpp
//
// This file provides a wrapper around the mysterious <hash_set> header file
// that seems to move around between GCC releases into and out of namespaces at
// will.  #including this header will cause hash_set to be available in the
// global namespace.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_HASH_SET
#define LLVM_ADT_HASH_SET

// Compiler Support Matrix
//
// Version   Namespace   Header File
//  2.95.x       ::        hash_set
//  3.0.4       std      ext/hash_set
//  3.1      __gnu_cxx   ext/hash_set
//  HP aCC6     std      stdex/rw/hashset.h
//  MS VC++    stdext      hash_set

#cmakedefine HAVE_GNU_EXT_HASH_SET
#cmakedefine HAVE_STD_EXT_HASH_SET
#cmakedefine HAVE_GLOBAL_HASH_SET
#cmakedefine HAVE_RW_STDEX_HASH_SET_H

// GCC versions 3.1 and later put hash_set in <ext/hash_set> and in
// the __gnu_cxx namespace.
#if defined(HAVE_GNU_EXT_HASH_SET)
# include <ext/hash_set>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE __gnu_cxx
# endif

// GCC 3.0.x puts hash_set in <ext/hash_set> and in the std namespace.
#elif defined(HAVE_STD_EXT_HASH_SET)
# include <ext/hash_set>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE std
# endif

// Older compilers such as GCC before version 3.0 do not keep
// extensions in the `ext' directory, and ignore the `std' namespace.
#elif defined(HAVE_GLOBAL_HASH_SET)
# include <hash_set>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE std
# endif

// HP aCC doesn't include an SGI-like hash_set. For this platform (or
// any others using Rogue Wave Software's Tools.h++ library), we wrap
// around them in std::
#elif defined(HAVE_RW_STDEX_HASH_SET_H)
# include <rw/stdex/hashset.h>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE std
# endif

// Support Microsoft VC++.
#elif defined(_MSC_VER)
# include <hash_set>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE stdext
# endif

// Give a warning if we couldn't find it, instead of (or in addition to)
// randomly doing something dumb.
#else
# warning "Autoconfiguration failed to find the hash_set header file."
#endif

// we wrap Rogue Wave Tools.h++ rw_hashset into something SGI-looking, here:
#ifdef HAVE_RW_STDEX_HASH_SET_H
namespace HASH_NAMESPACE {

/*
template <class DataType> struct hash {
    unsigned int operator()(const unsigned int& x) const {
	return x;
    }
};
*/

template <typename ValueType,
  class _HashFcn = hash<ValueType>,
  class _EqualKey = equal_to<ValueType>,
  class _A = allocator <ValueType> >
class hash_set :
  public rw_hashset<ValueType, class _HashFcn, class _EqualKey, class _A> {
};

} // end HASH_NAMESPACE;
#endif

using HASH_NAMESPACE::hash_set;

// Include vector because ext/hash_set includes stl_vector.h and leaves
// out specializations like stl_bvector.h, causing link conflicts.
#include <vector>

#include "llvm/ADT/HashExtras.h"

#endif
