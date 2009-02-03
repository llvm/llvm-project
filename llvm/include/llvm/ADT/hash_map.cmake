//===-- llvm/ADT/hash_map - "Portable" wrapper around hash_map --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a wrapper around the mysterious <hash_map> header file
// that seems to move around between GCC releases into and out of namespaces at
// will.  #including this header will cause hash_map to be available in the
// global namespace.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_HASH_MAP
#define LLVM_ADT_HASH_MAP

// Compiler Support Matrix
//
// Version   Namespace   Header File
//  2.95.x       ::        hash_map
//  3.0.4       std      ext/hash_map
//  3.1      __gnu_cxx   ext/hash_map
//  HP aCC6     std      stdex/rw/hashm*ap.h
//  MS VC++    stdext      hash_map

#cmakedefine HAVE_GNU_EXT_HASH_MAP
#cmakedefine HAVE_STD_EXT_HASH_MAP
#cmakedefine HAVE_GLOBAL_HASH_MAP
#cmakedefine HAVE_RW_STDEX_HASH_MAP_H

#if defined(HAVE_GNU_EXT_HASH_MAP)
// This is for GCC-3.1+ which puts hash in ext/hash_map
# include <ext/hash_map>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE __gnu_cxx
# endif

// GCC 3.0.x puts hash_map in <ext/hash_map> and in the std namespace.
#elif defined(HAVE_STD_EXT_HASH_MAP)
# include <ext/hash_map>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE std
# endif

// Older compilers such as GCC before version 3.0 do not keep
// extensions in the `ext' directory, and ignore the `std' namespace.
#elif defined(HAVE_GLOBAL_HASH_MAP)
# include <hash_map>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE std
# endif

// HP aCC doesn't include an SGI-like hash_map. For this platform (or
// any others using Rogue Wave Software's Tools.h++ library), we wrap
// around them in std::
#elif defined(HAVE_RW_STDEX_HASH_MAP_H)
# include <rw/stdex/hashmap.h>
# include <rw/stdex/hashmmap.h>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE std
# endif

// Support Microsoft VC++.
#elif defined(_MSC_VER)
# include <hash_map>
# ifndef HASH_NAMESPACE
#  define HASH_NAMESPACE stdext
   using std::_Distance;
# endif

// Give a warning if we couldn't find it, instead of (or in addition to)
// randomly doing something dumb.
#else
# warning "Autoconfiguration failed to find the hash_map header file."
#endif

// we wrap Rogue Wave Tools.h++ rw_hashmap into something SGI-looking, here:
#ifdef HAVE_RW_STDEX_HASH_MAP_H
namespace HASH_NAMESPACE {

template <class DataType> struct hash {
  unsigned int operator()(const unsigned int& x) const {
      return x;
  }
};

template <typename KeyType,
	  typename ValueType,
	  class _HashFcn = hash<KeyType>,
	  class _EqualKey = equal_to<KeyType>,
	  class _A = allocator <ValueType> >
class hash_map : public rw_hashmap<KeyType, ValueType, class _HashFcn,
				   class _EqualKey, class _A> {
};

template <typename KeyType,
	  typename ValueType,
	  class _HashFcn = hash<KeyType>,
	  class _EqualKey = equal_to<KeyType>,
	  class _A = allocator <ValueType> >
class hash_multimap : public rw_hashmultimap<KeyType, ValueType, class _HashFcn,
					     class _EqualKey, class _A> {
};

} // end HASH_NAMESPACE;
#endif

// Include vector because ext/hash_map includes stl_vector.h and leaves
// out specializations like stl_bvector.h, causing link conflicts.
#include <vector>

#ifdef _MSC_VER

// GCC and VC++ have differing ways of implementing hash_maps.  As it's not
// standardized, that's to be expected.  This adapter class allows VC++
// hash_map to use GCC's hash classes.
namespace stdext {
  template<class Key> struct hash;

  // Provide a hash function for unsigned ints...
  template<> struct hash<unsigned int> {
    inline size_t operator()(unsigned int Val) const {
      return Val;
    }
  };

  template<class Key> class hash_compare<Key, std::less<Key> > {
    std::less<Key> comp;
  public:
    enum { bucket_size = 4 };
    enum { min_buckets = 8 };
    hash_compare() {}
    hash_compare(std::less<Key> pred) : comp(pred) {}
    size_t operator()(const Key& key) const { return hash<Key>()(key); }
    bool operator()(const Key& k1, const Key& k2) const { return comp(k1, k2); }
  };
}

#endif

using HASH_NAMESPACE::hash_map;
using HASH_NAMESPACE::hash_multimap;
using HASH_NAMESPACE::hash;

#include "llvm/ADT/HashExtras.h"

#endif
