//==-- llvm/ADT/iterator.h - Portable wrapper around <iterator> --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a wrapper around the mysterious <iterator> header file.
// In GCC 2.95.3, the file defines a bidirectional_iterator class (and other
// friends), instead of the standard iterator class.  In GCC 3.1, the
// bidirectional_iterator class got moved out and the new, standards compliant,
// iterator<> class was added.  Because there is nothing that we can do to get
// correct behavior on both compilers, we have this header with #ifdef's.  Gross
// huh?
//
// By #includ'ing this file, you get the contents of <iterator> plus the
// following classes in the global namespace:
//
//   1. bidirectional_iterator
//   2. forward_iterator
//
// The #if directives' expressions are filled in by Autoconf.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ITERATOR_H
#define LLVM_ADT_ITERATOR_H

#include <iterator>

#undef HAVE_BI_ITERATOR
#undef HAVE_STD_ITERATOR
#undef HAVE_FWD_ITERATOR

#ifdef _MSC_VER
#  define HAVE_BI_ITERATOR 0
#  define HAVE_STD_ITERATOR 1
#  define HAVE_FWD_ITERATOR 0
#endif

#if !HAVE_BI_ITERATOR
# if HAVE_STD_ITERATOR
/// If the bidirectional iterator is not defined, we attempt to define it in
/// terms of the C++ standard iterator. Otherwise, we import it with a "using"
/// statement.
///
template<class Ty, class PtrDiffTy>
struct bidirectional_iterator
  : public std::iterator<std::bidirectional_iterator_tag, Ty, PtrDiffTy> {
};
# else
#  error "Need to have standard iterator to define bidirectional iterator!"
# endif
#else
using std::bidirectional_iterator;
#endif

#if !HAVE_FWD_ITERATOR
# if HAVE_STD_ITERATOR
/// If the forward iterator is not defined, attempt to define it in terms of
/// the C++ standard iterator. Otherwise, we import it with a "using" statement.
///
template<class Ty, class PtrDiffTy>
struct forward_iterator
  : public std::iterator<std::forward_iterator_tag, Ty, PtrDiffTy> {
};
# else
#  error "Need to have standard iterator to define forward iterator!"
# endif
#else
using std::forward_iterator;
#endif

#endif // LLVM_ADT_ITERATOR_H
