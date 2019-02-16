// -*- C++ -*-
//===-- parallel_impl.h ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PSTL_parallel_impl_H
#define __PSTL_parallel_impl_H

#include <atomic>
// This header defines the minimum set of parallel routines required to support Parallel STL,
// implemented on top of Intel(R) Threading Building Blocks (Intel(R) TBB) library

namespace __pstl
{
namespace internal
{

//------------------------------------------------------------------------
// parallel_find
//-----------------------------------------------------------------------
/** Return extremum value returned by brick f[i,j) for subranges [i,j) of [first,last)
Each f[i,j) must return a value in [i,j). */
template <class _ExecutionPolicy, class _Index, class _Brick, class _Compare>
_Index
parallel_find(_ExecutionPolicy&& __exec, _Index __first, _Index __last, _Brick __f, _Compare __comp, bool __b_first)
{
    typedef typename std::iterator_traits<_Index>::difference_type _DifferenceType;
    const _DifferenceType __n = __last - __first;
    _DifferenceType __initial_dist = __b_first ? __n : -1;
    std::atomic<_DifferenceType> __extremum(__initial_dist);
    // TODO: find out what is better here: parallel_for or parallel_reduce
    par_backend::parallel_for(std::forward<_ExecutionPolicy>(__exec), __first, __last,
                              [__comp, __f, __first, &__extremum](_Index __i, _Index __j) {
                                  // See "Reducing Contention Through Priority Updates", PPoPP '13, for discussion of
                                  // why using a shared variable scales fairly well in this situation.
                                  if (__comp(__i - __first, __extremum))
                                  {
                                      _Index __res = __f(__i, __j);
                                      // If not '__last' returned then we found what we want so put this to extremum
                                      if (__res != __j)
                                      {
                                          const _DifferenceType __k = __res - __first;
                                          for (_DifferenceType __old = __extremum; __comp(__k, __old);
                                               __old = __extremum)
                                          {
                                              __extremum.compare_exchange_weak(__old, __k);
                                          }
                                      }
                                  }
                              });
    return __extremum != __initial_dist ? __first + __extremum : __last;
}

//------------------------------------------------------------------------
// parallel_or
//------------------------------------------------------------------------
//! Return true if brick f[i,j) returns true for some subrange [i,j) of [first,last)
template <class _ExecutionPolicy, class _Index, class _Brick>
bool
parallel_or(_ExecutionPolicy&& __exec, _Index __first, _Index __last, _Brick __f)
{
    std::atomic<bool> __found(false);
    par_backend::parallel_for(std::forward<_ExecutionPolicy>(__exec), __first, __last,
                              [__f, &__found](_Index __i, _Index __j) {
                                  if (!__found.load(std::memory_order_relaxed) && __f(__i, __j))
                                  {
                                      __found.store(true, std::memory_order_relaxed);
                                      par_backend::cancel_execution();
                                  }
                              });
    return __found;
}

} // namespace internal
} // namespace __pstl

#endif /* __PSTL_parallel_impl_H */
