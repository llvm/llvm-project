# Parallel STL 

Parallel STL is an implementation of the C++ standard library algorithms with support for execution policies,
as specified in ISO/IEC 14882:2017 standard, commonly called C++17. The implementation also supports the unsequenced 
execution policy specified in Parallelism TS version 2 and proposed for the next version of the C++ standard in the 
C++ working group paper [P1001](https://wg21.link/p1001).
Parallel STL offers efficient support for both parallel and vectorized execution of algorithms. For sequential 
execution, it relies on an available implementation of the C++ standard library.

## Prerequisites

To use Parallel STL, you must have the following software installed:
* C++ compiler with:
  * Support for C++11
  * Support for OpenMP* 4.0 SIMD constructs
* Threading Building Blocks (TBB) which is available for download at https://github.com/01org/tbb/

## Known issues and limitations

* `unseq` and `par_unseq` policies only have effect with compilers that support `#pragma omp simd` or `#pragma simd`.
* Parallel and vector execution is only supported for the algorithms if random access iterators are provided,
  while for other iterator types the execution will remain serial.
* The following algorithms do not allow efficient SIMD execution: `includes`, `inplace_merge`, `merge`, `nth_element`,
  `partial_sort`, `partial_sort_copy`, `set_difference`, `set_intersection`, `set_symmetric_difference`, `set_union`,
  `sort`, `stable_partition`, `stable_sort`, `unique`.
* The initial value type for `exclusive_scan`, `inclusive_scan`, `transform_exclusive_scan`, `transform_inclusive_scan`
  shall be DefaultConstructible. A default constructed-instance of the initial value type shall be the identity element
  for the specified binary operation.
* For `max_element`, `min_element`, `minmax_element`, `partial_sort`, `partial_sort_copy`, `sort`, `stable_sort`
  the dereferenced value type of the provided iterators shall be DefaultConstructible.
* For `remove`, `remove_if`, `unique` the dereferenced value type of the provided iterators shall be MoveConstructible.
* The following algorithms require additional O(n) memory space for parallel execution: `copy_if`, `inplace_merge`,
  `partial_sort`, `partial_sort_copy`, `partition_copy`, `remove`, `remove_if`, `rotate`, `sort`, `stable_sort`,
  `unique`, `unique_copy`.

