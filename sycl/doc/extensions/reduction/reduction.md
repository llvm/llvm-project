# SYCL(TM) Proposals: Reductions for ND-Range Parallelism

**IMPORTANT**: This specification is a draft.

**NOTE**: Khronos(R) is a registered trademark and SYCL(TM) is a trademark of the Khronos Group, Inc.

It is common for parallel kernels to produce a single output resulting from some combination of all inputs (e.g. the sum).  Writing efficient reductions is a complex task, depending on both device and runtime characteristics: whether the value is being reduced across all parallel workers or some subset of them; whether the reduction can be accelerated by a specific scope of memory (e.g. work-group local memory); and whether the data type and reduction operator can be implemented with fast hardware atomics.  Providing an abstraction for reductions in SYCL would greatly improve programmer productivity.

This proposal focuses on introducing reductions to the ND-range version of `parallel_for`, using syntax that is roughly aligned with OpenMP and C++ [for_loop](https://wg21.link/p0075).  Reductions within hierarchical parallelism kernels and hints describing desired properties of reductions are left to a future iteration of the proposal.

# Reduction Semantics

A reduction produces a single value by _combining_ multiple values in an unspecified order, using an operator that is both _associative_ and _commutative_ (e.g. addition).  Only the _final_ value resulting from a reduction is of interest to the programmer.

It should also be noted that reductions are not limited to scalar values: the behavior of reductions is well-defined for structs and even containers, given appropriate reduction operators.

# `reduction` Objects

```c++
template <class T, class BinaryOperation>
unspecified reduction(accessor<T>& var, BinaryOperation combiner);

template <class T, class BinaryOperation>
unspecified reduction(accessor<T>& var, const T& identity, BinaryOperation combiner);
```

The exact behavior of a reduction is specific to an implementation; the only interface exposed to the user is the pair of functions above, which construct an unspecified `reduction` object encapsulating the reduction variable, an optional operator identity and the reduction operator.  For user-defined binary operations, an implementation should issue a compile-time warning if an identity is not specified and this is known to negatively impact performance (e.g. as a result of the implementation choosing a different reduction algorithm).  For standard binary operations (e.g. `std::plus`) on arithmetic types, the implementation must determine the correct identity automatically in order to avoid performance penalties.

The dimensionality of the `accessor` passed to the `reduction` function specifies the dimensionality of the reduction variable: a 0-dimensional `accessor` represents a scalar reduction, and any other dimensionality represents an array reduction.  Specifying an array reduction of size N is functionally equivalent to specifying N independent scalar reductions.  The access mode of the accessor determines whether the reduction variable's original value is included in the reduction (i.e. for `access::mode::read_write` it is included, and for `access::mode::discard_write` it is not).  Multiple reductions aliasing the same output results in undefined behavior.

`T` must be trivially copyable, permitting an implementation to (optionally) use atomic operations to implement the reduction.  This restriction is aligned with `std::atomic<T>` and `std::atomic_ref<T>`.

# `reducer` Objects

```c++
template <class T, class BinaryOperation, /* implementation-defined */>
class reducer
{
    // forbid reducer objects from being copied
    reducer(const reducer<T,BinaryOperation>&) = delete;
    reducer<T,BinaryOperation>& operator(const reducer<T,BinaryOperation>&) = delete;

    // combine partial result with reducer
    void combine(const T& partial);

    // get identity of the associated reduction (if known)
    T identity() const;
};

// other operators should be made available for standard functors
template <typename T> auto& operator+=(reducer<T,std::plus<T>>&, const T&);
```

The `reducer` class is not user-constructible, and can only be constructed by an implementation given a `reduction` object.  The `combine` function uses the specified `BinaryOperation` to combine the `partial` result with the value held (or referenced) by an instance of `reducer`, and is the only way to update the reducer value for user-supplied combination functions.  Other convenience operators should be defined for standard combination functions (e.g. `+=` for `std::plus`).

To enable compile-time specialization of reduction algorithms, an implementation may define additional template arguments to the `reducer` class.  The `reducer` type for a given reduction can be inspected using `decltype(reduction(var, identity, combiner))::reducer_type`.

# Adding `reduction` Objects to `parallel_for`

```c++
template <typename KernelName, typename KernelType, int dimensions, typename... Rest>
void parallel_for(range<dimensions>numWorkItems, Rest&&... rest);
```

The `rest` parameter pack consists of 0 or more `reduction` objects followed by the kernel functor.  For each `reduction` object operating on values of type `T`, the kernel functor should take an additional parameter of type `reducer<T, BinaryOperation>&`.  For convenience and to avoid supplying the same information twice, it is expected that developers using C++14 will typically make use of `auto&` in place of specifying the reducer type.

The implementation must guarantee that it is safe for each concurrently executing work-item to call the `combine` function of a reducer in parallel.  An implementation is free to re-use reducer variables (e.g. across work-groups scheduled to the same compute unit) if it can guarantee that it is safe to do so.

The combination order of different reducers is unspecified, as are when and how the value of each reducer is combined with the original variable.  The value of the original variable at any point during execution of the kernel is undefined, and the final value is only visible after the kernel completes.

## Example
```c++
// Compute a dot-product by reducing all computed values using standard plus functor
queue.submit([&](handler& cgh)
{
    auto a = a_buf.get_access<access::mode::read>(cgh);
    auto b = b_buf.get_access<access::mode::read>(cgh);
    auto sum = accessor<int,0,access::mode::write,access::target::global_buffer>(sum_buf, cgh);
    cgh.parallel_for<class dot_product>(nd_range<1>{N, M}, reduction(sum, 0, plus<int>()), [=](nd_item<1> it, auto& sum)
    {
        int i = it.get_global_id(0);
        sum += (a[i] * b[i]);
    });
});
```

# Reductions using USM Pointers

Unlike a buffer, a [USM pointer](https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions/usm) does not carry information describing the extent of the memory it points to; there is no way to distinguish between a scalar in device memory and an array.  This proposal assumes that the majority of reductions are scalar, and that a pointer passed to a reduction should therefore always be interpreted as a reduction of a single element.  The user must explicitly request an array reduction by passing a `span` denoting the memory region to include in the reduction.

## Example

```c++
// Treat an input pointer as N independent reductions
int* out = static_cast<int*>(sycl_malloc<alloc::shared>(4 * sizeof(int)));
queue.submit([&](handler& cgh)
{
    cgh.parallel_for<class sum>(nd_range<1>{N, M}, reduction(span(out, 4), 0, plus<int>()), [=](nd_item<1> it, auto& out)
    {
        int i = it.get_global_id(0);
        int j = foo(i);
        out[j] += in[i];
    });
});
```

# Code Generation

The semantics of this proposal have been carefully chosen to permit implementation freedom for different devices.  Example mappings of the dot-product code above to several potential implementations are given below.  This is not intended as an exhaustive list of implementations, but serves to demonstrate the flexibility of the proposal and its mapping to different hardware.

## Hierarchical Reduction

The simplest way to implement this proposal is to use built-in work-group reductions, followed by an atomic update to global memory.

```c++
__kernel void dot_product(__global float* a, __global float* b, __global float* sum)
{
    // Separate reducer per work-item is initialized to the reduction's identity value
    int item_partial_sum = 0;

    // User-provided lambda function
    int i = get_global_id(0);
    item_partial_sum += a[i] * b[i];

    // Reducer values are combined within a work-group before atomically updating global value
    int wg_partial_sum = work_group_reduce_add(item_partial_sum);
    if (get_local_id(0) == 0)
    {
        atomic_add(sum, wg_partial_sum);
    }
}
```

## Direct Atomics

For devices with very fast hardware atomics, it may be sufficient to simply update the global value atomically from each work-item.

```c++
__kernel void dot_product(__global float* a, __global float* b, __global float* sum)
{
    // User-provided lambda function
    // Each work-item directly updates the global value using (fast) hardware atomics
    int i = get_global_id(0);
    atomic_add(sum, a[i] * b[i]);
}
```
