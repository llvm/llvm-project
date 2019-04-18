# SYCL(TM) Proposals: Sub-groups for NDRange Parallelism

**IMPORTANT**: This specification is a draft.

**NOTE**: Khronos(R) is a registered trademark and SYCL(TM) is a trademark of the Khronos Group, Inc.

A _sub-group_ represents an implementation-defined grouping of work-items in a work-group. The work-items within a sub-group can communicate and synchronize independently of work-items in other sub-groups, and sub-groups are therefore commonly mapped to SIMD hardware where it exists.

Sub-groups have been part of the OpenCL execution model since OpenCL 2.0, but many important functions are missing: several hardware features are exposed only as vendor-specific extensions, and functions common in other programming models are not exposed at all.  This proposal defines SYCL syntax and semantics for the core OpenCL functionality, but also seeks to expose some of these missing functions.

The first version of this document is focused on exposing sub-group functionality to the NDRange form of SYCL `parallel_for`, and does not address hierarchical parallelism.

## Alignment with OpenCL vs C++

Where a feature is common to both OpenCL and C++, this proposal opts for C++-like naming:
- Collective operators are named as in `<functional>` (e.g. `plus` instead of `sum`) and to avoid clashes with names in `<algorithm>` (e.g. `minimum` instead of `min`).
- Scan operations are named as in `<algorithm>` (e.g. `inclusive_scan` instead of `scan_inclusive`).

## Towards a Generic Group Abstraction

Providing a generic group abstraction encapsulating the shared functionality of all synchronizable SYCL groups (i.e. work-groups and sub-groups) in a single interface would enable users to write more general code and simplify the introduction of additional SYCL groups in the future (e.g. device-wide synchronization groups).  Some names in this proposal are chosen to demonstrate how this may look:
- The common interface members of `sub_group` do not reference sub-groups by name, opting instead for generic names like `get_group_range()`.
- `get_enqueued_num_sub_groups()` is exposed as `get_uniform_group_range()`, since future generic groups may not be 'enqueued' but may still be non-uniform.
- `barrier()` is exposed as a member of the `sub_group` class rather than as a member of the `nd_item` class.

## Data Types

Many aspects of sub-group behavior are implementation-defined and/or device-specific.  In order to maximize the portability of code written to utilize the sub-group class, all functions are supported for the fundamental standard scalar types supported by SYCL (see Section 6.5 of the SYCL 1.2.1 specification): `bool`, `char`, `signed char`, `unsigned char`, `short int`, `unsigned short int`, `int`, `unsigned int`, `long int`, `unsigned long int`, `long long int`, `unsigned long long int`, `size_t`, `float`, `double`, `half`.

# Attributes

In keeping with Section 6.7 of the SYCL 1.2.1 specification, attributes are made available as a C++11 attribute specifier in the `cl` namespace, and the attributes of a kernel are the sum of all the kernel attributes of all device functions called.  Attribute names are prefixed with `intel` to denote that they are Intel extensions.

## Required Sub-group Size

The `[[cl::intel_reqd_sub_group_size(n)]]` attribute indicates that the kernel must be compiled and executed with a sub-group of size `n`.  The value of `n` must be set to a sub-group size supported by the device, or device compilation will fail.

In addition to device functions, the required sub-group size attribute may also be specified in the definition of a named functor object, as in the example below:

```c++
class Functor
{
    void operator()(item<1> item) [[cl::intel_reqd_sub_group_size(16)]]
    {
        /* kernel code */
    }
}
```

# Sub-group Queries

Under the OpenCL execution model (see Section 3.2.2 of the OpenCL 2.2 specification), several aspects of sub-group functionality are implementation-defined: the size and number of sub-groups is implementation-defined (and may differ for each kernel); and different devices may make different guarantees with respect to how sub-groups within a work-group are scheduled.  Developers can query these behaviors at a device level and for individual kernels.

To maximize portability across devices, developers should not assume that work-items within a sub-group execute in lockstep, nor that two sub-groups within a work-group will make independent forward progress with respect to one another.

|Device descriptors|Return type|Description|
|------------------|-----------|-----------|
| `info::device::max_num_sub_groups` | `cl_uint` | Returns the maximum number of sub-groups in a work-group for any kernel executed on the device.  The minimum value is 1. |
| `info::device::sub_group_independent_forward_progress` | `bool` | Returns `true` if the device supports independent forward progress of sub-groups with respect to other sub-groups in the same work-group. |
| `info::device::sub_group_sizes` | `vector_class<size_t>` | Returns a vector_class of `size_t` containing the set of sub-group sizes supported by the device. |

|Member functions|Description|
|----------------|-----------|
| `template <info::kernel_sub_group param>typename info::param_traits<info::kernel_sub_group, param>::return_type get_sub_group_info(const device &dev) const` | Query information from the sub-group from a kernel using the `info::kernel_sub_group` descriptor for a specific device. |
| `template <info::kernel_sub_group param>typename info::param_traits<info::kernel_sub_group, param>::return_type get_sub_group_info(const device &dev, typename info::param_traits<info::kernel_sub_group, param>::input_type value) const` | Query information from the sub-group from a kernel using the `info::kernel_sub_group` descriptor for a specific device and input parameter. The expected value of the input parameter depends on the information being queried. |

|Kernel descriptors|Input type|Return type|Description|
|------------------|----------|-----------|-----------|
| `info::kernel_sub_group::max_sub_group_size_for_ndrange` | `range<D>` | `uint32_t` | Returns the maximum sub-group size for the specified work-group size. |
| `info::kernel_sub_group::sub_group_count_for_ndrange` | `range<D>` | `uint32_t` | Returns the number of sub-groups for the specified work-group size. |
| `info::kernel_sub_group::local_size_for_sub_group_count` | `size_t` | `range<D>` | Returns a work-group size that will contain the specified number of sub-groups. |
| `info::kernel_sub_group::max_num_sub_groups` | N/A | `uint32_t` | Returns the maximum number of sub-groups for this kernel. |
| `info::kernel_sub_group::compile_num_sub_groups` | N/A | `uint32_t` | Returns the number of sub-groups specified by the kernel, or 0 (if not specified). |
| `info::kernel_sub_group::compile_sub_group_size` | N/A | `size_t` | Returns the required sub-group size specified by the kernel, or 0 (if not specified). |

# Using Subgroups within NDRange Kernels

The `sub_group` class encapsulates all functionality required to represent a particular sub-group within a parallel execution.  It is not user-constructable, and can only be accessed via the `nd_item` class.

|Member functions|Description|
|----------------|-----------|
| `sub_group get_sub_group() const` | Return the sub-group to which the work-item belongs. |

An example usage of the `sub_group` class is given below.

 ```c++
parallel_for<class kernel>(..., [&](nd_item item)
{
  sub_group sg = item.get_sub_group();
  for (int v = sg.get_local_id(); v < N; v += sg.get_local_range())
  {
    ...
  }
});
 ```

# Sub-group Functions

With the exception of the common interface members, all member functions of the `sub_group` class are _sub-group functions_.  Sub-group functions synchronize all work-items in a sub-group (i.e. they act as sub-group barriers) and must therefore be encountered within converged control flow across all work-items in the sub-group.  All the work-items of a sub-group must execute the sub-group function before any are allowed to continue execution beyond the sub-group function.

Each sub-group function applies only to the work-items within a single sub-group; communication between multiple sub-groups requires the use of work-group functions, or reads/writes from/to memory with appropriate work-group barriers and/or memory fences.

The sub-group functions in this proposal have been identified as a core set of functions that should ideally be supported by all implementations and have a clear mapping to all devices.  The vast majority of these functions have an equivalent in other specifications (e.g. OpenCL, SPIR), and the semantics defined here are intended to be compatible.  Additional, highly specialized, sub-group functions should be relegated to vendor- or device-specific extensions.

## Core Functionality

### Common Member Functions

The common member functions provide a mechanism for a developer to query properties of a sub-group and a work-item's position in it.

|Member functions|Description|
|----------------|-----------|
| `id<1> get_local_id() const` | Return an id representing the index of the work-item within the sub-group. |
| `range<1> get_local_range() const` | Return a SYCL range representing the number of work-items in the sub-group. |
| `range<1> get_max_local_range() const` | Return a SYCL range representing the maximum number of work-items in any sub-group within the nd-range. |
| `id<1> get_group_id() const` | Return an id representing the index of the sub-group within the work-group. |
| `uint32_t get_group_range() const` | Return the number of sub-groups within the work-group. |
| `uint32_t get_uniform_group_range() const` | Return the number of sub-groups per work-group in the uniform region of the nd-range. |

### Synchronization Functions

A sub-group barrier synchronizes all work-items in a sub-group, and orders memory operations to the specified address space(s).  On hardware where sub-groups are executed in SIMD, a sub-group barrier is expected to be a no-op.

|Member functions|Description|
|----------------|-----------|
| `void barrier(access::fence_space accessSpace = access::fence_space::global_and_local) const;` | Execute a sub-group barrier with an optional memory fence specified by `accessSpace`. |

### Vote / Ballot

The vote / ballot sub-group functions communicate Boolean conditions between the work-items in a sub-group, and enable developers to direct control flow at the sub-group level: a work-item may take a branch if _any_ work-item in its sub-group would do so; or may exit a loop only once _all_ work-items in its sub-group have finished.

|Member functions|Description|
|----------------|-----------|
| `bool any(bool predicate)` | Return `true` if `predicate` evaluates to `true` for any work-item in the sub-group. |
| `bool all(bool predicate)` | Return `true` if `predicate` evaluates to `true` for all work-items in the sub-group. |

### Collectives

The collective sub-group functions perform communications that involve all work-items in a sub-group, providing several common communication patterns: sharing a single value across the sub-group via a _broadcast_; combining all values from the sub-group into one value via a _reduction_; or performing a _scan_ across all values in the sub-group.

The `plus`, `minimum` and `maximum` functors in the `cl::sycl` namespace correspond to the collective operations supported by OpenCL 2.0.  Supporting other operations (e.g. `minus` and `multiplies`) and user-defined functors may be of interest in the future.

|Member functions|Description|
|----------------|-----------|
| `template <typename T>T broadcast(T x, id<1> local_id)` | Broadcast the value of `x` from the work-item with the specified id to all work-items within the sub-group. The value of `local_id` must be the same for all work-items in the sub-group. |
| `template <typename T, class BinaryOp>T reduce(T x, T init, BinaryOp binary_op)` | Combine the values of `x` from all work-items in the sub-group using the specified operator, which must be one of: `plus`, `minimum` or `maximum`. |
| `template <typename T, class BinaryOp>T exclusive_scan(T x, T init, BinaryOp binary_op)` | Perform an exclusive scan over the values of `x` from all work-items in the sub-group using the specified operator, which must be one of: `plus`, `minimum` or `maximum`. The value returned on work-item `i` is the exclusive scan of the first `i` work-items in the sub-group. |
| `template <typename T, class BinaryOp>T inclusive_scan(T x, BinaryOp binary_op, T init)` | Perform an inclusive scan over the values of `x` from all work-items in the sub-group using the specified operator, which must be one of: `plus`, `minimum` or `maximum`. The value returned on work-item `i` is the inclusive scan of the first `i` work-items in the sub-group. |

## Extended Functionality

### Shuffles

The shuffle sub-group functions perform arbitrary communication between pairs of work-items in a sub-group.  Common patterns -- such as shifting all values in a sub-group by a fixed number of work-items -- are exposed as specialized shuffles that may be accelerated in hardware.

|Member functions|Description|
|----------------|-----------|
| `template <typename T>T shuffle(T x, id<1> local_id)` | Exchange values of `x` between work-items in the sub-group in an arbitrary pattern.  Returns the value of `x` from the work-item with the specified id.  The value of `local_id` must be between 0 and the sub-group size. |
| `template <typename T>T shuffle_down(T x, uint32_t delta)` | Exchange values of `x` between work-items in the sub-group via a shift.  Returns the value of `x` from the work-item whose id is `delta` larger than the calling work-item. The value returned when the result of id + `delta` is greater than or equal to the sub-group size is undefined. |
| `template <typename T>T shuffle_up(T x, uint32_t delta)` | Exchange values of `x` between work-items in the sub-group via a shift.  Returns the value of `x` from the work-item whose id is `delta` smaller than the calling work-item.  The value of returned when the result of id - `delta` is less than zero is undefined. |
| `template <typename T>T shuffle_xor(T x, id<1> mask)` | Exchange pairs of values of `x` between work-items in the sub-group.  Returns the value of `x` from the work-item whose id is equal to the exclusive-or of the calling work-item's id and `mask`. `mask` must be a compile-time constant value that is the same for all work-items in the sub-group. |

### Two-Input Shuffles

This proposal makes a distinction between shuffles with one input per work-item and shuffles with two inputs per work-item.  The two-input versions map naturally to SIMD execution (see the `shuffle2` vector operation from OpenCL), and enable developers to avoid certain undefined behaviors from the one-input versions.  The simplest way to think of the two-input shuffles is that their operation is equivalent to a one-input shuffle on a virtual sub-group twice as big.

|Member functions|Description|
|----------------|-----------|
| `template <typename T>T shuffle(T x, T y, id<1> local_id)` | Exchange values of `x` and `y` between work-items in the sub-group in an arbitrary pattern.  If `local_id` is between 0 and the sub-group size, returns the value of `x` from the work-item with the specified id; if `local_id` is between the sub-group size and twice the sub-group size, returns the value of `y` from the work-item with the specified id (modulo the sub-group size).  The value of `local_id` must be between 0 and twice the sub-group size. |
| `template <typename T>T shuffle_down(T x, T y, uint32_t delta)` | Exchange values of `x` and `y` between work-items in the sub-group via a shift.  If the calling work-item's id + `delta` is between 0 and the sub-group size, returns the value of `x` from the work-item whose id is `delta` larger than the calling work-item; if the calling work-item's id + `delta` is between the sub-group size and twice the sub-group size, returns the value of `y` from the work-item with the specified id (modulo the sub-group size).  The value of `delta` must be less than the sub-group size. |
| `template <typename T>T shuffle_up(T x, T y, uint32_t delta)` | Exchange values of `x` and `y` between work-items in the sub-group via a shift.  If the calling work-item's id - `delta` is between 0 and the sub-group size, returns the value of `x` from the work-item whose id is `delta` smaller than the calling work-item; if the calling work-item's id - `delta` is between the sub-group size and twice the sub-group size, returns the value of `y` from the work-item with the specified id (modulo the sub-group size).  The value of `delta` must be less than the sub-group size. |

### Loads / Stores

The load and store sub-group functions enable developers to assert that all work-items in a sub-group read/write from/to contiguous locations in memory.  Such operations can be mapped directly to SIMD operations.

|Member functions|Description|
|----------------|-----------|
| `template <typename T, access::address_space Space>T load(const multi_ptr<T,Space> src)` | Load contiguous data from `src`.  Returns one element per work-item, corresponding to the memory location at `src` + `get_local_id()`. The value of `src` must be the same for all work-items in the sub-group. |
| `template <int N, typename T, access::address_space Space>vec<T,N> load(const multi_ptr<T,Space> src)` | Load contiguous data from `src`.  Returns `N` elements per work-item, corresponding to the `N` memory locations at `src` + `i` * `get_max_local_range()` + `get_local_id()` for `i` between 0 and `N`. The value of `src` must be the same for all work-items in the sub-group. |
| `template <typename T, access::address_space Space>void store(multi_ptr<T,Space> dst, const T& x)` | Store contiguous data to `dst`.  The value of `x` from each work-item is written to the memory location at `dst` + `get_local_id()`. The value of `dst` must be the same for all work-items in the sub-group. |
| `template <int N, typename T, access::address_space Space>void store(multi_ptr<T,Space> dst, const vec<T,N>& x)` | Store contiguous data to `dst`.  The `N` elements from each work-item are written to the memory locations at `dst` + `i` * `get_max_local_range()` + `get_local_id()` for `i` between 0 and `N`.  The value of `dst` must be the same for all work-items in the sub-group. |

# Sample Header

```c++
namespace cl {
namespace sycl {
namespace intel {
struct sub_group {

    /* --- common interface members --- */

    id<1> get_local_id() const;

    range<1> get_local_range() const;

    range<1> get_max_local_range();

    id<1> get_group_id() const;

    uint32_t get_group_range() const;

    uint32_t get_uniform_group_range() const;

    /* --- vote/ballot functions --- */

    bool any(bool predicate);

    bool all(bool predicate);

    /* --- data-sharing --- */

    template <typename T>
    T broadcast(T x, id<1> local_id);

    template <typename T, class BinaryOp>
    T reduce(T x, T init, BinaryOp binary_op);

    template <typename T, class BinaryOp>
    T exclusive_scan(T x, T init, BinaryOp binary_op);

    template <typename T, class BinaryOp>
    T inclusive_scan(T x, BinaryOp binary_op, T init);

    /* --- one-input shuffles --- */

    template <typename T>
    T shuffle(T x, id<1> local_id);

    template <typename T>
    T shuffle_down(T x, uint32_t delta);

    template <typename T>
    T shuffle_up(T x, uint32_t delta);

    template <typename T>
    T shuffle_xor(T x, id<1> value);

    /* --- two-input shuffles --- */

    template <typename T>
    T shuffle(T x, T y, id<1> local_id);

    template <typename T>
    T shuffle_down(T current, T next, uint32_t delta);

    template <typename T>
    T shuffle_up(T previous, T current, uint32_t delta);

    /* --- sub-group load/stores --- */

    template <typename T, access::address_space Space>
    T load(const multi_ptr<T,Space> src);

    template <typename T, int N, access::address_space Space>
    vec<T,N> load(const multi_ptr<T,Space> src);

    template <typename T, int N, access::address_space Space>
    void store(multi_ptr<T,Space> dst, const T& x);

    template <typename T, int N, access::address_space Space>
    void store(multi_ptr<T,Space> dst, const vec<T,N>& x);

};
} // intel
} // sycl
} // cl
```
