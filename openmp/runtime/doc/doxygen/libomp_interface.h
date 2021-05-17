// clang-format off
// This file does not contain any code; it just contains additional text and formatting
// for doxygen.


//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


/*! @mainpage LLVM&nbsp; OpenMP* Runtime Library Interface
@section sec_intro Introduction

This document describes the interface provided by the
LLVM &nbsp;OpenMP\other runtime library to the compiler.
Routines that are directly called as simple functions by user code are
not currently described here, since their definition is in the OpenMP
specification available from http://openmp.org

The aim here is to explain the interface from the compiler to the runtime.

The overall design is described, and each function in the interface
has its own description. (At least, that's the ambition, we may not be there yet).

@section sec_building Quickly Building the Runtime
For the impatient, we cover building the runtime as the first topic here.

CMake is used to build the OpenMP runtime.  For details and a full list of options for the CMake build system,
see <tt>README.rst</tt> in the source code repository.  These instructions will provide the most typical build.

In-LLVM-tree build:.
@code
$ cd where-you-want-to-live
Check out openmp into llvm/projects
$ cd where-you-want-to-build
$ mkdir build && cd build
$ cmake path/to/llvm -DCMAKE_C_COMPILER=<C compiler> -DCMAKE_CXX_COMPILER=<C++ compiler>
$ make omp
@endcode
Out-of-LLVM-tree build:
@code
$ cd where-you-want-to-live
Check out openmp
$ cd where-you-want-to-live/openmp
$ mkdir build && cd build
$ cmake path/to/openmp -DCMAKE_C_COMPILER=<C compiler> -DCMAKE_CXX_COMPILER=<C++ compiler>
$ make
@endcode

@section sec_supported Supported RTL Build Configurations

The architectures supported are IA-32 architecture, Intel&reg;&nbsp; 64, and
Intel&reg;&nbsp; Many Integrated Core Architecture.  The build configurations
supported are shown in the table below.

<table border=1>
<tr><th> <th>icc/icl<th>gcc<th>clang
<tr><td>Linux\other OS<td>Yes(1,5)<td>Yes(2,4)<td>Yes(4,6,7)
<tr><td>FreeBSD\other<td>Yes(1,5)<td>Yes(2,4)<td>Yes(4,6,7,8)
<tr><td>OS X\other<td>Yes(1,3,4)<td>No<td>Yes(4,6,7)
<tr><td>Windows\other OS<td>Yes(1,4)<td>No<td>No
</table>
(1) On IA-32 architecture and Intel&reg;&nbsp; 64, icc/icl versions 12.x
    are supported (12.1 is recommended).<br>
(2) gcc version 4.7 is supported.<br>
(3) For icc on OS X\other, OS X\other version 10.5.8 is supported.<br>
(4) Intel&reg;&nbsp; Many Integrated Core Architecture not supported.<br>
(5) On Intel&reg;&nbsp; Many Integrated Core Architecture, icc/icl versions 13.0 or later are required.<br>
(6) Clang\other version 3.3 is supported.<br>
(7) Clang\other currently does not offer a software-implemented 128 bit extended
    precision type.  Thus, all entry points reliant on this type are removed
    from the library and cannot be called in the user program.  The following
    functions are not available:
@code
    __kmpc_atomic_cmplx16_*
    __kmpc_atomic_float16_*
    __kmpc_atomic_*_fp
@endcode
(8) Community contribution provided AS IS, not tested by Intel.

Supported Architectures: IBM(R) Power 7 and Power 8
<table border=1>
<tr><th> <th>gcc<th>clang
<tr><td>Linux\other OS<td>Yes(1,2)<td>Yes(3,4)
</table>
(1) On Power 7, gcc version 4.8.2 is supported.<br>
(2) On Power 8, gcc version 4.8.2 is supported.<br>
(3) On Power 7, clang version 3.7 is supported.<br>
(4) On Power 8, clang version 3.7 is supported.<br>

@section sec_frontend Front-end Compilers that work with this RTL

The following compilers are known to do compatible code generation for
this RTL: icc/icl, gcc.  Code generation is discussed in more detail
later in this document.

@section sec_outlining Outlining

The runtime interface is based on the idea that the compiler
"outlines" sections of code that are to run in parallel into separate
functions that can then be invoked in multiple threads.  For instance,
simple code like this

@code
void foo()
{
#pragma omp parallel
    {
        ... do something ...
    }
}
@endcode
is converted into something that looks conceptually like this (where
the names used are merely illustrative; the real library function
names will be used later after we've discussed some more issues...)

@code
static void outlinedFooBody()
{
    ... do something ...
}

void foo()
{
    __OMP_runtime_fork(outlinedFooBody, (void*)0);   // Not the real function name!
}
@endcode

@subsection SEC_SHAREDVARS Addressing shared variables

In real uses of the OpenMP\other API there are normally references
from the outlined code  to shared variables that are in scope in the containing function.
Therefore the containing function must be able to address
these variables. The runtime supports two alternate ways of doing
this.

@subsubsection SEC_SEC_OT Current Technique
The technique currently supported by the runtime library is to receive
a separate pointer to each shared variable that can be accessed from
the outlined function.  This is what is shown in the example below.

We hope soon to provide an alternative interface to support the
alternate implementation described in the next section. The
alternative implementation has performance advantages for small
parallel regions that have many shared variables.

@subsubsection SEC_SEC_PT Future Technique
The idea is to treat the outlined function as though it
were a lexically nested function, and pass it a single argument which
is the pointer to the parent's stack frame. Provided that the compiler
knows the layout of the parent frame when it is generating the outlined
function it can then access the up-level variables at appropriate
offsets from the parent frame.  This is a classical compiler technique
from the 1960s to support languages like Algol (and its descendants)
that support lexically nested functions.

The main benefit of this technique is that there is no code required
at the fork point to marshal the arguments to the outlined function.
Since the runtime knows statically how many arguments must be passed to the
outlined function, it can easily copy them to the thread's stack
frame.  Therefore the performance of the fork code is independent of
the number of shared variables that are accessed by the outlined
function.

If it is hard to determine the stack layout of the parent while generating the
outlined code, it is still possible to use this approach by collecting all of
the variables in the parent that are accessed from outlined functions into
a single `struct` which is placed on the stack, and whose address is passed
to the outlined functions. In this way the offsets of the shared variables
are known (since they are inside the struct) without needing to know
the complete layout of the parent stack-frame. From the point of view
of the runtime either of these techniques is equivalent, since in either
case it only has to pass a single argument to the outlined function to allow
it to access shared variables.

A scheme like this is how gcc\other generates outlined functions.

@section SEC_INTERFACES Library Interfaces
The library functions used for specific parts of the OpenMP\other language implementation
are documented in different modules.

 - @ref BASIC_TYPES fundamental types used by the runtime in many places
 - @ref DEPRECATED  functions that are in the library but are no longer required
 - @ref STARTUP_SHUTDOWN functions for initializing and finalizing the runtime
 - @ref PARALLEL functions for implementing `omp parallel`
 - @ref THREAD_STATES functions for supporting thread state inquiries
 - @ref WORK_SHARING functions for work sharing constructs such as `omp for`, `omp sections`
 - @ref THREADPRIVATE functions to support thread private data, copyin etc
 - @ref SYNCHRONIZATION functions to support `omp critical`, `omp barrier`, `omp master`, reductions etc
 - @ref ATOMIC_OPS functions to support atomic operations
 - @ref STATS_GATHERING macros to support developer profiling of libomp
 - Documentation on tasking has still to be written...

@section SEC_EXAMPLES Examples
@subsection SEC_WORKSHARING_EXAMPLE Work Sharing Example
This example shows the code generated for a parallel for with reduction and dynamic scheduling.

@code
extern float foo( void );

int main () {
    int i;
    float r = 0.0;
    #pragma omp parallel for schedule(dynamic) reduction(+:r)
    for ( i = 0; i < 10; i ++ ) {
        r += foo();
    }
}
@endcode

The transformed code looks like this.
@code
extern float foo( void );

int main () {
    static int zero = 0;
    auto int gtid;
    auto float r = 0.0;
    __kmpc_begin( & loc3, 0 );
    // The gtid is not actually required in this example so could be omitted;
    // We show its initialization here because it is often required for calls into
    // the runtime and should be locally cached like this.
    gtid = __kmpc_global thread num( & loc3 );
    __kmpc_fork call( & loc7, 1, main_7_parallel_3, & r );
    __kmpc_end( & loc0 );
    return 0;
}

struct main_10_reduction_t_5 { float r_10_rpr; };

static kmp_critical_name lck = { 0 };
static ident_t loc10; // loc10.flags should contain KMP_IDENT_ATOMIC_REDUCE bit set
                      // if compiler has generated an atomic reduction.

void main_7_parallel_3( int *gtid, int *btid, float *r_7_shp ) {
    auto int i_7_pr;
    auto int lower, upper, liter, incr;
    auto struct main_10_reduction_t_5 reduce;
    reduce.r_10_rpr = 0.F;
    liter = 0;
    __kmpc_dispatch_init_4( & loc7,*gtid, 35, 0, 9, 1, 1 );
    while ( __kmpc_dispatch_next_4( & loc7, *gtid, & liter, & lower, & upper, & incr ) ) {
        for( i_7_pr = lower; upper >= i_7_pr; i_7_pr ++ )
          reduce.r_10_rpr += foo();
    }
    switch( __kmpc_reduce_nowait( & loc10, *gtid, 1, 4, & reduce, main_10_reduce_5, & lck ) ) {
        case 1:
           *r_7_shp += reduce.r_10_rpr;
           __kmpc_end_reduce_nowait( & loc10, *gtid, & lck );
           break;
        case 2:
           __kmpc_atomic_float4_add( & loc10, *gtid, r_7_shp, reduce.r_10_rpr );
           break;
        default:;
    }
}

void main_10_reduce_5( struct main_10_reduction_t_5 *reduce_lhs,
                       struct main_10_reduction_t_5 *reduce_rhs )
{
    reduce_lhs->r_10_rpr += reduce_rhs->r_10_rpr;
}
@endcode

@defgroup BASIC_TYPES Basic Types
Types that are used throughout the runtime.

@defgroup DEPRECATED Deprecated Functions
Functions in this group are for backwards compatibility only, and
should not be used in new code.

@defgroup STARTUP_SHUTDOWN Startup and Shutdown
These functions are for library initialization and shutdown.

@defgroup PARALLEL Parallel (fork/join)
These functions are used for implementing <tt>\#pragma omp parallel</tt>.

@defgroup THREAD_STATES Thread Information
These functions return information about the currently executing thread.

@defgroup WORK_SHARING Work Sharing
These functions are used for implementing
<tt>\#pragma omp for</tt>, <tt>\#pragma omp sections</tt>, <tt>\#pragma omp single</tt> and
<tt>\#pragma omp master</tt> constructs.

When handling loops, there are different functions for each of the signed and unsigned 32 and 64 bit integer types
which have the name suffixes `_4`, `_4u`, `_8` and `_8u`. The semantics of each of the functions is the same,
so they are only described once.

Static loop scheduling is handled by  @ref __kmpc_for_static_init_4 and friends. Only a single call is needed,
since the iterations to be executed by any give thread can be determined as soon as the loop parameters are known.

Dynamic scheduling is handled by the @ref __kmpc_dispatch_init_4 and @ref __kmpc_dispatch_next_4 functions.
The init function is called once in each thread outside the loop, while the next function is called each
time that the previous chunk of work has been exhausted.

@defgroup SYNCHRONIZATION Synchronization
These functions are used for implementing barriers.

@defgroup THREADPRIVATE Thread private data support
These functions support copyin/out and thread private data.

@defgroup STATS_GATHERING Statistics Gathering from OMPTB
These macros support profiling the libomp library.  Use --stats=on when building with build.pl to enable
and then use the KMP_* macros to profile (through counts or clock ticks) libomp during execution of an OpenMP program.

@section sec_stats_env_vars Environment Variables

This section describes the environment variables relevant to stats-gathering in libomp

@code
KMP_STATS_FILE
@endcode
This environment variable is set to an output filename that will be appended *NOT OVERWRITTEN* if it exists.  If this environment variable is undefined, the statistics will be output to stderr

@code
KMP_STATS_THREADS
@endcode
This environment variable indicates to print thread-specific statistics as well as aggregate statistics.  Each thread's statistics will be shown as well as the collective sum of all threads.  The values "true", "on", "1", "yes" will all indicate to print per thread statistics.

@defgroup TASKING Tasking support
These functions support tasking constructs.

@defgroup USER User visible functions
These functions can be called directly by the user, but are runtime library specific, rather than being OpenMP interfaces.

*/

