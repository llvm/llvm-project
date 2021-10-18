.. _openmp_runtimes:

LLVM/OpenMP Runtimes
====================

There are four distinct types of LLVM/OpenMP runtimes 

LLVM/OpenMP Host Runtime (``libomp``)
-------------------------------------

An `early (2015) design document <https://openmp.llvm.org/Reference.pdf>`_ for
the LLVM/OpenMP host runtime, aka.  `libomp.so`, is available as a `pdf
<https://openmp.llvm.org/Reference.pdf>`_.


LLVM/OpenMP Target Host Runtime (``libomptarget``)
--------------------------------------------------

.. _libopenmptarget_environment_vars:

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

``libomptarget`` uses environment variables to control different features of the
library at runtime. This allows the user to obtain useful runtime information as
well as enable or disable certain features. A full list of supported environment
variables is defined below.

    * ``LIBOMPTARGET_DEBUG=<Num>``
    * ``LIBOMPTARGET_PROFILE=<Filename>``
    * ``LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=<Num>``
    * ``LIBOMPTARGET_INFO=<Num>``
    * ``LIBOMPTARGET_HEAP_SIZE=<Num>``
    * ``LIBOMPTARGET_STACK_SIZE=<Num>``
    * ``LIBOMPTARGET_SHARED_MEMORY_SIZE=<Num>``

LIBOMPTARGET_DEBUG
""""""""""""""""""

``LIBOMPTARGET_DEBUG`` controls whether or not debugging information will be
displayed. This feature is only availible if ``libomptarget`` was built with
``-DOMPTARGET_DEBUG``. The debugging output provided is intended for use by
``libomptarget`` developers. More user-friendly output is presented when using
``LIBOMPTARGET_INFO``.

LIBOMPTARGET_PROFILE
""""""""""""""""""""
``LIBOMPTARGET_PROFILE`` allows ``libomptarget`` to generate time profile output
similar to Clang's ``-ftime-trace`` option. This generates a JSON file based on
`Chrome Tracing`_ that can be viewed with ``chrome://tracing`` or the
`Speedscope App`_. Building this feature depends on the `LLVM Support Library`_
for time trace output. Using this library is enabled by default when building
using the CMake option ``OPENMP_ENABLE_LIBOMPTARGET_PROFILING``. The output will
be saved to the filename specified by the environment variable. For multi-threaded
applications, profiling in ``libomp`` is also needed. Setting the CMake option
``OPENMP_ENABLE_LIBOMP_PROFILING=ON`` to enable the feature. Note that this will
turn ``libomp`` into a C++ library.

.. _`Chrome Tracing`: https://www.chromium.org/developers/how-tos/trace-event-profiling-tool

.. _`Speedscope App`: https://www.speedscope.app/

.. _`LLVM Support Library`: https://llvm.org/docs/SupportLibrary.html

LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD
"""""""""""""""""""""""""""""""""""""

``LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD`` sets the threshold size for which the
``libomptarget`` memory manager will handle the allocation. Any allocations
larger than this threshold will not use the memory manager and be freed after
the device kernel exits. The default threshold value is ``8KB``. If
``LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD`` is set to ``0`` the memory manager
will be completely disabled.

LIBOMPTARGET_INFO
"""""""""""""""""

``LIBOMPTARGET_INFO`` allows the user to request different types of runtime
information from ``libomptarget``. ``LIBOMPTARGET_INFO`` uses a 32-bit field to
enable or disable different types of information. This includes information
about data-mappings and kernel execution. It is recommended to build your
application with debugging information enabled, this will enable filenames and
variable declarations in the information messages. OpenMP Debugging information
is enabled at any level of debugging so a full debug runtime is not required.
For minimal debugging information compile with `-gline-tables-only`, or compile
with `-g` for full debug information. A full list of flags supported by
``LIBOMPTARGET_INFO`` is given below. 

    * Print all data arguments upon entering an OpenMP device kernel: ``0x01``
    * Indicate when a mapped address already exists in the device mapping table:
      ``0x02``
    * Dump the contents of the device pointer map at kernel exit: ``0x04``
    * Indicate when an entry is changed in the device mapping table: ``0x08``
    * Print OpenMP kernel information from device plugins: ``0x10``
    * Indicate when data is copied to and from the device: ``0x20``

Any combination of these flags can be used by setting the appropriate bits. For
example, to enable printing all data active in an OpenMP target region along
with ``CUDA`` information, run the following ``bash`` command.

.. code-block:: console

   $ env LIBOMPTARGET_INFO=$((1 << 0x1 | 1 << 0x10)) ./your-application

Or, to enable every flag run with every bit set.

.. code-block:: console

   $ env LIBOMPTARGET_INFO=-1 ./your-application

For example, given a small application implementing the ``ZAXPY`` BLAS routine,
``Libomptarget`` can provide useful information about data mappings and thread
usages.

.. code-block:: c++

    #include <complex>

    using complex = std::complex<double>;

    void zaxpy(complex *X, complex *Y, complex D, std::size_t N) {
    #pragma omp target teams distribute parallel for
      for (std::size_t i = 0; i < N; ++i)
        Y[i] = D * X[i] + Y[i];
    }

    int main() {
      const std::size_t N = 1024;
      complex X[N], Y[N], D;
    #pragma omp target data map(to:X[0 : N]) map(tofrom:Y[0 : N])
      zaxpy(X, Y, D, N);
    }

Compiling this code targeting ``nvptx64`` with all information enabled will
provide the following output from the runtime library.

.. code-block:: console

    $ clang++ -fopenmp -fopenmp-targets=nvptx64 -O3 -gline-tables-only zaxpy.cpp -o zaxpy
    $ env LIBOMPTARGET_INFO=-1 ./zaxpy

.. code-block:: text

    Info: Entering OpenMP data region at zaxpy.cpp:14:1 with 2 arguments:
    Info: to(X[0:N])[16384]
    Info: tofrom(Y[0:N])[16384]
    Info: Creating new map entry with HstPtrBegin=0x00007fff0d259a40,
          TgtPtrBegin=0x00007fdba5800000, Size=16384, RefCount=1, Name=X[0:N]
    Info: Copying data from host to device, HstPtr=0x00007fff0d259a40,
          TgtPtr=0x00007fdba5800000, Size=16384, Name=X[0:N]
    Info: Creating new map entry with HstPtrBegin=0x00007fff0d255a40,
          TgtPtrBegin=0x00007fdba5804000, Size=16384, RefCount=1, Name=Y[0:N]
    Info: Copying data from host to device, HstPtr=0x00007fff0d255a40,
          TgtPtr=0x00007fdba5804000, Size=16384, Name=Y[0:N]
    Info: OpenMP Host-Device pointer mappings after block at zaxpy.cpp:14:1:
    Info: Host Ptr           Target Ptr         Size (B) RefCount Declaration
    Info: 0x00007fff0d255a40 0x00007fdba5804000 16384    1        Y[0:N] at zaxpy.cpp:13:17
    Info: 0x00007fff0d259a40 0x00007fdba5800000 16384    1        X[0:N] at zaxpy.cpp:13:11
    Info: Entering OpenMP kernel at zaxpy.cpp:6:1 with 4 arguments:
    Info: firstprivate(N)[8] (implicit)
    Info: use_address(Y)[0] (implicit)
    Info: tofrom(D)[16] (implicit)
    Info: use_address(X)[0] (implicit)
    Info: Mapping exists (implicit) with HstPtrBegin=0x00007fff0d255a40,
          TgtPtrBegin=0x00007fdba5804000, Size=0, RefCount=2 (incremented), Name=Y
    Info: Creating new map entry with HstPtrBegin=0x00007fff0d2559f0,
          TgtPtrBegin=0x00007fdba5808000, Size=16, RefCount=1, Name=D
    Info: Copying data from host to device, HstPtr=0x00007fff0d2559f0,
          TgtPtr=0x00007fdba5808000, Size=16, Name=D
    Info: Mapping exists (implicit) with HstPtrBegin=0x00007fff0d259a40,
          TgtPtrBegin=0x00007fdba5800000, Size=0, RefCount=2 (incremented), Name=X
    Info: Mapping exists with HstPtrBegin=0x00007fff0d255a40,
          TgtPtrBegin=0x00007fdba5804000, Size=0, RefCount=2 (update suppressed)
    Info: Mapping exists with HstPtrBegin=0x00007fff0d2559f0,
          TgtPtrBegin=0x00007fdba5808000, Size=16, RefCount=1 (update suppressed)
    Info: Mapping exists with HstPtrBegin=0x00007fff0d259a40,
          TgtPtrBegin=0x00007fdba5800000, Size=0, RefCount=2 (update suppressed)
    Info: Launching kernel __omp_offloading_10305_c08c86__Z5zaxpyPSt7complexIdES1_S0_m_l6
          with 8 blocks and 128 threads in SPMD mode
    Info: Mapping exists with HstPtrBegin=0x00007fff0d259a40,
          TgtPtrBegin=0x00007fdba5800000, Size=0, RefCount=1 (decremented)
    Info: Mapping exists with HstPtrBegin=0x00007fff0d2559f0,
          TgtPtrBegin=0x00007fdba5808000, Size=16, RefCount=1 (deferred final decrement)
    Info: Copying data from device to host, TgtPtr=0x00007fdba5808000,
          HstPtr=0x00007fff0d2559f0, Size=16, Name=D
    Info: Mapping exists with HstPtrBegin=0x00007fff0d255a40,
          TgtPtrBegin=0x00007fdba5804000, Size=0, RefCount=1 (decremented)
    Info: Removing map entry with HstPtrBegin=0x00007fff0d2559f0,
          TgtPtrBegin=0x00007fdba5808000, Size=16, Name=D
    Info: OpenMP Host-Device pointer mappings after block at zaxpy.cpp:6:1:
    Info: Host Ptr           Target Ptr         Size (B) RefCount Declaration
    Info: 0x00007fff0d255a40 0x00007fdba5804000 16384    1        Y[0:N] at zaxpy.cpp:13:17
    Info: 0x00007fff0d259a40 0x00007fdba5800000 16384    1        X[0:N] at zaxpy.cpp:13:11
    Info: Exiting OpenMP data region at zaxpy.cpp:14:1 with 2 arguments:
    Info: to(X[0:N])[16384]
    Info: tofrom(Y[0:N])[16384]
    Info: Mapping exists with HstPtrBegin=0x00007fff0d255a40,
          TgtPtrBegin=0x00007fdba5804000, Size=16384, RefCount=1 (deferred final decrement)
    Info: Copying data from device to host, TgtPtr=0x00007fdba5804000,
          HstPtr=0x00007fff0d255a40, Size=16384, Name=Y[0:N]
    Info: Mapping exists with HstPtrBegin=0x00007fff0d259a40,
          TgtPtrBegin=0x00007fdba5800000, Size=16384, RefCount=1 (deferred final decrement)
    Info: Removing map entry with HstPtrBegin=0x00007fff0d255a40,
          TgtPtrBegin=0x00007fdba5804000, Size=16384, Name=Y[0:N]
    Info: Removing map entry with HstPtrBegin=0x00007fff0d259a40,
          TgtPtrBegin=0x00007fdba5800000, Size=16384, Name=X[0:N]

From this information, we can see the OpenMP kernel being launched on the CUDA
device with enough threads and blocks for all ``1024`` iterations of the loop in
simplified :doc:`SPMD Mode <Offloading>`. The information from the OpenMP data
region shows the two arrays ``X`` and ``Y`` being copied from the host to the
device. This creates an entry in the host-device mapping table associating the
host pointers to the newly created device data. The data mappings in the OpenMP
device kernel show the default mappings being used for all the variables used
implicitly on the device. Because ``X`` and ``Y`` are already mapped in the
device's table, no new entries are created. Additionally, the default mapping
shows that ``D`` will be copied back from the device once the OpenMP device
kernel region ends even though it isn't written to. Finally, at the end of the
OpenMP data region the entries for ``X`` and ``Y`` are removed from the table.

The information level can be controlled at runtime using an internal
libomptarget library call ``__tgt_set_info_flag``. This allows for different
levels of information to be enabled or disabled for certain regions of code.
Using this requires declaring the function signature as an external function so
it can be linked with the runtime library.

.. code-block:: c++

    extern "C" void __tgt_set_info_flag(uint32_t);

    extern foo();

    int main() {
      __tgt_set_info_flag(0x10);
    #pragma omp target
      foo();
    }

.. _libopenmptarget_errors:

Errors:
^^^^^^^

``libomptarget`` provides error messages when the program fails inside the
OpenMP target region. Common causes of failure could be an invalid pointer
access, running out of device memory, or trying to offload when the device is
busy. If the application was built with debugging symbols the error messages
will additionally provide the source location of the OpenMP target region.

For example, consider the following code that implements a simple parallel
reduction on the GPU. This code has a bug that causes it to fail in the
offloading region.

.. code-block:: c++

    #include <cstdio>

    double sum(double *A, std::size_t N) {
      double sum = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:sum)
      for (int i = 0; i < N; ++i)
        sum += A[i];
    
      return sum;
    }
    
    int main() {
      const int N = 1024;
      double A[N];
      sum(A, N);
    }

If this code is compiled and run, there will be an error message indicating what is
going wrong.

.. code-block:: console

    $ clang++ -fopenmp -fopenmp-targets=nvptx64 -O3 -gline-tables-only sum.cpp -o sum
    $ ./sum

.. code-block:: text

    CUDA error: an illegal memory access was encountered 
    Libomptarget error: Copying data from device failed.
    Libomptarget error: Call to targetDataEnd failed, abort target.
    Libomptarget error: Failed to process data after launching the kernel.
    Libomptarget error: Run with LIBOMPTARGET_INFO=4 to dump host-target pointer mappings.
    sum.cpp:5:1: Libomptarget error 1: failure of target construct while offloading is mandatory

This shows that there is an illegal memory access occuring inside the OpenMP
target region once execution has moved to the CUDA device, suggesting a
segmentation fault. This then causes a chain reaction of failures in
``libomptarget``. Another message suggests using the ``LIBOMPTARGET_INFO``
environment variable as described in :ref:`libopenmptarget_environment_vars`. If
we do this it will print the sate of the host-target pointer mappings at the
time of failure.

.. code-block:: console

    $ clang++ -fopenmp -fopenmp-targets=nvptx64 -O3 -gline-tables-only sum.cpp -o sum
    $ env LIBOMPTARGET_INFO=4 ./sum

.. code-block:: text

    info: OpenMP Host-Device pointer mappings after block at sum.cpp:5:1:
    info: Host Ptr           Target Ptr         Size (B) RefCount Declaration
    info: 0x00007ffc058280f8 0x00007f4186600000 8        1        sum at sum.cpp:4:10

This tells us that the only data mapped between the host and the device is the
``sum`` variable that will be copied back from the device once the reduction has
ended. There is no entry mapping the host array ``A`` to the device. In this
situation, the compiler cannot determine the size of the array at compile time
so it will simply assume that the pointer is mapped on the device already by
default. The solution is to add an explicit map clause in the target region.

.. code-block:: c++

    double sum(double *A, std::size_t N) {
      double sum = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:sum) map(to:A[0 : N])
      for (int i = 0; i < N; ++i)
        sum += A[i];
    
      return sum;
    }

LIBOMPTARGET_STACK_SIZE
"""""""""""""""""""""""

This environment variable sets the stack size in bytes for the CUDA plugin. This
can be used to increase or decrease the standard amount of memory reserved for
each thread's stack.

LIBOMPTARGET_HEAP_SIZE
"""""""""""""""""""""""

This environment variable sets the amount of memory in bytes that can be
allocated using ``malloc`` and ``free`` for the CUDA plugin. This is necessary
for some applications that allocate too much memory either through the user or
globalization.

LIBOMPTARGET_SHARED_MEMORY_SIZE
"""""""""""""""""""""""""""""""

This environment variable sets the amount of dynamic shared memory in bytes used 
by the kernel once it is launched. A pointer to the dynamic memory buffer can 
currently only be accessed using the ``__kmpc_get_dynamic_shared`` device 
runtime call.

.. toctree::
   :hidden:
   :maxdepth: 1

   Offloading

LLVM/OpenMP Target Host Runtime Plugins (``libomptarget.rtl.XXXX``)
-------------------------------------------------------------------

.. _device_runtime:


.. _remote_offloading_plugin:

Remote Offloading Plugin:
^^^^^^^^^^^^^^^^^^^^^^^^^

The remote offloading plugin permits the execution of OpenMP target regions
on devices in remote hosts in addition to the devices connected to the local
host. All target devices on the remote host will be exposed to the
application as if they were local devices, that is, the remote host CPU or
its GPUs can be offloaded to with the appropriate device number. If the
server is running on the same host, each device may be identified twice:
once through the device plugins and once through the device plugins that the
server application has access to.

This plugin consists of ``libomptarget.rtl.rpc.so`` and
``openmp-offloading-server`` which should be running on the (remote) host. The
server application does not have to be running on a remote host, and can
instead be used on the same host in order to debug memory mapping during offloading.
These are implemented via gRPC/protobuf so these libraries are required to
build and use this plugin. The server must also have access to the necessary
target-specific plugins in order to perform the offloading.

Due to the experimental nature of this plugin, the CMake variable
``LIBOMPTARGET_ENABLE_EXPERIMENTAL_REMOTE_PLUGIN`` must be set in order to
build this plugin. For example, the rpc plugin is not designed to be
thread-safe, the server cannot concurrently handle offloading from multiple
applications at once (it is synchronous) and will terminate after a single
execution. Note that ``openmp-offloading-server`` is unable to
remote offload onto a remote host itself and will error out if this is attempted.

Remote offloading is configured via environment variables at runtime of the OpenMP application:
    * ``LIBOMPTARGET_RPC_ADDRESS=<Address>:<Port>``
    * ``LIBOMPTARGET_RPC_ALLOCATOR_MAX=<NumBytes>``
    * ``LIBOMPTARGET_BLOCK_SIZE=<NumBytes>``
    * ``LIBOMPTARGET_RPC_LATENCY=<Seconds>``

LIBOMPTARGET_RPC_ADDRESS
""""""""""""""""""""""""
The address and port at which the server is running. This needs to be set for
the server and the application, the default is ``0.0.0.0:50051``. A single
OpenMP executable can offload onto multiple remote hosts by setting this to
comma-seperated values of the addresses.

LIBOMPTARGET_RPC_ALLOCATOR_MAX
""""""""""""""""""""""""""""""
After allocating this size, the protobuf allocator will clear. This can be set for both endpoints.

LIBOMPTARGET_BLOCK_SIZE
"""""""""""""""""""""""
This is the maximum size of a single message while streaming data transfers between the two endpoints and can be set for both endpoints.

LIBOMPTARGET_RPC_LATENCY
""""""""""""""""""""""""
This is the maximum amount of time the client will wait for a response from the server.

LLVM/OpenMP Target Device Runtime (``libomptarget-ARCH-SUBARCH.bc``)
--------------------------------------------------------------------

The target device runtime is an LLVM bitcode library that implements OpenMP 
runtime functions on the target device. It is linked with the device code's LLVM 
IR during compilation.

Debugging
^^^^^^^^^

The device runtime supports debugging in the runtime itself. This is configured
at compile-time using the flag ``-fopenmp-target-debug=<N>`` rather than using a
separate debugging build. If debugging is not enabled, the debugging paths will
be considered trivially dead and removed by the compiler with zero overhead.
Debugging is enabled at runtime by running with the environment variable
``LIBOMPTARGET_DEVICE_RTL_DEBUG=<N>`` set. The number set is a 32-bit field used
to selectively enable and disable different features.  Currently, the following
debugging features are supported.

    * Enable debugging assertions in the device. ``0x01``
    * Enable OpenMP runtime function traces in the device. ``0x2``

.. code-block:: c++

    void copy(double *X, double *Y) {
    #pragma omp target teams distribute parallel for
      for (std::size_t i = 0; i < N; ++i)
        Y[i] = X[i];
    }

Compiling this code targeting ``nvptx64`` with debugging enabled will
provide the following output from the device runtime library.

.. code-block:: console

    $ clang++ -fopenmp -fopenmp-targets=nvptx64 -fopenmp-target-new-runtime \
      -fopenmp-target-debug=3
    $ env LIBOMPTARGET_DEVICE_RTL_DEBUG=3 ./zaxpy

.. code-block:: text

    Kernel.cpp:70: Thread 0 Entering int32_t __kmpc_target_init()
    Parallelism.cpp:196: Thread 0 Entering int32_t __kmpc_global_thread_num()
    Mapping.cpp:239: Thread 0 Entering uint32_t __kmpc_get_hardware_num_threads_in_block()
    Workshare.cpp:616: Thread 0 Entering void __kmpc_distribute_static_init_4()
    Parallelism.cpp:85: Thread 0 Entering void __kmpc_parallel_51()
      Parallelism.cpp:69: Thread 0 Entering <OpenMP Outlined Function>
        Workshare.cpp:575: Thread 0 Entering void __kmpc_for_static_init_4()
        Workshare.cpp:660: Thread 0 Entering void __kmpc_distribute_static_fini()
    Workshare.cpp:660: Thread 0 Entering void __kmpc_distribute_static_fini()
    Kernel.cpp:103: Thread 0 Entering void __kmpc_target_deinit()
