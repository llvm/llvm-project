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
be saved to the filename specified by the environment variable.

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
    * Print OpenMP kernel information from device plugins: ``0x10``

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

    Info: Device supports up to 65536 CUDA blocks and 1024 threads with a warp size of 32
    Info: Entering OpenMP data region at zaxpy.cpp:14:1 with 2 arguments:
    Info: to(X[0:N])[16384] 
    Info: tofrom(Y[0:N])[16384] 
    Info: OpenMP Host-Device pointer mappings after block at zaxpy.cpp:14:1:
    Info: Host Ptr           Target Ptr         Size (B) RefCount Declaration
    Info: 0x00007fff963f4000 0x00007fd225004000 16384    1        Y[0:N] at zaxpy.cpp:13:17
    Info: 0x00007fff963f8000 0x00007fd225000000 16384    1        X[0:N] at zaxpy.cpp:13:11
    Info: Entering OpenMP kernel at zaxpy.cpp:6:1 with 4 arguments:
    Info: firstprivate(N)[8] (implicit)
    Info: use_address(Y)[0] (implicit)
    Info: tofrom(D)[16] (implicit)
    Info: use_address(X)[0] (implicit)
    Info: Mapping exists (implicit) with HstPtrBegin=0x00007ffe37d8be80, 
          TgtPtrBegin=0x00007f90ff004000, Size=0, updated RefCount=2, Name=Y
    Info: Mapping exists (implicit) with HstPtrBegin=0x00007ffe37d8fe80, 
          TgtPtrBegin=0x00007f90ff000000, Size=0, updated RefCount=2, Name=X
    Info: Launching kernel __omp_offloading_fd02_c2c4ac1a__Z5daxpyPNSt3__17complexIdEES2_S1_m_l6
          with 8 blocks and 128 threads in SPMD mode
    Info: OpenMP Host-Device pointer mappings after block at zaxpy.cpp:6:1:
    Info: Host Ptr           Target Ptr         Size (B) RefCount Declaration
    Info: 0x00007fff963f4000 0x00007fd225004000 16384    1        Y[0:N] at zaxpy.cpp:13:17
    Info: 0x00007fff963f8000 0x00007fd225000000 16384    1        X[0:N] at zaxpy.cpp:13:11
    Info: Exiting OpenMP data region at zaxpy.cpp:14:1 with 2 arguments:
    Info: to(X[0:N])[16384] 
    Info: tofrom(Y[0:N])[16384] 

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

.. toctree::
   :hidden:
   :maxdepth: 1

   Offloading

LLVM/OpenMP Target Host Runtime Plugins (``libomptarget.rtl.XXXX``)
-------------------------------------------------------------------

.. _device_runtime:

LLVM/OpenMP Target Device Runtime (``libomptarget-ARCH-SUBARCH.bc``)
--------------------------------------------------------------------

