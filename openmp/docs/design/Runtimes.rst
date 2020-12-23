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
the device kernel exits The default threshold value is ``8Kb``. If
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

LLVM/OpenMP Target Host Runtime Plugins (``libomptarget.rtl.XXXX``)
-------------------------------------------------------------------

.. _device_runtime:

LLVM/OpenMP Target Device Runtime (``libomptarget-ARCH-SUBARCH.bc``)
--------------------------------------------------------------------

