OpenMP Command-Line Argument Reference
======================================
Welcome to the OpenMP in LLVM command line argument reference. The content is 
not a complete list of arguments but includes the essential command-line 
arguments you may need when compiling and linking OpenMP. 
Section :ref:`general_command_line_arguments` lists OpenMP command line options 
for multicore programming while  :ref:`offload_command_line_arguments` lists 
options relevant to OpenMP target offloading.

.. _general_command_line_arguments:

OpenMP Command-Line Arguments
-----------------------------

``-fopenmp``
^^^^^^^^^^^^
Enable the OpenMP compilation toolchain. The compiler will parse OpenMP 
compiler directives and generate parallel code.

``-fopenmp-extensions``
^^^^^^^^^^^^^^^^^^^^^^^
Enable all ``Clang`` extensions for OpenMP directives and clauses. A list of 
current extensions and their implementation status can be found on the 
`support <https://clang.llvm.org/docs/OpenMPSupport.html#openmp-extensions>`_ 
page.

``-fopenmp-simd``
^^^^^^^^^^^^^^^^^
This option enables OpenMP only for single instruction, multiple data 
(SIMD) constructs.

``-static-openmp``
^^^^^^^^^^^^^^^^^^
Use the static OpenMP host runtime while linking.

``-fopenmp-version=<arg>``
^^^^^^^^^^^^^^^^^^^^^^^^^^
Set the OpenMP version to a specific version ``<arg>`` of the OpenMP standard. 
For example, you may use ``-fopenmp-version=45`` to select version 4.5 of 
the OpenMP standard. The default value is ``-fopenmp-version=51`` for ``Clang``.

.. _offload_command_line_arguments:

Offloading Specific Command-Line Arguments
------------------------------------------

.. _fopenmp-targets:

``-fopenmp-targets``
^^^^^^^^^^^^^^^^^^^^
| Specify which OpenMP offloading targets should be supported. For example, you 
  may specify ``-fopenmp-targets=amdgcn-amd-amdhsa,nvptx64``. This option is 
  often optional when :ref:`offload_arch` is provided.
| It is also possible to offload to CPU architectures, for instance with 
  ``-fopenmp-targets=x86_64-pc-linux-gnu``.

.. _offload_arch:

``--offload-arch``
^^^^^^^^^^^^^^^^^^
| Specify the device architecture for OpenMP offloading. For instance 
  ``--offload-arch=sm_80`` to target an Nvidia Tesla A100, 
  ``--offload-arch=gfx90a`` to target an AMD Instinct MI250X, or 
  ``--offload-arch=sm_80,gfx90a`` to target both.
| It is also possible to specify :ref:`fopenmp-targets` without specifying 
  ``--offload-arch``. In that case, the executables ``amdgpu-arch`` or
  ``nvptx-arch`` will be executed as part of the compiler driver to 
  detect the device architecture automatically.
| Finally, the device architecture will also be automatically inferred with 
  ``--offload-arch=native``.

``--offload-device-only``
^^^^^^^^^^^^^^^^^^^^^^^^^
Compile only the code that goes on the device. This option is mainly for 
debugging purposes. It is primarily used for inspecting the intermediate 
representation (IR) output when compiling for the device. It may also be used 
if device-only runtimes are created.

``--offload-host-only``
^^^^^^^^^^^^^^^^^^^^^^^
Compile only the code that goes on the host. With this option enabled, the
``.llvm.offloading`` section with embedded device code will not be included in 
the intermediate representation.

``--offload-host-device``
^^^^^^^^^^^^^^^^^^^^^^^^^
Compile the target regions for both the host and the device. That is the 
default option.

``-Xopenmp-target <arg>``
^^^^^^^^^^^^^^^^^^^^^^^^^
Pass an argument ``<arg>`` to the offloading toolchain, for instance 
``-Xopenmp-target -march=sm_80``.

``-Xopenmp-target=<triple> <arg>``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pass an argument ``<arg>`` to the offloading toolchain for the target 
``<triple>``. That is especially  useful when an argument must differ for each 
triple. For instance ``-Xopenmp-target=nvptx64 --offload-arch=sm_80 
-Xopenmp-target=amdgcn --offload-arch=gfx90a`` to specify the device 
architecture.  Alternatively, :ref:`Xarch_host` and :ref:`Xarch_device` can 
pass an argument to the host and device compilation toolchain.

``-Xoffload-linker<triple> <arg>``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pass an argument ``<arg>`` to the offloading linker for the target specified in 
``<triple>``.

.. _Xarch_device:

``-Xarch_device <arg>``
^^^^^^^^^^^^^^^^^^^^^^^
Pass an argument ``<arg>`` to the device compilation toolchain.

.. _Xarch_host:

``-Xarch_host <arg>``
^^^^^^^^^^^^^^^^^^^^^
Pass an argument ``<arg>`` to the host compilation toolchain.

``-foffload-lto[=<arg>]``
^^^^^^^^^^^^^^^^^^^^^^^^^
Enable device link time optimization (LTO) and select the LTO mode ``<arg>``. 
Select either ``-foffload-lto=thin`` or ``-foffload-lto=full``. Thin LTO takes 
less time while still achieving some performance gains. If no argument is set, 
this option defaults to ``-foffload-lto=full``. 

``-fopenmp-offload-mandatory``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
| This option is set to avoid generating the host fallback code  
  executed when offloading to the device fails. That is 
  helpful when the target contains code that cannot be compiled for the host, for 
  instance, if it contains unguarded device intrinsics.
| This option can also be used to reduce compile time.
| This option should not be used when one wants to verify that the code is being 
  offloaded to the device. Instead, set the environment variable 
  ``OMP_TARGET_OFFLOAD='MANDATORY'`` to confirm that the code is being offloaded to 
  the device.

``-fopenmp-target-debug[=<arg>]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Enable debugging in the device runtime library (RTL). Note that it is both 
necessary to configure the debugging in the device runtime at compile-time with 
``-fopenmp-target-debug=<arg>`` and enable debugging at runtime with the 
environment  variable ``LIBOMPTARGET_DEVICE_RTL_DEBUG=<arg>``. Further, it is 
currently only supported for Nvidia targets as of July 2023. Alternatively, the 
environment variable ``LIBOMPTARGET_DEBUG`` can be set to debug both Nvidia and 
AMD GPU targets. For more information, see the 
`debugging instructions <https://openmp.llvm.org/design/Runtimes.html#debugging>`_. 
The debugging instructions list the supported debugging arguments.

``-fopenmp-target-jit``
^^^^^^^^^^^^^^^^^^^^^^^
| Emit code that is Just-in-Time (JIT) compiled for OpenMP offloading. Embed 
  LLVM-IR for the device code in the object files rather than binary code for the 
  respective target. At runtime, the LLVM-IR is optimized again and compiled for 
  the target device. The optimization level can be set at runtime with 
  ``LIBOMPTARGET_JIT_OPT_LEVEL``, for instance, 
  ``LIBOMPTARGET_JIT_OPT_LEVEL=3`` corresponding to optimizations level ``-O3``. 
  See the 
  `OpenMP JIT details <https://openmp.llvm.org/design/Runtimes.html#libomptarget-jit-pre-opt-ir-module>`_ 
  for instructions on extracting the embedded device code before or after the 
  JIT and more.
| We want to emphasize that JIT for OpenMP offloading is good for debugging  as 
  the target IR can be extracted, modified, and injected at runtime.

``--offload-new-driver``
^^^^^^^^^^^^^^^^^^^^^^^^
In upstream LLVM, OpenMP only uses the new driver. However, enabling this 
option for experimental linking with CUDA or HIP files is necessary.

``--offload-link``
^^^^^^^^^^^^^^^^^^
Use the new offloading linker `clang-linker-wrapper` to perform the link job. 
`clang-linker-wrapper` is the default offloading linker for OpenMP. This option 
can be used to use the new offloading linker in toolchains that do not automatically 
use it. It is necessary to enable this option when linking with CUDA or HIP files.

``-nogpulib``
^^^^^^^^^^^^^
Do not link the device library for CUDA or HIP device compilation.

``-nogpuinc``
^^^^^^^^^^^^^
Do not include the default CUDA or HIP headers, and do not add CUDA or HIP
include paths.
