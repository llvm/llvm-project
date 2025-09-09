Support, Getting Involved, and FAQ
==================================

Please do not hesitate to reach out to us on the `Discourse forums (Runtimes - OpenMP) <https://discourse.llvm.org/c/runtimes/openmp/35>`_ or join
one of our :ref:`regular calls <calls>`. Some common questions are answered in
the :ref:`faq`.

.. _calls:

Calls
-----

OpenMP in LLVM Technical Call
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   Development updates on OpenMP (and OpenACC) in the LLVM Project, including Clang, optimization, and runtime work.
-   Join `OpenMP in LLVM Technical Call <https://bluejeans.com/544112769//webrtc>`__.
-   Time: Weekly call on every Wednesday 7:00 AM Pacific time.
-   Meeting minutes are `here <https://docs.google.com/document/d/1Tz8WFN13n7yJ-SCE0Qjqf9LmjGUw0dWO9Ts1ss4YOdg/edit>`__.
-   Status tracking `page <https://openmp.llvm.org/docs>`__.


OpenMP in Flang Technical Call
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
-   Development updates on OpenMP and OpenACC in the Flang Project.
-   Join `OpenMP in Flang Technical Call <https://bit.ly/39eQW3o>`_
-   Time: Weekly call on every Thursdays 8:00 AM Pacific time.
-   Meeting minutes are `here <https://docs.google.com/document/d/1yA-MeJf6RYY-ZXpdol0t7YoDoqtwAyBhFLr5thu5pFI>`__.
-   Status tracking `page <https://docs.google.com/spreadsheets/d/1FvHPuSkGbl4mQZRAwCIndvQx9dQboffiD-xD0oqxgU0/edit#gid=0>`__.


.. _faq:

FAQ
---

.. note::
   The FAQ is a work in progress and most of the expected content is not
   yet available. While you can expect changes, we always welcome feedback and
   additions. Please post on the `Discourse forums (Runtimes - OpenMP) <https://discourse.llvm.org/c/runtimes/openmp/35>`__.


Q: How to contribute a patch to the webpage or any other part?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All patches go through the regular `LLVM review process
<https://llvm.org/docs/Contributing.html#how-to-submit-a-patch>`_.


.. _build_offload_capable_compiler:

Q: How to build an OpenMP GPU offload capable compiler?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to create an offload capable compiler is to use the provided 
CMake cache file. This will enable the projects and runtimes necessary for 
offloading as well as some extra options.

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> cmake ../llvm -G Ninja                                                 \
     -C ../offload/cmake/caches/Offload.cmake \ # The preset cache file
     -DCMAKE_BUILD_TYPE=<Debug|Release>   \ # Select build type
     -DCMAKE_INSTALL_PREFIX=<PATH>        \ # Where the libraries will live
  $> ninja install

To manually build an *effective* OpenMP offload capable compiler, only one extra CMake
option, ``LLVM_ENABLE_RUNTIMES="openmp;offload"``, is needed when building LLVM (Generic
information about building LLVM is available `here
<https://llvm.org/docs/GettingStarted.html>`__.). Make sure all backends that
are targeted by OpenMP are enabled. That can be done by adjusting the CMake 
option ``LLVM_TARGETS_TO_BUILD``. The corresponding targets for offloading to AMD 
and Nvidia GPUs are ``"AMDGPU"`` and ``"NVPTX"``, respectively. By default, 
Clang will be built with all backends enabled. When building with 
``LLVM_ENABLE_RUNTIMES="openmp"`` OpenMP should not be enabled in 
``LLVM_ENABLE_PROJECTS`` because it is enabled by default.

Support for the device library comes from a separate build of the OpenMP library
that targets the GPU architecture. Building it requires enabling the runtime
targets, or setting the target manually when doing a standalone build. This is
done with the ``LLVM_RUNTIME_TARGETS`` option and then enabling the OpenMP
runtime for the GPU target. ``RUNTIMES_<triple>_LLVM_ENABLE_RUNTIMES``. Refer to
the cache file for the specific invocation.

For Nvidia offload, please see :ref:`build_nvidia_offload_capable_compiler`.
For AMDGPU offload, please see :ref:`build_amdgpu_offload_capable_compiler`.

.. note::
  The compiler that generates the offload code should be the same (version) as
  the compiler that builds the OpenMP device runtimes. The OpenMP host runtime
  can be built by a different compiler.

.. _advanced_builds: https://llvm.org//docs/AdvancedBuilds.html

.. _build_nvidia_offload_capable_compiler:

Q: How to build an OpenMP Nvidia offload capable compiler?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The CUDA SDK is required on the machine that will build and execute the
offloading application. Normally this is only required at runtime by dynamically
opening the CUDA driver API. This can be disabled in the build by omitting
``cuda`` from the ``LIBOMPTARGET_DLOPEN_PLUGINS`` list which is present by
default. With this setting we will instead find the CUDA library at LLVM build
time and link against it directly.

.. _build_amdgpu_offload_capable_compiler:

Q: How to build an OpenMP AMDGPU offload capable compiler?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The OpenMP AMDGPU offloading support depends on the ROCm math libraries and the
HSA ROCr / ROCt runtimes. These are normally provided by a standard ROCm
installation, but can be built and used independently if desired. Building the
libraries does not depend on these libraries by default by dynamically loading
the HSA runtime at program execution. As in the CUDA case, this can be change by
omitting ``amdgpu`` from the ``LIBOMPTARGET_DLOPEN_PLUGINS`` list.

Q: What are the known limitations of OpenMP AMDGPU offload?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LD_LIBRARY_PATH or rpath/runpath are required to find libomp.so and
libomptarget.so correctly. The recommended way to configure this is with the
``-frtlib-add-rpath`` option. Alternatively, set the ``LD_LIBRARY_PATH``
environment variable to point to the installation. Normally, these libraries are
installed in the target specific runtime directory. For example, a typical
installation will have
``<install>/lib/x86_64-unknown-linux-gnu/llibomptarget.so``

Some versions of the driver for the radeon vii (gfx906) will error unless the
environment variable 'export HSA_IGNORE_SRAMECC_MISREPORT=1' is set.

Q: What are the LLVM components used in offloading and how are they found?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The libraries used by an executable compiled for target offloading are:

- ``libomp.so`` (or similar), the host openmp runtime
- ``libomptarget.so``, the target-agnostic target offloading openmp runtime
- ``libompdevice.a``, the device-side OpenMP runtime.
- dependencies of those plugins, e.g. cuda/rocr for nvptx/amdgpu

The compiled executable is dynamically linked against a host runtime, e.g.
``libomp.so``, and against the target offloading runtime, ``libomptarget.so``. These
are found like any other dynamic library, by setting rpath or runpath on the
executable, by setting ``LD_LIBRARY_PATH``, or by adding them to the system search.

``libomptarget.so`` is only supported to work with the associated ``clang`` 
compiler. On systems with globally installed ``libomptarget.so`` this can be 
problematic. For this reason it is recommended to use a `Clang configuration 
file <https://clang.llvm.org/docs/UsersManual.html#configuration-files>`__ to 
automatically configure the environment. For example, store the following file 
as ``openmp.cfg`` next to your ``clang`` executable.

.. code-block:: text

  # Library paths for OpenMP offloading.
  -L '<CFGDIR>/../lib'
  -Wl,-rpath='<CFGDIR>/../lib'

The plugins will try to find their dependencies in plugin-dependent fashion.

The cuda plugin is dynamically linked against libcuda if cmake found it at
compiler build time. Otherwise it will attempt to dlopen ``libcuda.so``. It does
not have rpath set.

The amdgpu plugin is linked against ROCr if cmake found it at compiler build
time. Otherwise it will attempt to dlopen ``libhsa-runtime64.so``. It has rpath
set to ``$ORIGIN``, so installing ``libhsa-runtime64.so`` in the same directory is a
way to locate it without environment variables.

In addition to those, there is a compiler runtime library called deviceRTL.
This is compiled from mostly common code into an architecture specific
bitcode library, e.g. ``libomptarget-nvptx-sm_70.bc``.

Clang and the deviceRTL need to match closely as the interface between them
changes frequently. Using both from the same monorepo checkout is strongly
recommended.

Unlike the host side which lets environment variables select components, the
deviceRTL that is located in the clang lib directory is preferred. Only if
it is absent, the ``LIBRARY_PATH`` environment variable is searched to find a
bitcode file with the right name. This can be overridden by passing a clang
flag, ``--libomptarget-nvptx-bc-path`` or ``--libomptarget-amdgcn-bc-path``. That
can specify a directory or an exact bitcode file to use.


Q: Does OpenMP offloading support work in pre-packaged LLVM releases?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For now, the answer is most likely *no*. Please see :ref:`build_offload_capable_compiler`.

Q: Does OpenMP offloading support work in packages distributed as part of my OS?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For now, the answer is most likely *no*. Please see :ref:`build_offload_capable_compiler`.

.. _math_and_complex_in_target_regions:

Q: Does Clang support `<math.h>` and `<complex.h>` operations in OpenMP target on GPUs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, LLVM/Clang allows math functions and complex arithmetic inside of OpenMP
target regions that are compiled for GPUs.

Clang provides a set of wrapper headers that are found first when `math.h` and
`complex.h`, for C, `cmath` and `complex`, for C++, or similar headers are
included by the application. These wrappers will eventually include the system
version of the corresponding header file after setting up a target device
specific environment. The fact that the system header is included is important
because they differ based on the architecture and operating system and may
contain preprocessor, variable, and function definitions that need to be
available in the target region regardless of the targeted device architecture.
However, various functions may require specialized device versions, e.g.,
`sin`, and others are only available on certain devices, e.g., `__umul64hi`. To
provide "native" support for math and complex on the respective architecture,
Clang will wrap the "native" math functions, e.g., as provided by the device
vendor, in an OpenMP begin/end declare variant. These functions will then be
picked up instead of the host versions while host only variables and function
definitions are still available. Complex arithmetic and functions are support
through a similar mechanism. It is worth noting that this support requires
`extensions to the OpenMP begin/end declare variant context selector
<https://clang.llvm.org/docs/AttributeReference.html#pragma-omp-declare-variant>`__
that are exposed through LLVM/Clang to the user as well.

Q: Can I use dynamically linked libraries with OpenMP offloading?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dynamically linked libraries can be used if there is no device code shared
between the library and application. Anything declared on the device inside the
shared library will not be visible to the application when it's linked. This is
because device code only supports static linking.

Q: How to build an OpenMP offload capable compiler with an outdated host compiler?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enabling the OpenMP runtime will perform a two-stage build for you.
If your host compiler is different from your system-wide compiler, you may need
to set ``CMAKE_{C,CXX}_FLAGS`` like
``--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/12`` so that clang will be
able to find the correct GCC toolchain in the second stage of the build.

For example, if your system-wide GCC installation is too old to build LLVM and
you would like to use a newer GCC, set ``--gcc-install-dir=``
to inform clang of the GCC installation you would like to use in the second stage.


Q: What does 'Stack size for entry function cannot be statically determined' mean?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a warning that the Nvidia tools will sometimes emit if the offloading
region is too complex. Normally, the CUDA tools attempt to statically determine
how much stack memory each thread. This way when the kernel is launched each
thread will have as much memory as it needs. If the control flow of the kernel
is too complex, containing recursive calls or nested parallelism, this analysis
can fail. If this warning is triggered it means that the kernel may run out of
stack memory during execution and crash. The environment variable
``LIBOMPTARGET_STACK_SIZE`` can be used to increase the stack size if this
occurs.

Q: Can OpenMP offloading compile for multiple architectures?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since LLVM version 15.0, OpenMP offloading supports offloading to multiple
architectures at once. This allows for executables to be run on different
targets, such as offloading to AMD and NVIDIA GPUs simultaneously, as well as
multiple sub-architectures for the same target. Additionally, static libraries
will only extract archive members if an architecture is used, allowing users to
create generic libraries.

The architecture can either be specified manually using ``--offload-arch=``. If
``--offload-arch=`` is present and no ``-fopenmp-targets=`` flag is present then
the targets will be inferred from the architectures. Conversely, if
``--fopenmp-targets=`` is present with no ``--offload-arch`` then the target
architecture will be set to a default value, usually the architecture supported
by the system LLVM was built on by executing the ``offload-arch`` utility.

For example, an executable can be built that runs on AMDGPU and NVIDIA hardware
given that the necessary build tools are installed for both.

.. code-block:: shell

   clang example.c -fopenmp --offload-arch=gfx90a --offload-arch=sm_80

If just given the architectures we should be able to infer the triples,
otherwise we can specify them manually.

.. code-block:: shell

   clang example.c -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda \
      -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx90a \
      -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_80

When linking against a static library that contains device code for multiple
architectures, only the images used by the executable will be extracted.

.. code-block:: shell

   clang example.c -fopenmp --offload-arch=gfx90a,gfx90a,sm_70,sm_80 -c
   llvm-ar rcs libexample.a example.o
   clang app.c -fopenmp --offload-arch=gfx90a -o app

The supported device images can be viewed using the ``--offloading`` option with
``llvm-objdump``.

.. code-block:: shell

   clang example.c -fopenmp --offload-arch=gfx90a --offload-arch=sm_80 -o example
   llvm-objdump --offloading example

   a.out:  file format elf64-x86-64

   OFFLOADING IMAGE [0]:
   kind            elf
   arch            gfx90a
   triple          amdgcn-amd-amdhsa
   producer        openmp

   OFFLOADING IMAGE [1]:
   kind            elf
   arch            sm_80
   triple          nvptx64-nvidia-cuda
   producer        openmp

Q: Can I link OpenMP offloading with CUDA or HIP?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenMP offloading files can currently be experimentally linked with CUDA and HIP
files. This will allow OpenMP to call a CUDA device function or vice-versa.
However, the global state will be distinct between the two images at runtime.
This means any global variables will potentially have different values when
queried from OpenMP or CUDA.

Linking CUDA and HIP currently requires enabling a different compilation mode
for CUDA / HIP with ``--offload-new-driver`` and to link using
``--offload-link``. Additionally, ``-fgpu-rdc`` must be used to create a
linkable device image.

.. code-block:: shell

   clang++ openmp.cpp -fopenmp --offload-arch=sm_80 -c
   clang++ cuda.cu --offload-new-driver --offload-arch=sm_80 -fgpu-rdc -c
   clang++ openmp.o cuda.o --offload-link -o app

Q: Are libomptarget and plugins backward compatible?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No. libomptarget and plugins are now built as LLVM libraries starting from LLVM
15. Because LLVM libraries are not backward compatible, libomptarget and plugins
are not as well. Given that fact, the interfaces between 1) the Clang compiler
and libomptarget, 2) the Clang compiler and device runtime library, and
3) libomptarget and plugins are not guaranteed to be compatible with an earlier
version. Users are responsible for ensuring compatibility when not using the
Clang compiler and runtime libraries from the same build. Nevertheless, in order
to better support third-party libraries and toolchains that depend on existing
libomptarget entry points, contributors are discouraged from making
modifications to them.

Q: Can I use libc functions on the GPU?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LLVM provides basic ``libc`` functionality through the LLVM C Library. For 
building instructions, refer to the associated `LLVM libc documentation 
<https://libc.llvm.org/gpu/using.html#building-the-gpu-library>`_. Once built, 
this provides a static library called ``libcgpu.a``. See the documentation for a 
list of `supported functions <https://libc.llvm.org/gpu/support.html>`_ as well. 
To utilize these functions, simply link this library as any other when building 
with OpenMP.

.. code-block:: shell

   clang++ openmp.cpp -fopenmp --offload-arch=gfx90a -Xoffload-linker -lc

For more information on how this is implemented in LLVM/OpenMP's offloading 
runtime, refer to the `runtime documentation <libomptarget_libc>`_.

Q: What command line options can I use for OpenMP?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We recommend taking a look at the OpenMP 
:doc:`command line argument reference <CommandLineArgumentReference>` page.

Q: Can I build the offloading runtimes without CUDA or HSA?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, the offloading runtime will load the associated vendor runtime 
during initialization rather than directly linking against them. This allows the 
program to be built and run on many machine. If you wish to directly link 
against these libraries, use the ``LIBOMPTARGET_DLOPEN_PLUGINS=""`` option to 
suppress it for each plugin. The default value is every plugin enabled with 
``LIBOMPTARGET_PLUGINS_TO_BUILD``.

Q: Why is my build taking a long time?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When installing OpenMP and other LLVM components, the build time on multicore 
systems can be significantly reduced with parallel build jobs. As suggested in 
*LLVM Techniques, Tips, and Best Practices*, one could consider using ``ninja`` as the
generator. This can be done with the CMake option ``cmake -G Ninja``. Afterward, 
use ``ninja install`` and specify the number of parallel jobs with ``-j``. The build
time can also be reduced by setting the build type to ``Release`` with the 
``CMAKE_BUILD_TYPE`` option. Recompilation can also be sped up by caching previous
compilations. Consider enabling ``Ccache`` with 
``CMAKE_CXX_COMPILER_LAUNCHER=ccache``.

Q: Did this FAQ not answer your question?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Feel free to post questions or browse old threads at 
`LLVM Discourse <https://discourse.llvm.org/c/runtimes/openmp/>`__.
