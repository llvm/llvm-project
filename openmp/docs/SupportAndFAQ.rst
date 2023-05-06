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
To build an *effective* OpenMP offload capable compiler, only one extra CMake
option, `LLVM_ENABLE_RUNTIMES="openmp"`, is needed when building LLVM (Generic
information about building LLVM is available `here
<https://llvm.org/docs/GettingStarted.html>`__.).  Make sure all backends that
are targeted by OpenMP to be enabled. By default, Clang will be built with all
backends enabled.  When building with `LLVM_ENABLE_RUNTIMES="openmp"` OpenMP
should not be enabled in `LLVM_ENABLE_PROJECTS` because it is enabled by
default.

For Nvidia offload, please see :ref:`build_nvidia_offload_capable_compiler`.
For AMDGPU offload, please see :ref:`build_amdgpu_offload_capable_compiler`.

.. note::
  The compiler that generates the offload code should be the same (version) as
  the compiler that builds the OpenMP device runtimes. The OpenMP host runtime
  can be built by a different compiler.

.. _advanced_builds: https://llvm.org//docs/AdvancedBuilds.html

.. _build_nvidia_offload_capable_compiler:

Q: How to build an OpenMP NVidia offload capable compiler?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Cuda SDK is required on the machine that will execute the openmp application.

If your build machine is not the target machine or automatic detection of the
available GPUs failed, you should also set:

- `LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=YY` where `YY` is the numeric compute capacity of your GPU, e.g., 75.


.. _build_amdgpu_offload_capable_compiler:

Q: How to build an OpenMP AMDGPU offload capable compiler?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A subset of the `ROCm <https://github.com/radeonopencompute>`_ toolchain is
required to build the LLVM toolchain and to execute the openmp application.
Either install ROCm somewhere that cmake's find_package can locate it, or
build the required subcomponents ROCt and ROCr from source.

The two components used are ROCT-Thunk-Interface, roct, and ROCR-Runtime, rocr.
Roct is the userspace part of the linux driver. It calls into the driver which
ships with the linux kernel. It is an implementation detail of Rocr from
OpenMP's perspective. Rocr is an implementation of `HSA
<http://www.hsafoundation.com>`_.

.. code-block:: text

  SOURCE_DIR=same-as-llvm-source # e.g. the checkout of llvm-project, next to openmp
  BUILD_DIR=somewhere
  INSTALL_PREFIX=same-as-llvm-install

  cd $SOURCE_DIR
  git clone git@github.com:RadeonOpenCompute/ROCT-Thunk-Interface.git -b roc-4.2.x \
    --single-branch
  git clone git@github.com:RadeonOpenCompute/ROCR-Runtime.git -b rocm-4.2.x \
    --single-branch

  cd $BUILD_DIR && mkdir roct && cd roct
  cmake $SOURCE_DIR/ROCT-Thunk-Interface/ -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
  make && make install

  cd $BUILD_DIR && mkdir rocr && cd rocr
  cmake $SOURCE_DIR/ROCR-Runtime/src -DIMAGE_SUPPORT=OFF \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON
  make && make install

``IMAGE_SUPPORT`` requires building rocr with clang and is not used by openmp.

Provided cmake's find_package can find the ROCR-Runtime package, LLVM will
build a tool ``bin/amdgpu-arch`` which will print a string like ``gfx906`` when
run if it recognises a GPU on the local system. LLVM will also build a shared
library, libomptarget.rtl.amdgpu.so, which is linked against rocr.

With those libraries installed, then LLVM build and installed, try:

.. code-block:: shell

    clang -O2 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa example.c -o example && ./example

Q: What are the known limitations of OpenMP AMDGPU offload?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LD_LIBRARY_PATH or rpath/runpath are required to find libomp.so and libomptarget.so

There is no libc. That is, malloc and printf do not exist. Libm is implemented in terms
of the rocm device library, which will be searched for if linking with '-lm'.

Some versions of the driver for the radeon vii (gfx906) will error unless the
environment variable 'export HSA_IGNORE_SRAMECC_MISREPORT=1' is set.

It is a recent addition to LLVM and the implementation differs from that which
has been shipping in ROCm and AOMP for some time. Early adopters will encounter
bugs.

Q: What are the LLVM components used in offloading and how are they found?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The libraries used by an executable compiled for target offloading are:

- ``libomp.so`` (or similar), the host openmp runtime
- ``libomptarget.so``, the target-agnostic target offloading openmp runtime
- plugins loaded by libomptarget.so:

  - ``libomptarget.rtl.amdgpu.so``
  - ``libomptarget.rtl.cuda.so``
  - ``libomptarget.rtl.x86_64.so``
  - ``libomptarget.rtl.ve.so``
  - and others

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
time. Otherwise it will attempt to dlopen ``libhsa-runtime64.so.1``. It has rpath
set to ``$ORIGIN``, so installing ``libhsa-runtime64.so.1`` in the same directory is a
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

Q: What is a way to debug errors from mapping memory to a target device?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An experimental way to debug these errors is to use :ref:`remote process
offloading <remote_offloading_plugin>`.
By using ``libomptarget.rtl.rpc.so`` and ``openmp-offloading-server``, it is
possible to explicitly perform memory transfers between processes on the host
CPU and run sanitizers while doing so in order to catch these errors.

Q: Why does my application say "Named symbol not found" and abort when I run it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is most likely caused by trying to use OpenMP offloading with static
libraries. Static libraries do not contain any device code, so when the runtime
attempts to execute the target region it will not be found and you will get an
an error like this.

.. code-block:: text

   CUDA error: Loading '__omp_offloading_fd02_3231c15__Z3foov_l2' Failed
   CUDA error: named symbol not found
   Libomptarget error: Unable to generate entries table for device id 0.

Currently, the only solution is to change how the application is built and avoid
the use of static libraries.

Q: Can I use dynamically linked libraries with OpenMP offloading?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dynamically linked libraries can be only used if there is no device code split
between the library and application. Anything declared on the device inside the
shared library will not be visible to the application when it's linked.

Q: How to build an OpenMP offload capable compiler with an outdated host compiler?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enabling the OpenMP runtime will perform a two-stage build for you.
If your host compiler is different from your system-wide compiler, you may need
to set the CMake variable `GCC_INSTALL_PREFIX` so clang will be able to find the
correct GCC toolchain in the second stage of the build.

For example, if your system-wide GCC installation is too old to build LLVM and
you would like to use a newer GCC, set the CMake variable `GCC_INSTALL_PREFIX`
to inform clang of the GCC installation you would like to use in the second stage.

Q: How can I include OpenMP offloading support in my CMake project?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, there is an experimental CMake find module for OpenMP target
offloading provided by LLVM. It will attempt to find OpenMP target offloading
support for your compiler. The flags necessary for OpenMP target offloading will
be loaded into the ``OpenMPTarget::OpenMPTarget_<device>`` target or the
``OpenMPTarget_<device>_FLAGS`` variable if successful. Currently supported
devices are ``AMDGPU`` and ``NVPTX``.

To use this module, simply add the path to CMake's current module path and call
``find_package``. The module will be installed with your OpenMP installation by
default. Including OpenMP offloading support in an application should now only
require a few additions.

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.20.0)
  project(offloadTest VERSION 1.0 LANGUAGES CXX)

  list(APPEND CMAKE_MODULE_PATH "${PATH_TO_OPENMP_INSTALL}/lib/cmake/openmp")

  find_package(OpenMPTarget REQUIRED NVPTX)

  add_executable(offload)
  target_link_libraries(offload PRIVATE OpenMPTarget::OpenMPTarget_NVPTX)
  target_sources(offload PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp)

Using this module requires at least CMake version 3.20.0. Supported languages
are C and C++ with Fortran support planned in the future. Compiler support is
best for Clang but this module should work for other compiler vendors such as
IBM, GNU.

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
``--offload-arch=`` is present no ``-fopenmp-targets=`` flag is present then the
targets will be inferred from the architectures. Conversely, if
``--fopenmp-targets=`` is present with no ``--offload-arch``  then the target
architecture will be set to a default value, usually the architecture supported
by the system LLVM was built on.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
