Support, Getting Involved, and FAQ
==================================

Please do not hesitate to reach out to us via openmp-dev@lists.llvm.org or join
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
   additions. Please contact, e.g., through ``openmp-dev@lists.llvm.org``.


Q: How to contribute a patch to the webpage or any other part?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All patches go through the regular `LLVM review process
<https://llvm.org/docs/Contributing.html#how-to-submit-a-patch>`_.


.. _build_offload_capable_compiler:

Q: How to build an OpenMP offload capable compiler?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build an *effective* OpenMP offload capable compiler we recommend a two
stage build. The first stage Clang does not require to be offload capable but
all backends that are targeted by OpenMP need to be enabled. By default, Clang
will be build with all backends enabled. This initial (stage 1) Clang is used
to create a second Clang compiler that is offload capable as well as the
:ref:`device runtime libraries <device_runtime>` that will be linked into the
offloaded code to provide OpenMP runtime support on the device.

Generic information about building LLVM is available `here
<https://llvm.org/docs/GettingStarted.html>`__. The CMake options for the
second stage Clang should include:

- `LIBOMPTARGET_NVPTX_CUDA_COMPILER=$STAGE1/bin/clang` to use the stage one
  compiler for the device runtime compilation.
- `LIBOMPTARGET_NVPTX_ENABLE_BCLIB=ON` to enable efficient device runtimes in
  bitcode format.

If your build machine is not the target machine or automatic detection of the
available GPUs failed, you should also set:

- `CLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_XX` where `XX` is the architecture of your GPU, e.g, 80.
- `LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=YY` where `YY` is the numeric compute capacity of your GPU, e.g., 75.

.. note::
  The compiler that generates the offload code should be the same (version) as
  the compiler that build the OpenMP device runtimes. The OpenMP host runtime
  can be build by a different compiler.

.. _advanced_builds: https://llvm.org//docs/AdvancedBuilds.html



Q: Does OpenMP offloading support work in pre-packaged LLVM releases?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For now, the answer is most likely *no*. Please see :ref:`build_offload_capable_compiler`.

Q: Does OpenMP offloading support work in packages distributed as part of my OS?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For now, the answer is most likely *no*. Please see :ref:`build_offload_capable_compiler`.
